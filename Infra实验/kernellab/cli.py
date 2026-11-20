import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

# Keep JIT caches local, including on Windows installations with Chinese paths.
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
# Triton 3.5's C++ getenv decodes Windows ANSI bytes as UTF-8. Set the
# Python knob directly so a workspace path containing Chinese remains valid.
if os.name == "nt" and importlib.util.find_spec("triton") is not None:
    from triton import knobs
    knobs.cache.dir = os.environ["TRITON_CACHE_DIR"]

import torch

from kernellab.registry import discover
from kernellab.runner import inputs_for, measure, metadata, profile_cuda, source_hash, validate


def parser():
    p = argparse.ArgumentParser(description="Kernel practice lab (forward only)")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check Python, CUDA and optionally compile a copy kernel")
    doctor.add_argument("--smoke", action="store_true", help="Compile and run the independent Triton copy demo")
    doctor.add_argument("--cuda-smoke", action="store_true", help="Compile a real .cu file using NVIDIA NVRTC and launch it")
    sub.add_parser("list", help="List exercises and registered implementations")
    for command in ("check", "bench"):
        s = sub.add_parser(command)
        s.add_argument("op")
        s.add_argument("--impl", default="mine", help="Implementation name, or all")
        s.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
        s.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "all"], default="float16")
        s.add_argument("--case", default="all", help="Comma separated case names (see list)")
        s.add_argument("--seed", type=int, default=42)
        if command == "bench":
            s.add_argument("--baseline-only", action="store_true", help="Measure only PyTorch while the exercise is empty")
            s.add_argument("--warmup", type=int, default=20)
            s.add_argument("--samples", type=int, default=15)
            s.add_argument("--iterations", type=int, default=50)
            s.add_argument("--timer", choices=["events", "cuda-graph"], default="events", help="cuda-graph removes Python launch overhead from repeated GPU work")
            s.add_argument("--tag", default="experiment")
            s.add_argument("--output", type=Path, default=ROOT / "results")
    s = sub.add_parser("profile", help="Compare one implementation with its PyTorch baseline using torch.profiler")
    s.add_argument("op")
    s.add_argument("--impl", default="cuda")
    s.add_argument("--device", default="cuda")
    s.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    s.add_argument("--case", default="largest", help="One case name, or largest (the final registered case)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--repeat", type=int, default=1, help="Calls recorded in each separate profiler session")
    s.add_argument("--row-limit", type=int, default=10)
    s.add_argument("--skip-check", action="store_true", help="Skip correctness validation before profiling")
    s.add_argument("--output", type=Path, default=ROOT / "results" / "profiles")
    s = sub.add_parser("report", help="Merge one or more benchmark JSON files into an offline HTML report")
    s.add_argument("inputs", nargs="+", help="JSON paths or glob patterns")
    s.add_argument("--output", type=Path, default=ROOT / "results" / "comparison.html")
    return p


def resolve_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CPU and CUDA timing are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Run: python -m kernellab doctor")
        # Canonicalize cuda to cuda:0 for output comparisons and metadata.
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def doctor(smoke, cuda_smoke=False):
    device = resolve_device("auto")
    print(json.dumps(metadata(device), indent=2, ensure_ascii=False))
    available = importlib.util.find_spec("triton") is not None
    print(f"Triton installed: {available}")
    if device.type != "cuda":
        print("CPU only: framework works; GPU kernel performance cannot be measured.")
        return 1
    if smoke:
        if not available:
            print("Install matching Triton first: python scripts/setup.py")
            return 1
        from kernellab.demo_copy import SPEC, triton_copy
        validate(SPEC, triton_copy, SPEC.cases[0], torch.float32, device, 42)
        print("PASS: Triton JIT compile, GPU copy, odd-size masking, and correctness.")
    if cuda_smoke:
        from kernellab.cuda import apis
        from kernellab.demo_copy import SPEC, cuda_copy
        print(json.dumps(apis().info(), indent=2, ensure_ascii=False))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            validate(SPEC, cuda_copy, SPEC.cases[0], dtype, device, 42)
        print("PASS: CUDA C++ .cu -> NVIDIA NVRTC -> PTX -> GPU; all three dtypes.")
    return 0


def selections(args, specs):
    if args.op not in specs:
        raise ValueError(f"Unknown exercise {args.op}; choose {', '.join(specs)}")
    spec = specs[args.op]
    requested = args.case.split(",")
    cases = spec.cases if args.case == "all" else [c for c in spec.cases if c.name in requested]
    if args.case != "all" and set(requested) - {c.name for c in cases}:
        raise ValueError(f"Unknown cases. Valid: {', '.join(c.name for c in spec.cases)}")
    if getattr(args, "baseline_only", False):
        implementations = {}
    elif args.impl == "all":
        implementations = spec.implementations
    else:
        if args.impl not in spec.implementations:
            raise ValueError(f"Unknown implementation {args.impl}; choose {', '.join(spec.implementations)}")
        implementations = {args.impl: spec.implementations[args.impl]}
    dtypes = ["float32", "float16", "bfloat16"] if args.dtype == "all" else [args.dtype]
    return spec, cases, implementations, dtypes


def print_table(rows):
    print(f"{'case':>16} {'dtype':>9} {'implementation':>15} {'status':>14} {'median us':>11} {'speedup':>9} {'GB/s*':>10}")
    for r in rows:
        latency = f"{r['median_us']:.3f}" if "median_us" in r else "-"
        speedup = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
        bandwidth = f"{r['gb_s']:.2f}" if "gb_s" in r else "-"
        print(f"{r['case']:>16} {r['dtype']:>9} {r['implementation']:>15} {r['status']:>14} {latency:>11} {speedup:>9} {bandwidth:>10}")
        if "error" in r:
            print("  " + r["error"].replace("\n", " ")[:350])


def write_results(payload, directory):
    from kernellab.report import write_report
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = directory / f"{payload['op']}-{stamp}"
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["op", "case", "dtype", "implementation", "status", "median_us", "p25_us", "p75_us", "speedup", "gb_s", "max_abs_error", "error"]
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["rows"])
    write_report([payload], base.with_suffix(".html"))
    print(f"\nJSON: {json_path}\nCSV:  {base.with_suffix('.csv')}\nHTML: {base.with_suffix('.html')}")


def run_profile(args, specs):
    if args.op not in specs:
        raise ValueError(f"Unknown exercise {args.op}; choose {', '.join(specs)}")
    spec = specs[args.op]
    if args.impl not in spec.implementations:
        raise ValueError(f"Unknown implementation {args.impl}; choose {', '.join(spec.implementations)}")
    if args.case == "largest":
        case = spec.cases[-1]
    else:
        matches = [candidate for candidate in spec.cases if candidate.name == args.case]
        if not matches:
            raise ValueError(f"Unknown case {args.case}. Valid: {', '.join(c.name for c in spec.cases)}")
        case = matches[0]
    if args.warmup < 1 or args.repeat < 1 or args.row_limit < 1:
        raise ValueError("Require warmup>=1, repeat>=1, row-limit>=1")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("profile requires a CUDA device")
    dtype = getattr(torch, args.dtype)
    fn = spec.implementations[args.impl]
    print(f"{spec.name} | {case.name} | {args.dtype} | {device}", flush=True)
    if not args.skip_check:
        error = validate(spec, fn, case, dtype, device, args.seed)
        print(f"correctness: pass (max abs error {error:.6g})", flush=True)
    inputs = inputs_for(spec, case, dtype, device, args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = args.output / f"{spec.name}-{case.name}-{args.dtype}-{stamp}"
    candidate_trace = base.with_name(base.name + f"-{args.impl}.json")
    baseline_trace = base.with_name(base.name + "-pytorch.json")

    print(f"\n=== {args.impl}: {args.repeat} profiled call(s) ===", flush=True)
    candidate = profile_cuda(fn, inputs, device, args.warmup, args.repeat, candidate_trace)
    candidate_table = candidate.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)
    print(candidate_table)

    print(f"\n=== pytorch baseline: {args.repeat} profiled call(s) ===", flush=True)
    baseline = profile_cuda(spec.reference, inputs, device, args.warmup, args.repeat, baseline_trace)
    baseline_table = baseline.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)
    print(baseline_table)

    text_path = base.with_suffix(".txt")
    text_path.write_text(
        f"operation: {spec.name}\ncase: {case.name}\ndtype: {args.dtype}\ndevice: {device}\n"
        f"warmup: {args.warmup}\nprofiled calls per session: {args.repeat}\n\n"
        f"=== {args.impl} ===\n{candidate_table}\n\n=== pytorch ===\n{baseline_table}\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {text_path}\n{args.impl} trace: {candidate_trace}\nPyTorch trace: {baseline_trace}")
    print("CUDA totals cover all profiled calls; divide totals by --repeat for per-call values.")
    return 0


def run(args):
    if args.command == "report":
        import glob
        from kernellab.report import write_report
        paths = sorted({p for pattern in args.inputs for p in glob.glob(pattern)})
        if not paths:
            raise ValueError("No matching JSON results")
        payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        write_report(payloads, args.output)
        print(f"Report: {args.output.resolve()}")
        return 0
    torch.set_num_threads(1)
    # Avoid machine-dependent TF32 settings for exercise correctness.
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.command == "doctor":
        return doctor(args.smoke, args.cuda_smoke)
    specs = discover()
    if args.command == "list":
        for spec in specs.values():
            print(f"{spec.name}: {spec.description}")
            print(f"  implementations: {', '.join(spec.implementations)}")
            print(f"  cases: {', '.join(c.name for c in spec.cases)}")
        return 0
    if args.command == "profile":
        return run_profile(args, specs)
    if args.command == "bench" and (args.warmup < 1 or args.samples < 4 or args.iterations < 1):
        raise ValueError("Require warmup>=1, samples>=4, iterations>=1")
    device = resolve_device(args.device)
    if args.command == "bench" and args.timer == "cuda-graph" and device.type != "cuda":
        raise ValueError("--timer cuda-graph requires a CUDA device")
    spec, cases, implementations, dtypes = selections(args, specs)
    print(f"{spec.name} | {device} | seed={args.seed}", flush=True)
    if device.type == "cpu":
        print("CPU timing: these numbers do not represent GPU kernel performance.")
    rows = []
    failed = False
    # Complete validation of every requested case before starting the timed phase.
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for case in cases:
            for name, fn in implementations.items():
                row = {"op": spec.name, "case": case.name, "dtype": dtype_name, "implementation": name}
                try:
                    row["max_abs_error"] = validate(spec, fn, case, dtype, device, args.seed)
                    row["source_hash"] = source_hash(fn)
                    row["status"] = "pass"
                except NotImplementedError as e:
                    row.update(status="unimplemented", error=str(e))
                    failed = True
                except Exception as e:
                    row.update(status="failed", error=f"{type(e).__name__}: {e}")
                    failed = True
                rows.append(row)
                print(f"  {row['status']}: {case.name} / {dtype_name} / {name}", flush=True)
    if args.command == "check":
        print_table(rows)
        return int(failed)
    # PyTorch-only results remain explicitly labelled as a baseline.
    timing_errors = False
    with torch.inference_mode():
        for dtype_name in dtypes:
            dtype = getattr(torch, dtype_name)
            for case in cases:
                inputs = inputs_for(spec, case, dtype, device, args.seed)
                output = spec.reference(*inputs)
                nbytes = spec.logical_bytes(inputs, output)
                del output
                baseline = {"op": spec.name, "case": case.name, "dtype": dtype_name,
                            "implementation": "pytorch", "status": "baseline"}
                try:
                    baseline.update(measure(spec.reference, inputs, device, args.warmup, args.samples, args.iterations, args.timer))
                    baseline.update(speedup=1.0, gb_s=nbytes / (baseline["median_us"] * 1000))
                    rows.append(baseline)
                    for row in rows:
                        if row["case"] == case.name and row["dtype"] == dtype_name and row["status"] == "pass":
                            fn = implementations[row["implementation"]]
                            row.update(measure(fn, inputs, device, args.warmup, args.samples, args.iterations, args.timer))
                            row.update(speedup=baseline["median_us"] / row["median_us"],
                                       gb_s=nbytes / (row["median_us"] * 1000))
                except Exception as e:
                    timing_errors = True
                    rows.append({"op": spec.name, "case": case.name, "dtype": dtype_name,
                                 "implementation": "timing", "status": "failed", "error": str(e)})
    print_table(rows)
    print("* GB/s = logical input+output bytes / time; not measured DRAM bandwidth.")
    payload = {"schema_version": 1, "op": spec.name, "tag": args.tag,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "environment": metadata(device),
               "config": {"seed": args.seed, "warmup": args.warmup, "samples": args.samples,
                          "iterations": args.iterations, "baseline_only": args.baseline_only,
                          "timer": ("CUDA Graph replay + CUDA events" if args.timer == "cuda-graph" else "CUDA events, batched launches") if device.type == "cuda" else "perf_counter_ns",
                          "input_policy": "same resident inputs, warm cache; " + ("graph-pool outputs allocated during capture, outside replay" if args.timer == "cuda-graph" else "output allocations included"),
                          "tolerances": spec.tolerances},
               "rows": rows}
    write_results(payload, args.output)
    return int(failed or timing_errors)


def main():
    args = parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
