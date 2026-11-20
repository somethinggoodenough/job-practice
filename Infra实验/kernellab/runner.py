import hashlib
import importlib.metadata
import inspect
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch


def metadata(device):
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device)}
    for package in ("triton", "triton-windows"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, memory_bytes=props.total_memory,
                    compute_capability=list(torch.cuda.get_device_capability(device)))
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    info["torch_threads"] = torch.get_num_threads()
    info["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    cuda_backend = sys.modules.get("kernellab.cuda")
    if cuda_backend and cuda_backend.apis.cache_info().currsize:
        info["cuda_compiler"] = cuda_backend.apis().info()
    return info


def source_hash(fn):
    try:
        path = inspect.getsourcefile(fn)
    except TypeError:
        return "builtin"
    if path:
        digest = hashlib.sha256()
        for source_path in (path, *getattr(fn, "kernel_sources", ())):
            with open(source_path, "rb") as source:
                digest.update(source.read())
        return digest.hexdigest()[:16]
    return "builtin"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inputs_for(spec, case, dtype, device, seed, pattern="random"):
    generator = torch.Generator(device=device).manual_seed(seed)
    return spec.make_inputs(case, dtype, device, generator, pattern)


@torch.inference_mode()
def validate(spec, fn, case, dtype, device, seed):
    """Check corner patterns, input immutability and repeated calls before timing."""
    rtol, atol = spec.tolerances[str(dtype).split(".")[-1]]
    max_error = 0.0
    for pattern in spec.patterns:
        args = inputs_for(spec, case, dtype, device, seed, pattern)
        saved = tuple(x.clone() for x in args)
        expected = spec.reference(*tuple(x.clone() for x in args))
        # Repeated calls should remain correct and leave inputs intact.
        for _ in range(2):
            actual = fn(*args)
            sync(device)
            if not isinstance(actual, torch.Tensor):
                raise AssertionError("The implementation must return a torch.Tensor")
            if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
                raise AssertionError(f"Output contract mismatch: {actual.shape}/{actual.dtype}/{actual.device}")
            if not torch.isfinite(actual).all().item():
                raise AssertionError(f"Non-finite output for pattern={pattern}")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                       msg=lambda msg: f"pattern={pattern}: {msg}")
            for before, after in zip(saved, args):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            max_error = max(max_error, (actual.float() - expected.float()).abs().max().item())
    # One additional random input, with a different seed.
    args = inputs_for(spec, case, dtype, device, seed + 1)
    saved = tuple(x.clone() for x in args)
    expected = spec.reference(*tuple(x.clone() for x in args))
    actual = fn(*args)
    sync(device)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for before, after in zip(saved, args):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    return max(max_error, (actual.float() - expected.float()).abs().max().item())


@torch.inference_mode()
def measure(fn, args, device, warmup, samples, iterations, timer="events"):
    if timer == "cuda-graph" and device.type != "cuda":
        raise ValueError("CUDA Graph timing requires a CUDA device")
    # Compilation/autotuning happen before timing, with identical inputs.
    for _ in range(warmup):
        fn(*args)
    sync(device)
    values = []
    if device.type == "cuda":
        with torch.cuda.device(device):
            graph = None
            if timer == "cuda-graph":
                # Capture repeated launches so Python/ctypes launch overhead is
                # outside timed replay. Allocations use PyTorch's graph pool.
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream(device=device)
                capture_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.graph(graph, stream=capture_stream):
                    for _ in range(iterations):
                        output = fn(*args)
                torch.cuda.current_stream(device).wait_stream(capture_stream)
                for _ in range(3):
                    graph.replay()
                sync(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(samples):
                start.record()
                if graph is not None:
                    graph.replay()
                else:
                    for _ in range(iterations):
                        fn(*args)
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) * 1000 / iterations)
    else:
        for _ in range(samples):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                fn(*args)
            values.append((time.perf_counter_ns() - start) / 1000 / iterations)
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {"median_us": statistics.median(values), "p25_us": q1, "p75_us": q3,
            "samples_us": values}


@torch.inference_mode()
def profile_cuda(fn, args, device, warmup, repeat, trace_path):
    """Profile CUDA work after compilation/warmup and save a Chrome trace."""
    if device.type != "cuda":
        raise ValueError("CUDA profiling requires a CUDA device")
    if warmup < 1 or repeat < 1:
        raise ValueError("Profiler warmup and repeat must be positive")
    with torch.cuda.device(device):
        output = None
        for _ in range(warmup):
            output = fn(*args)
        sync(device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(repeat):
                output = fn(*args)
            sync(device)
        # Keep the final output alive through synchronization and trace export.
        _ = output
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Kineto on Windows cannot open a trace path containing non-ASCII text.
    # Export through the system temp directory, then let Python move it back.
    temporary_trace = Path(tempfile.gettempdir()) / f"kernellab-{uuid.uuid4().hex}.json"
    prof.export_chrome_trace(str(temporary_trace))
    if not temporary_trace.is_file():
        raise RuntimeError(f"Profiler did not create its Chrome trace: {temporary_trace}")
    temporary_trace.replace(trace_path)
    return prof
