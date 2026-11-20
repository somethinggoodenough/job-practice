import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from exercises.vector_add import SPEC
from kernellab.cli import parser, run
from kernellab.report import write_report
from kernellab.runner import measure, source_hash, validate
from kernellab.spec import Case


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.case = Case("257", ((257,), (257,)))
        self.device = torch.device("cpu")

    def test_correct_callable_and_builtin_hash(self):
        self.assertEqual(validate(SPEC, torch.add, self.case, torch.float32, self.device, 42), 0)
        self.assertEqual(source_hash(torch.add), "builtin")

    def test_wrong_results_rejected(self):
        with self.assertRaises(AssertionError):
            validate(SPEC, lambda a, b: a - b, self.case, torch.float32, self.device, 42)

    def test_input_mutation_rejected(self):
        def mutating(a, b):
            return a.add_(b)
        with self.assertRaises(AssertionError):
            validate(SPEC, mutating, self.case, torch.float32, self.device, 42)

    def test_wrong_dtype_and_nonfinite_rejected(self):
        for fn in (lambda a, b: (a + b).double(), lambda a, b: torch.full_like(a, float("nan"))):
            with self.assertRaises(AssertionError):
                validate(SPEC, fn, self.case, torch.float32, self.device, 42)

    def test_cpu_timer_returns_real_samples(self):
        x = torch.ones(257)
        result = measure(torch.add, (x, x), self.device, 1, 4, 5)
        self.assertEqual(len(result["samples_us"]), 4)
        self.assertGreater(result["median_us"], 0)
        self.assertLessEqual(result["p25_us"], result["p75_us"])

    def test_empty_exercise_fails_and_is_never_timed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(["bench", "vector_add", "--device", "cpu", "--dtype", "float32",
                "--case", "257", "--samples", "4", "--iterations", "1", "--warmup", "1", "--output", directory])
            from kernellab.runner import measure as real_measure
            with contextlib.redirect_stdout(io.StringIO()), patch("kernellab.cli.measure", wraps=real_measure) as timer:
                self.assertEqual(run(args), 1)
            self.assertEqual(timer.call_count, 1)  # Only the PyTorch baseline.
            payload = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
            row = next(r for r in payload["rows"] if r["implementation"] == "mine")
            self.assertEqual(row["status"], "unimplemented")
            self.assertNotIn("median_us", row)

    def test_html_escapes_embedded_script_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            write_report([{"tag": "</script><script>alert(1)</script>"}], path)
            html = path.read_text(encoding="utf-8")
            self.assertNotIn("</script><script>alert", html)
            self.assertIn("\\u003c", html)

    def test_unknown_case_rejected(self):
        args = parser().parse_args(["check", "vector_add", "--case", "typo", "--device", "cpu"])
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            run(args)

    def test_profile_parser_defaults_to_largest_cuda_case(self):
        args = parser().parse_args(["profile", "rms_norm"])
        self.assertEqual(args.impl, "cuda")
        self.assertEqual(args.case, "largest")
        self.assertEqual(args.dtype, "float32")


if __name__ == "__main__":
    unittest.main()
