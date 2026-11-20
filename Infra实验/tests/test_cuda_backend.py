import ctypes as C
from pathlib import Path
import tempfile
import unittest

import torch

from kernellab.cuda import CUDAKernel, _dim3, get_kernel
from kernellab.demo_copy import CUDA_COPY_SOURCE, SPEC, cuda_copy
from kernellab.runner import measure, source_hash, validate


class CUDALaunchContractTests(unittest.TestCase):
    def test_dimensions_reject_invalid_launches(self):
        for dims in (0, -1, (1, 0), (1, 2, 3, 4), (1.5,)):
            with self.assertRaises(ValueError):
                _dim3(dims)

    def test_cuda_source_changes_fingerprint(self):
        def wrapper():
            pass
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.cu"
            source.write_text("// v1", encoding="utf-8")
            wrapper.kernel_sources = (source,)
            first = source_hash(wrapper)
            source.write_text("// v2", encoding="utf-8")
            self.assertNotEqual(first, source_hash(wrapper))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class CUDABackendTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_actual_cuda_copy_all_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.assertEqual(validate(SPEC, cuda_copy, SPEC.cases[0], dtype, self.device, 123), 0)

    def test_current_nondefault_stream(self):
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            x = torch.randn(65537, device=self.device)
            y = cuda_copy(x)
            expected = x.clone()
        stream.synchronize()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    def test_invalid_scalar_rejected_before_launch(self):
        kernel = get_kernel(CUDA_COPY_SOURCE, "copy_bytes", self.device.index)
        x = torch.ones(16, device=self.device)
        with self.assertRaises(TypeError):
            kernel.launch((x, x, 64), grid=1)
        with self.assertRaises(ValueError):
            kernel.launch((x[::2], x, C.c_uint64(32)), grid=1)

    def test_compile_error_includes_source_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.cu"
            source.write_text('#error "deliberate-compile-failure"\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "deliberate-compile-failure"):
                CUDAKernel(source, "unused", self.device)

    def test_cuda_graph_captures_driver_launches(self):
        x = torch.randn(257, device=self.device)
        result = measure(cuda_copy, (x,), self.device, 2, 4, 5, "cuda-graph")
        self.assertEqual(len(result["samples_us"]), 4)
        self.assertGreater(result["median_us"], 0)

    def test_profiler_exports_trace_to_unicode_workspace_path(self):
        from kernellab.runner import profile_cuda
        x = torch.randn(257, device=self.device)
        with tempfile.TemporaryDirectory(dir=".", prefix="分析-") as directory:
            trace = Path(directory) / "轨迹.json"
            prof = profile_cuda(cuda_copy, (x,), self.device, 1, 1, trace)
            self.assertTrue(trace.is_file())
            self.assertGreater(trace.stat().st_size, 0)
            self.assertIn("cuda_copy", getattr(cuda_copy, "__name__", ""))
            self.assertIsNotNone(prof)


if __name__ == "__main__":
    unittest.main()
