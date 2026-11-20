"""Completed copy kernel used ONLY to verify the toolchain and demo benchmarking."""
import torch
import ctypes
from pathlib import Path
from kernellab.spec import Case, Spec, io_bytes, tensor

CUDA_COPY_SOURCE = str(Path(__file__).resolve().parent.parent / "kernels/cuda/copy.cu")


def triton_copy(x):
    if not x.is_cuda:
        raise RuntimeError("triton_copy requires CUDA; use --impl torch_copy on CPU")
    from kernellab.triton_smoke import copy_kernel
    import triton
    y = torch.empty_like(x)
    copy_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y


def make_inputs(case, dtype, device, generator, pattern):
    return (tensor(case.shapes[0], dtype, device, generator, pattern),)


def cuda_copy(x):
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("cuda_copy requires a contiguous CUDA tensor")
    from kernellab.cuda import get_kernel
    kernel = get_kernel(CUDA_COPY_SOURCE, "copy_bytes", x.device.index)
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    kernel.launch((x, y, ctypes.c_uint64(nbytes)), grid=(nbytes + 255) // 256, block=256)
    return y


cuda_copy.kernel_sources = (CUDA_COPY_SOURCE,)


SPEC = Spec(
    "demo_copy", "Completed copy demo for validating the environment",
    [Case(str(n), ((n,),)) for n in (257, 4096, 1048576, 16777216)],
    make_inputs, torch.clone, {"triton_copy": triton_copy, "cuda_copy": cuda_copy, "torch_copy": torch.clone}, io_bytes,
)
