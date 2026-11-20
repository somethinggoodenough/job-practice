import torch
from kernellab.spec import Case, Spec, io_bytes, tensor
from exercises.cuda_wrappers import softmax_cuda


def softmax(x):
    """Softmax over the last dimension; return the same shape/dtype/device."""
    raise NotImplementedError("Implement exercises/softmax.py:softmax yourself")


def make_inputs(case, dtype, device, generator, pattern):
    return (tensor(case.shapes[0], dtype, device, generator, pattern),)


SPEC = Spec(
    "softmax", "Row-wise softmax forward, contiguous [M,N]",
    [Case(f"{m}x{n}", ((m, n),)) for m, n in [(1, 1), (4, 127), (32, 513), (128, 4096), (1024, 8192)]],
    make_inputs, lambda x: torch.softmax(x, dim=-1), {"mine": softmax, "cuda": softmax_cuda}, io_bytes,
)
