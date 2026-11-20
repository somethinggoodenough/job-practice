import torch
from kernellab.spec import Case, Spec, io_bytes, tensor
from exercises.cuda_wrappers import matmul_cuda


def matmul(a, b):
    """Contiguous A[M,K] @ B[K,N]; no bias, output same dtype/device."""
    raise NotImplementedError("Implement exercises/matmul.py:matmul yourself")


def make_inputs(case, dtype, device, generator, pattern):
    return tuple(tensor(s, dtype, device, generator, pattern) for s in case.shapes)


SPEC = Spec(
    "matmul", "Dense matrix multiplication [M,K] @ [K,N]",
    [Case(f"{m}x{k}x{n}", ((m, k), (k, n))) for m, k, n in [
        (1, 1, 1), (31, 63, 17), (128, 128, 128), (512, 1024, 512), (1024, 1024, 1024),
    ]], make_inputs, torch.matmul, {"mine": matmul, "cuda": matmul_cuda}, io_bytes,
    patterns=("random", "zeros", "tiny"),
    tolerances={"float32": (1e-3, 2e-3), "float16": (1e-2, 5e-2), "bfloat16": (5e-2, 3e-1)},
)
