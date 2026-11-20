import torch
from kernellab.spec import Case, Spec, io_bytes, tensor
from exercises.cuda_wrappers import vector_add_cuda


def vector_add(a, b):
    """Return a+b with the same shape/dtype/device, without modifying inputs."""
    raise NotImplementedError("Implement exercises/vector_add.py:vector_add yourself")


def make_inputs(case, dtype, device, generator, pattern):
    return tuple(tensor(s, dtype, device, generator, pattern) for s in case.shapes)


SPEC = Spec(
    "vector_add", "Elementwise addition, contiguous 1D tensors",
    [Case(str(n), ((n,), (n,))) for n in (1, 257, 4096, 1048576, 16777216)],
    make_inputs, torch.add, {"mine": vector_add, "cuda": vector_add_cuda}, io_bytes,
)
