"""Forward-only RMSNorm exercise. The implementation is intentionally empty."""
import torch
import torch.nn.functional as F

from kernellab.spec import Case, Spec, io_bytes, tensor
from exercises.cuda_wrappers import rms_norm_cuda


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Your implementation: contiguous x[M,N], weight[N], output same as x.

    Supported dtypes: float32, float16, bfloat16. No backward required.
    Do not modify inputs. You may use Triton, CUDA/C++, or PyTorch.
    Keep the epsilon fixed at 1e-6 when comparing with the baseline.
    """
    raise NotImplementedError("Implement exercises/rms_norm.py:rms_norm yourself")


def make_inputs(case, dtype, device, generator, pattern):
    return (
        tensor(case.shapes[0], dtype, device, generator, pattern),
        tensor(case.shapes[1], dtype, device, generator, "random"),
    )


def reference(x, weight):
    # Use the library operator as an oracle; no hand-written RMSNorm here.
    return F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6)


SPEC = Spec(
    name="rms_norm",
    description="RMSNorm forward, contiguous [M,N] + weight[N], eps=1e-6",
    cases=[Case(f"{m}x{n}", ((m, n), (n,))) for m, n in [
        (1, 1), (1, 127), (4, 513), (32, 1024), (128, 4096),
        (1024, 4096), (2048, 8192),
    ]],
    make_inputs=make_inputs,
    reference=reference,
    implementations={"mine": rms_norm, "cuda": rms_norm_cuda},
    logical_bytes=io_bytes,
)
