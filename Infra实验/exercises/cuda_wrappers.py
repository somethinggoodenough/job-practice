"""Fill the Python launch wrappers together with kernels/cuda/*.cu.

See kernellab/demo_copy.py:cuda_copy for a complete compilation/launch example.
Use kernellab.cuda.get_kernel and ctypes scalars to bind your .cu signature.
"""
from pathlib import Path
import ctypes
import torch
from kernellab.cuda import get_kernel

EPS = 1e-6
def rms_norm_cuda(x, weight):
    assert x.is_cuda and x.dtype == torch.float32, "这个 kernel 只支持 CUDA 上的 fp32"

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()   # 保证是 2D + row-major
    weight = weight.contiguous()
    rows, cols = x2d.shape

    output = torch.empty_like(x2d)

    kernel = get_kernel("kernels/cuda/rms_norm.cu", "rms_norm_f32", x.device.index)
    kernel.launch(
        args=(
            x2d, weight, output,
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            ctypes.c_float(EPS),
        ),
        grid=rows,    # 一个 block 负责一行
        block=256,    # 必须跟 block_reduce_sum 里的 s[256] 对上
    )
    return output.reshape(orig_shape)

def vector_add_cuda(a, b):
    raise NotImplementedError("Implement kernels/cuda/vector_add.cu and its Python launch wrapper yourself")


def softmax_cuda(x):
    raise NotImplementedError("Implement kernels/cuda/softmax.cu and its Python launch wrapper yourself")


def matmul_cuda(a, b):
    raise NotImplementedError("Implement kernels/cuda/matmul.cu and its Python launch wrapper yourself")


# Include CUDA source contents in benchmark provenance fingerprints.
_sources = Path(__file__).resolve().parent.parent / "kernels/cuda"
for _fn, _name in [(rms_norm_cuda, "rms_norm"), (vector_add_cuda, "vector_add"),
                   (softmax_cuda, "softmax"), (matmul_cuda, "matmul")]:
    _fn.kernel_sources = (_sources / f"{_name}.cu",)
