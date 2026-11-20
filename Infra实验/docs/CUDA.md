# CUDA C++ 练习

这里可以直接写 CUDA `.cu`，并复用与 Triton 相同的校验、benchmark、历史报告。**RMSNorm 的 Python wrapper 和 CUDA kernel 都未实现。**

当前已在本机 RTX 4060 上验证：`.cu` → NVIDIA NVRTC 12.8 → PTX → CUDA Driver API → PyTorch 当前 stream。NVRTC 是 NVIDIA 的 CUDA C++ 运行时编译器，不需要 MSVC 来编译 device code；详见 [NVIDIA NVRTC 官方说明](https://docs.nvidia.com/cuda/nvrtc/index.html)。

## 先确认工具链和计时

```powershell
.\lab.cmd doctor --cuda-smoke
.\lab.cmd bench demo_copy --impl cuda_copy --dtype all --timer cuda-graph --tag cuda-ready
.\lab.cmd bench demo_copy --impl all --dtype all --timer cuda-graph --tag cuda-vs-triton
```

copy demo 是用于验证环境的简单字节复制，没有做性能优化，也没有 RMSNorm 答案。它能复制各种 dtype 的原始存储，不代表练习模板已经实现了低精度运算。

## 要编辑的文件

| 文件 | 用途 |
| --- | --- |
| `kernels/cuda/rms_norm.cu` | RMSNorm 空 kernel 模板 |
| `kernels/cuda/vector_add.cu` | Vector Add 空模板 |
| `kernels/cuda/softmax.cu` | Softmax 空模板 |
| `kernels/cuda/matmul.cu` | Matmul 空模板 |
| `exercises/cuda_wrappers.py` | 自己填写 Tensor 分配、参数和 launch 配置 |
| `kernels/cuda/copy.cu` | 已完成的独立 copy 参考，仅展示 CUDA 基础结构 |
| `kernellab/demo_copy.py:cuda_copy` | 已完成的 Python 编译与 launch 接线参考 |
| `kernellab/cuda.py` | 基础设施，一般无需修改 |

练习 `.cu` 中有 `#error TODO...`，请在开始实现时删掉；Python wrapper 会先抛 `NotImplementedError`，避免把空模板当成正确结果。模板初始指针类型是 `float`，先用 `--dtype float32`，之后再自行扩展 fp16/bf16。

## 编译和启动接口

```python
import ctypes as C
from kernellab.cuda import get_kernel

# source_path 指向你自己的 .cu；name 是 extern "C" __global__ 的函数名。
kernel = get_kernel(source_path, name, x.device.index)

# 参数顺序和类型必须与你的 .cu 函数签名完全一致。
# Tensor 自动传 data_ptr，标量必须显式写出 ABI 类型：
#   int -> C.c_int，unsigned long long -> C.c_uint64，float -> C.c_float
kernel.launch(
    args=(x, output, C.c_int(n)),
    grid=(grid_x, grid_y),
    block=(block_x, block_y),
    shared_mem=dynamic_shared_memory_bytes,
)
```

`grid`、`block` 可为一维整数或 1–3 个正整数的 tuple。动态 shared memory 用 `.cu` 的 `extern __shared__` 声明；若需超过默认每 block 限额，可先调用 `kernel.allow_dynamic_shared_memory(nbytes)`，上限取决于 GPU。

wrapper 必须返回输出 Tensor，并保证输入、输出为同 device 的连续张量。运行器会检查输出 dtype/shape/device、数值、输入是否被修改；越界访问等内存错误仍须自己调试。CUDA launch 异步执行，编译日志与 PTX 保存在 `.cache/cuda/<hash>/`，校验和计时会在适当位置同步。

每个进程/device/options 只编译一次。修改代码后重新运行 CLI 就会重新读取源码；不要在单个长驻 Python 进程中期待自动热更新。使用 `extern "C"` 避免 C++ name mangling。默认以实际 GPU 的 compute capability 编译，不开启 `--use_fast_math`。

可通过 `options` 添加 NVIDIA NVRTC 支持的编译选项，例如 include 目录或宏：

```python
kernel = get_kernel(source_path, name, x.device.index,
                    options=("--include-path=你的CUDA头文件目录", "-DMY_OPTION=1"))
```

`cuda_fp16.h`、`cuda_bf16.h`、WMMA 等头文件需要可用的 CUDA include 目录，当前 loader 不自动拼接系统中旧版本 Toolkit 的头文件。先写不依赖额外头文件的 float32 kernel 即可练习大多数基本概念。

## 校验和性能对比

```powershell
# 填完自己的 .cu 和 wrapper 后
.\lab.cmd check rms_norm --impl cuda --dtype float32
.\lab.cmd bench rms_norm --impl cuda --dtype float32 --timer cuda-graph --tag cuda-v1

# mine 是原来的自由练习入口，可放自己的 Triton 实现；都写完后同场对比
.\lab.cmd bench rms_norm --impl all --dtype float32 --timer cuda-graph --tag compare
.\lab.cmd report 'results/rms_norm-*.json' --output results/rms-cuda-history.html
```

要检查最大 case 的 kernel 数量和 GPU 时间，可运行：

```powershell
.\lab.cmd profile rms_norm --impl cuda --dtype float32 --case 2048x8192
```

命令会分别 profile 自定义实现和 PyTorch baseline，避免一张表中的父子事件或多次调用混淆归因，并把 Chrome traces 保存到 `results/profiles/`。

小 kernel 优先用 `--timer cuda-graph` 看 GPU 工作耗时；`--timer events` 则可观察普通 Python 调用路径。ctypes 参数封装、指针检查等会增加普通调用的 CPU 开销，两种计时模式不要混着比较。需要性能剖析时，graph timing 仍不能代替 Nsight Compute。

## 与 nvcc / PyTorch C++ 扩展的区别

当前路线编译的是 CUDA C++ **device code**；可以使用 `__global__`、`__device__`、线程/block、shared memory、warp 原语等。Python 负责内存分配和 launch，不支持在同一文件中编译任意 host C++、`#include <torch/extension.h>` 或 host 上的 `<<<...>>>` 调用。

如果要练习完整 C++ host 程序、PyTorch CUDA extension 或 `.cu` 的离线 nvcc 构建，需要额外安装兼容的 CUDA Toolkit、MSVC C++ Build Tools / Windows SDK 和构建工具。当前系统检测到 nvcc 11.8，PyTorch 使用 CUDA 12.8，未找到可用的 MSVC；这条路线尚未配置和验证。相关依赖见 [PyTorch extension 文档](https://docs.pytorch.org/docs/stable/cpp_extension.html) 与 [CUDA 12.8 Windows 编译器要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)。

现在的 NVRTC 路线不需要更改全局 CUDA_PATH、不替换现有 Toolkit，也不安装或替换 GPU 驱动。
