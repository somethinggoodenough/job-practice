# Kernel Practice Lab

在自己的 kernel 和 PyTorch 之间做正确性检查与性能对比。面向本机 Windows + RTX 4060，也可搬到 Linux / WSL / 远程 NVIDIA GPU。

**RMSNorm 没有实现，也没有隐藏答案。** Python 和 CUDA `.cu` 中都只留接口；reference 直接调用 PyTorch 内置 `F.rms_norm`。其他练习也留空。写好的 Triton / CUDA kernel 仅为独立内存 copy，用于环境验收和演示 benchmark。

## 立即开始（Windows PowerShell）

```powershell
# 首次安装 / 重建环境：下载量约 3GB，使用独立 .venv
python scripts/setup.py

# 检查 CUDA，并实际编译执行 Triton copy
.\lab.cmd doctor --smoke
.\lab.cmd doctor --cuda-smoke
.\lab.cmd list

# 先体验完整的正确性检查 + GPU benchmark + HTML 图表
.\lab.cmd bench demo_copy --impl triton_copy --dtype float32 --tag toolchain-smoke

# 同场比较 CUDA C++ / Triton / PyTorch，Graph 模式减少 Python 启动开销
.\lab.cmd bench demo_copy --impl all --dtype float32 --timer cuda-graph --tag three-backends

# 还没开始写 RMSNorm 时，先记录 PyTorch 性能
.\lab.cmd bench rms_norm --baseline-only --tag rms-baseline
```

每次 benchmark 自动生成 `results/<op>-<UTC时间>.json`、`.csv`、`.html`。双击 HTML 可离线查看柱状图、P25/P75 和环境信息，可以筛选 shape / dtype / 实验记录。这里的时间戳使用 UTC，避免跨时区文件名冲突；页面展示完整时区信息。

`lab.cmd` 可以从 PowerShell 和 cmd 启动，不受 PowerShell 脚本签名策略影响。等价的 Python 命令是：

```powershell
.\.venv\Scripts\python.exe -m kernellab doctor --smoke
```

## 练 RMSNorm

编辑 `exercises/rms_norm.py` 中的 `rms_norm`。接口约定：

- 输入 `x[M,N]` 与 `weight[N]` 都是连续张量、相同 dtype / device。
- 返回与 `x` 相同 shape / dtype / device 的张量，不能修改输入。
- epsilon 固定为 `1e-6`，只做 forward，不要求 backward。
- 支持 `float32`、`float16`、`bfloat16`；测试包含 N=1、非 2 的幂长度以及不同批大小。
- 可以写 Triton，也可以封装 CUDA/C++ 扩展，或先写自己的 PyTorch 版本。框架只要求一个 Python callable。

```powershell
# 先从小 case 开始；空函数会明确报 unimplemented，并返回非零退出码
.\lab.cmd check rms_norm --case 4x513 --dtype float32

# 完整检查：随机、全零、极小、较大输入、重复调用和另一随机种子
.\lab.cmd check rms_norm --dtype all

# 正确性通过后对比所有形状
.\lab.cmd bench rms_norm --dtype float16 --tag v1
.\lab.cmd bench rms_norm --dtype all --tag v2

# 只看关心的形状
.\lab.cmd bench rms_norm --case '128x4096,1024x4096' --tag large-batch

# 把多个历史结果放进同一个图表
.\lab.cmd report 'results/rms_norm-*.json' --output results/rms-history.html
```

默认 `--device auto` 优先 GPU。`--device cuda` 可要求必须有 GPU；`--device cpu` 只适合调试 Python 和框架，其结果会明确标注 CPU。Triton 实现本身不能在 CPU 上运行。

## 保留多个实现对比

在练习文件里自己定义多个函数，然后在 `SPEC.implementations` 注册，例如：

```python
implementations={"v1": my_first_version, "v2": my_second_version}
```

```powershell
.\lab.cmd bench rms_norm --impl all --tag compare-v1-v2
.\lab.cmd bench rms_norm --impl v2 --case 128x4096
```

每个实现都会独立校验；未实现或错误的实现没有性能数字，不参与加速比。`bench` 仍然会保存失败详情和可用 baseline，最后返回非零退出码。`--baseline-only` 明确表示只测库函数，不会声称你的练习已通过。

## 其他 kernel / 添加练习

`exercises/` 已提供 `vector_add`、`softmax`、`matmul` 空练习，每个都有自己的 shape、数据生成器、库函数 reference 和误差阈值。运行方式和 RMSNorm 相同。

添加 `exercises/my_kernel.py` 并导出 `SPEC = Spec(...)` 就会自动注册，无需修改 CLI。参照 `exercises/vector_add.py` 填写：名字、测试 case、`make_inputs`、`reference`、`implementations`、逻辑 I/O 字节估算函数。默认接口返回单个 Tensor；多输出 kernel 需扩展校验器。

项目支持 Triton 与直接编写 CUDA C++ `.cu` 两条路线。CUDA 路线的具体文件和使用方法见 [CUDA 练习说明](docs/CUDA.md)。`.cu` 用 NVIDIA NVRTC 编译，通过 CUDA Driver API 在 PyTorch 当前 stream 上启动，可以练习线程、block、shared memory、warp 原语等。

```powershell
# 自己填写 kernels/cuda/rms_norm.cu 和 exercises/cuda_wrappers.py 后：
.\lab.cmd check rms_norm --impl cuda --dtype float32
.\lab.cmd bench rms_norm --impl cuda --dtype float32 --timer cuda-graph --tag cuda-v1
```

分析某个 case 实际启动了哪些 CUDA kernel：

```powershell
# largest 会选择 RMSNorm 注册的最后一个 case：2048x8192
.\lab.cmd profile rms_norm --impl cuda --dtype float32 --case largest

# 记录多次调用时，表中的 CUDA total 是总和；除以 repeat 得到单次时间
.\lab.cmd profile rms_norm --impl cuda --dtype float32 --case 2048x8192 --repeat 10
```

Profiler 会在计时外分别预热自定义实现和 PyTorch baseline，再生成两张独立的 CUDA kernel 表。原始 Chrome trace 和文本汇总保存在 `results/profiles/`，可以用 `chrome://tracing` 或 Perfetto 打开。这里使用普通 eager 调用，便于看到实际 kernel 数量；它与 `bench --timer cuda-graph` 的 graph replay 结果应分开理解。

CUDA C++ 运行时编译已经实测通过。传统 `nvcc + PyTorch C++ extension` 是另一条构建路线，本机尚缺匹配 CUDA 12.8 的开发工具链和 MSVC，当前没有宣称它已配置完成。

## 如何理解 benchmark

- 默认 20 次预热、15 组采样、每组连续调用 50 次；JIT 编译与初次 autotune 在计时前完成。
- CUDA 使用事件计时并同步，展示各组「平均单次调用时间」的中位数及 P25/P75。CPU 使用 `perf_counter_ns`，固定单线程。
- 默认 `--timer events` 测普通 Python 调用路径，包含 GPU 时间线中的 launch 间隙；输出分配也在调用范围内。
- `--timer cuda-graph` 会捕获每组多次调用，再用 CUDA events 测量 graph replay，按调用次数归一化，避免把 Python / ctypes 的逐次启动开销当成 kernel 时间。编译、捕获、graph-pool 输出分配都在计时外；它仍包含 graph 的 GPU 调度成本，不等同于逐指令 profiling。输入地址和 launch 配置在 replay 内固定；实现必须支持 stream capture。
- 每组复用常驻输入，属于 warm-cache 场景。小输入可能由 L2 服务，**逻辑 GB/s 不等于显存带宽**。
- `speedup = 同次运行的 PyTorch 中位延迟 / 当前实现中位延迟`，只有同 shape、dtype、device、计时配置才适合比较。
- benchmark 输入不启用梯度；没有测 H2D/D2H、backward、冷启动或端到端模型性能。
- JSON 保存每组原始样本、软件 / GPU / 驱动 / NVRTC 信息、seed、计时模式、误差阈值、候选函数所在文件的 SHA256 短指纹。CUDA 练习还纳入已声明 `.cu` 文件的内容；其他外部依赖和 include 文件需自行加入 `fn.kernel_sources`，重要实验应另外保存代码版本。
- 4060 同时承担桌面显示。测试时关闭 GPU 密集应用并观察 P25/P75，避免把一次抖动当成优化收益。没有为此修改系统电源设置或 GPU 时钟。

可调参数：

```powershell
.\lab.cmd bench rms_norm --warmup 50 --samples 25 --iterations 100 --seed 123 --tag careful-run
```

## 环境与维护

环境使用 Python 3.11、PyTorch 2.9.1 CUDA 12.8、Triton 3.5 系列。Windows 使用 `triton-windows`；根据其兼容表，PyTorch 2.9 对应 Triton 3.5。此组合固定用于练习，不追求最新版。

- [PyTorch 官方安装记录](https://pytorch.org/get-started/previous-versions/)
- [triton-windows 的硬件、版本和工具链说明](https://github.com/triton-lang/triton-windows/blob/readme/README.md)
- [Triton 官方文档](https://triton-lang.org/main/index.html)

`triton-windows` 的 wheel 自带精简 CUDA 工具链和 C 编译器，通常无需单独装 Visual Studio 来运行 Triton。实际以 `doctor --smoke` 成功为准。本机默认 `python` 指向旧环境，运行练习务必使用 `.\lab.cmd` 或本项目 `.venv` 的 Python。`scripts/setup.py` 可以从旧 Python 启动，会为练习选择独立的 Python 3.11。

本机已经验证的版本是 `torch==2.9.1+cu128`、`triton-windows==3.5.1.post24`、`numpy==2.2.6`。CLI 针对 Windows 上中文缓存路径的编码问题设置了 Triton 的 Python cache knob，不修改系统区域设置或第三方包源码。

Linux / WSL 中安装好 `uv` 和 NVIDIA 驱动后：

```bash
bash scripts/setup.sh
.venv/bin/python -m kernellab bench demo_copy --impl triton_copy
```

源码可以共用，**Windows 和 Linux 的 `.venv` 不能共用**，请分别在各自目录重建。CLI 自动把 Triton 缓存放在项目 `.cache/triton`；uv 缓存位于 `.cache/uv`。

运行框架回归测试（不会替你实现练习）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
