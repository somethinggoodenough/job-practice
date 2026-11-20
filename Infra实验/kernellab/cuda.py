"""Compile CUDA C++ device code with NVIDIA NVRTC and launch on PyTorch streams.

No host C++ compiler or PyTorch C++ extension is involved. Kernel arguments are
torch.Tensor pointers or explicitly sized ctypes scalars, matching the .cu ABI.
"""
import ctypes as C
import ctypes.util
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
P = C.c_void_p
INT = C.c_int
UINT = C.c_uint
SIZE = C.c_size_t


def _bind(library, name, args, result=INT):
    fn = getattr(library, name)
    fn.argtypes = args
    fn.restype = result
    return fn


class APIs:
    def __init__(self):
        torch_lib = Path(torch.__file__).parent / "lib"
        if os.name == "nt":
            candidates = sorted(p for p in torch_lib.glob("nvrtc64*.dll") if ".alt." not in p.name)
            if not candidates:
                raise RuntimeError("NVRTC DLL not found in torch/lib. Install this lab's CUDA PyTorch build.")
            self.nvrtc_path = str(candidates[0])
            self.nvrtc = C.WinDLL(self.nvrtc_path)
            self.driver = C.WinDLL("nvcuda.dll")
        else:
            candidates = list(torch_lib.glob("libnvrtc.so*"))
            # CUDA-enabled Linux wheels usually keep NVRTC in an nvidia package.
            candidates += list((torch_lib.parent.parent / "nvidia/cuda_nvrtc/lib").glob("libnvrtc.so*"))
            self.nvrtc_path = str(candidates[0]) if candidates else C.util.find_library("nvrtc")
            if not self.nvrtc_path:
                raise RuntimeError("NVRTC not found. Install CUDA NVRTC or the lab's CUDA PyTorch build.")
            self.nvrtc = C.CDLL(self.nvrtc_path)
            self.driver = C.CDLL("libcuda.so.1")
        self.version = _bind(self.nvrtc, "nvrtcVersion", [C.POINTER(INT), C.POINTER(INT)])
        self.error_string = _bind(self.nvrtc, "nvrtcGetErrorString", [INT], C.c_char_p)
        self.create = _bind(self.nvrtc, "nvrtcCreateProgram", [C.POINTER(P), C.c_char_p, C.c_char_p, INT, P, P])
        self.compile = _bind(self.nvrtc, "nvrtcCompileProgram", [P, INT, C.POINTER(C.c_char_p)])
        self.destroy = _bind(self.nvrtc, "nvrtcDestroyProgram", [C.POINTER(P)])
        self.log_size = _bind(self.nvrtc, "nvrtcGetProgramLogSize", [P, C.POINTER(SIZE)])
        self.log = _bind(self.nvrtc, "nvrtcGetProgramLog", [P, P])
        self.ptx_size = _bind(self.nvrtc, "nvrtcGetPTXSize", [P, C.POINTER(SIZE)])
        self.ptx = _bind(self.nvrtc, "nvrtcGetPTX", [P, P])
        self.init = _bind(self.driver, "cuInit", [UINT])
        self.error_name = _bind(self.driver, "cuGetErrorName", [INT, C.POINTER(C.c_char_p)])
        self.error_desc = _bind(self.driver, "cuGetErrorString", [INT, C.POINTER(C.c_char_p)])
        self.get_context = _bind(self.driver, "cuCtxGetCurrent", [C.POINTER(P)])
        self.load = _bind(self.driver, "cuModuleLoadData", [C.POINTER(P), P])
        self.function = _bind(self.driver, "cuModuleGetFunction", [C.POINTER(P), P, C.c_char_p])
        self.unload = _bind(self.driver, "cuModuleUnload", [P])
        self.launch = _bind(self.driver, "cuLaunchKernel", [P, UINT, UINT, UINT, UINT, UINT, UINT, UINT, P, C.POINTER(P), P])
        self.attribute = _bind(self.driver, "cuFuncSetAttribute", [P, INT, INT])
        self.check_driver(self.init(0))

    def check_nvrtc(self, code):
        if code:
            raise RuntimeError(self.error_string(code).decode("utf-8", errors="replace"))

    def check_driver(self, code):
        if code:
            name, desc = C.c_char_p(), C.c_char_p()
            self.error_name(code, C.byref(name))
            self.error_desc(code, C.byref(desc))
            raise RuntimeError(f"CUDA driver error {code}: {name.value!r}: {desc.value!r}")

    def info(self):
        major, minor = INT(), INT()
        self.check_nvrtc(self.version(C.byref(major), C.byref(minor)))
        return {"compiler": "NVIDIA NVRTC", "version": f"{major.value}.{minor.value}", "library": self.nvrtc_path}


@lru_cache(maxsize=1)
def apis():
    return APIs()


def _dim3(value):
    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3 or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("grid/block must contain 1 to 3 positive integers")
    return values + (1,) * (3 - len(values))


class CUDAKernel:
    def __init__(self, path, name, device=None, options=()):
        self.path = Path(path).resolve()
        self.device = torch.device(device if device is not None else f"cuda:{torch.cuda.current_device()}")
        if self.device.type != "cuda":
            raise ValueError("CUDAKernel requires a CUDA device")
        self.device = torch.device("cuda", self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.api = apis()
        self.module = P()
        self.handle = P()
        with torch.cuda.device(self.device):
            # Create/activate PyTorch's primary context; never create a separate one.
            torch.empty(1, device=self.device)
            self.context = P()
            self.api.check_driver(self.api.get_context(C.byref(self.context)))
            major, minor = torch.cuda.get_device_capability(self.device)
            self.options = (f"--gpu-architecture=compute_{major}{minor}", "--std=c++17", "--device-as-default-execution-space", *options)
            source = self.path.read_bytes()
            key = hashlib.sha256(source + repr(self.options).encode() + str(self.api.info()).encode()).hexdigest()
            cache = ROOT / ".cache/cuda" / key
            cache.mkdir(parents=True, exist_ok=True)
            program = P()
            self.api.check_nvrtc(self.api.create(C.byref(program), source, self.path.name.encode(), 0, None, None))
            try:
                encoded = [arg.encode() for arg in self.options]
                compile_options = (C.c_char_p * len(encoded))(*encoded)
                result = self.api.compile(program, len(encoded), compile_options)
                size = SIZE()
                self.api.check_nvrtc(self.api.log_size(program, C.byref(size)))
                log = C.create_string_buffer(size.value)
                self.api.check_nvrtc(self.api.log(program, log))
                (cache / "compile.log").write_bytes(log.value)
                if result:
                    raise RuntimeError(f"CUDA compilation failed for {self.path}:\n{log.value.decode('utf-8', errors='replace')}")
                self.api.check_nvrtc(self.api.ptx_size(program, C.byref(size)))
                ptx = C.create_string_buffer(size.value)
                self.api.check_nvrtc(self.api.ptx(program, ptx))
                (cache / "kernel.ptx").write_bytes(ptx.value)
                self.api.check_driver(self.api.load(C.byref(self.module), ptx))
                try:
                    self.api.check_driver(self.api.function(C.byref(self.handle), self.module, name.encode()))
                except Exception:
                    self.api.unload(self.module)
                    self.module = P()
                    raise
            finally:
                self.api.destroy(C.byref(program))
        # Modules live for the process lifetime; their function handles stay valid
        # for asynchronous launches and the process-wide get_kernel cache.

    def launch(self, args, grid, block=256, shared_mem=0):
        """Launch asynchronously on PyTorch's CURRENT stream of this device.

        Tensor arguments become device pointers; scalars must use ctypes types
        (c_int, c_float, c_uint64, ...), exactly matching the CUDA signature.
        Keep input/output tensors alive until the current stream finishes.
        """
        grid, block = _dim3(grid), _dim3(block)
        if not isinstance(shared_mem, int) or shared_mem < 0:
            raise ValueError("shared_mem must be a non-negative byte count")
        packed = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.device != self.device or not arg.is_contiguous():
                    raise ValueError("CUDA tensor arguments must be contiguous and on the kernel's device")
                packed.append(P(arg.data_ptr()))
            elif isinstance(arg, (C.c_int, C.c_uint, C.c_int64, C.c_uint64, C.c_float, C.c_double, C.c_void_p)):
                packed.append(arg)
            else:
                raise TypeError("Kernel arguments must be CUDA tensors or explicitly sized ctypes scalars")
        pointers = (P * len(packed))(*(C.cast(C.byref(arg), P) for arg in packed))
        with torch.cuda.device(self.device):
            current = P()
            self.api.check_driver(self.api.get_context(C.byref(current)))
            if current.value != self.context.value:
                raise RuntimeError("The CUDA context changed; create a new CUDAKernel")
            stream = torch.cuda.current_stream(self.device)
            # Make tensor allocator lifetimes safe on non-default streams.
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg.record_stream(stream)
            self.api.check_driver(self.api.launch(self.handle, *grid, *block, shared_mem, P(stream.cuda_stream), pointers, None))

    def allow_dynamic_shared_memory(self, nbytes):
        """Opt in to larger per-block dynamic shared memory (GPU limits apply)."""
        with torch.cuda.device(self.device):
            self.api.check_driver(self.api.attribute(self.handle, 8, nbytes))


@lru_cache(maxsize=None)
def get_kernel(path, name, device_index=0, options=()):
    """Compile once per process/device/options. A new CLI run reloads your code."""
    return CUDAKernel(path, name, f"cuda:{device_index}", options)
