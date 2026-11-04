"""RKNN NPU 推理后端：通过 ctypes 直接调用开发板上的 librknnrt.so。

本模块位于 vision 包中，用于隔离 RK3576 NPU 的平台适配代码。

为什么不用官方的 rknn-toolkit-lite2：它最高只支持 Python 3.12，开发板是 Python 3.13 装不上。
librknnrt.so 是板子系统自带的 C 运行库（/usr/lib/librknnrt.so），用 ctypes 调用不需要额外安装任何东西。

结构体定义与 RKNN SDK 2.3.2 的 rknn_api.h 对应（64 位系统，rknn_context 为 uint64）。
"""
import ctypes
import os
from ctypes import (POINTER, Structure, byref, c_char, c_float, c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64,
                    c_void_p, sizeof)

import numpy as np

LIB_CANDIDATES = ("librknnrt.so", "/usr/lib/librknnrt.so", "/usr/lib/aarch64-linux-gnu/librknnrt.so",
                  "/usr/local/lib/librknnrt.so")

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

# rknn_query_cmd
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_SDK_VERSION = 5

# rknn_tensor_type / rknn_tensor_format
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1

# rknn_core_mask（RK3576 有 2 个 NPU 核）
CORE_MASKS = {"auto": 0, "0": 1, "1": 2, "0_1": 3, "all": 0xFFFF}

TYPE_NAMES = {0: "float32", 1: "float16", 2: "int8", 3: "uint8", 4: "int16", 5: "uint16", 6: "int32",
              7: "uint32", 8: "int64", 9: "bool", 10: "int4"}
FMT_NAMES = {0: "NCHW", 1: "NHWC", 2: "NC1HWC2", 3: "UNDEFINED"}


class RknnInputOutputNum(Structure):
    _fields_ = [("n_input", c_uint32), ("n_output", c_uint32)]


class RknnTensorAttr(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("n_dims", c_uint32),
        ("dims", c_uint32 * RKNN_MAX_DIMS),
        ("name", c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", c_uint32),
        ("size", c_uint32),
        ("fmt", c_int),
        ("type", c_int),
        ("qnt_type", c_int),
        ("fl", c_int8),
        ("zp", c_int32),
        ("scale", c_float),
        ("w_stride", c_uint32),
        ("size_with_stride", c_uint32),
        ("pass_through", c_uint8),
        ("h_stride", c_uint32),
    ]

    def describe(self):
        dims = [self.dims[i] for i in range(self.n_dims)]
        return (f"{self.name.decode(errors='replace')} dims={dims} "
                f"{TYPE_NAMES.get(self.type, self.type)} {FMT_NAMES.get(self.fmt, self.fmt)}")


class RknnSdkVersion(Structure):
    _fields_ = [("api_version", c_char * 256), ("drv_version", c_char * 256)]


class RknnInput(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32),
        ("pass_through", c_uint8),
        ("type", c_int),
        ("fmt", c_int),
    ]


class RknnOutput(Structure):
    _fields_ = [
        ("want_float", c_uint8),
        ("is_prealloc", c_uint8),
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32),
    ]


class RknnError(RuntimeError):
    pass


def load_library():
    errors = []
    for name in LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            errors.append(f"{name}: {e}")
    raise RknnError("找不到 NPU 运行库 librknnrt.so：" + "; ".join(errors))


def rknn_available():
    """板子上有 NPU 运行库时返回 True（主机上通常为 False）。"""
    try:
        load_library()
        return True
    except RknnError:
        return False


class RknnModel:
    """加载 .rknn 模型并在 NPU 上推理。

    输入：NHWC uint8 RGB（转换时已把 /255 归一化写进模型，见 tools/convert_rknn.py）
    输出：float32 numpy 数组列表，形状与 ONNX 输出一致
    """

    def __init__(self, model_path: str, core_mask: str = "auto"):
        self.lib = lib = load_library()
        lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p, c_uint32, c_uint32, c_void_p]
        lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
        lib.rknn_inputs_set.argtypes = [c_uint64, c_uint32, POINTER(RknnInput)]
        lib.rknn_run.argtypes = [c_uint64, c_void_p]
        lib.rknn_outputs_get.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput), c_void_p]
        lib.rknn_outputs_release.argtypes = [c_uint64, c_uint32, POINTER(RknnOutput)]
        lib.rknn_destroy.argtypes = [c_uint64]
        lib.rknn_set_core_mask.argtypes = [c_uint64, c_int]

        with open(model_path, "rb") as f:
            data = f.read()
        self._model_buf = ctypes.create_string_buffer(data, len(data))
        self.ctx = c_uint64(0)
        self._check(lib.rknn_init(byref(self.ctx), self._model_buf, len(data), 0, None), "rknn_init")
        self._model_buf = None  # rknn_init 内部已拷贝模型，释放这份内存

        if core_mask in CORE_MASKS and core_mask != "auto":
            self._check(lib.rknn_set_core_mask(self.ctx, CORE_MASKS[core_mask]), "rknn_set_core_mask")

        version = RknnSdkVersion()
        self._check(lib.rknn_query(self.ctx, RKNN_QUERY_SDK_VERSION, byref(version), sizeof(version)), "query version")
        self.api_version = version.api_version.decode(errors="replace")
        self.driver_version = version.drv_version.decode(errors="replace")

        io_num = RknnInputOutputNum()
        self._check(lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM, byref(io_num), sizeof(io_num)), "query io")
        self.inputs = [self._query_attr(RKNN_QUERY_INPUT_ATTR, i) for i in range(io_num.n_input)]
        self.outputs = [self._query_attr(RKNN_QUERY_OUTPUT_ATTR, i) for i in range(io_num.n_output)]

        attr = self.inputs[0]
        dims = [attr.dims[i] for i in range(attr.n_dims)]
        if attr.fmt == RKNN_TENSOR_NHWC:
            self.input_height, self.input_width = dims[1], dims[2]
        else:
            self.input_height, self.input_width = dims[2], dims[3]

    def _check(self, ret, what):
        if ret != 0:
            raise RknnError(f"{what} 失败，错误码 {ret}")

    def _query_attr(self, cmd, index):
        attr = RknnTensorAttr()
        attr.index = index
        self._check(self.lib.rknn_query(self.ctx, cmd, byref(attr), sizeof(attr)), f"query attr {index}")
        return attr

    def run(self, image_rgb: np.ndarray):
        """image_rgb：HxWx3 uint8 RGB，尺寸需等于模型输入尺寸。返回 float32 输出列表。"""
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        inp = RknnInput()
        inp.index = 0
        inp.buf = image_rgb.ctypes.data
        inp.size = image_rgb.nbytes
        inp.pass_through = 0
        inp.type = RKNN_TENSOR_UINT8
        inp.fmt = RKNN_TENSOR_NHWC
        self._check(self.lib.rknn_inputs_set(self.ctx, 1, byref(inp)), "rknn_inputs_set")
        self._check(self.lib.rknn_run(self.ctx, None), "rknn_run")

        n = len(self.outputs)
        outs = (RknnOutput * n)()
        for i in range(n):
            outs[i].want_float = 1
            outs[i].is_prealloc = 0
            outs[i].index = i
        self._check(self.lib.rknn_outputs_get(self.ctx, n, outs, None), "rknn_outputs_get")
        try:
            results = []
            for i, attr in enumerate(self.outputs):
                count = outs[i].size // 4
                arr = np.ctypeslib.as_array((c_float * count).from_address(outs[i].buf)).copy()
                shape = [attr.dims[d] for d in range(attr.n_dims)]
                results.append(arr.reshape(shape) if int(np.prod(shape)) == count else arr)
        finally:
            self.lib.rknn_outputs_release(self.ctx, n, outs)
        return results

    def release(self):
        if getattr(self, "ctx", None) is not None and self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = c_uint64(0)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/yolov8n_rk3576_fp16.rknn"
    print("运行库可用:", rknn_available(), "| PID", os.getpid())
    m = RknnModel(path)
    print("API", m.api_version, "| 驱动", m.driver_version)
    for a in m.inputs:
        print("输入", a.describe())
    for a in m.outputs:
        print("输出", a.describe())
