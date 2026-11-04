"""运行环境自检（主机和 RK3576 板子通用）。

用法：PYTHONPATH=src python -m usb_person_detector.entrypoints.env_check

也被 GUI 入口在启动时调用：检测到有问题的 PySide6 时给出明确提示，而不是运行几分钟后闪退。
"""
import os
import sys
from pathlib import Path

RECOMMENDED_PYSIDE = "6.8.3"
FIX_COMMANDS = f"""\
  source .venv/bin/activate
  python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials PySide6_Pdf PySide6_WebEngine shiboken6
  python -m pip install "PySide6-Essentials=={RECOMMENDED_PYSIDE}"
  PYTHONPATH=src python -m usb_person_detector.entrypoints.env_check"""


def pyside_none_leak(calls: int = 50) -> bool:
    """检测 PySide6 是否存在 None 引用计数泄漏的 bug。

    PySide6 6.12.0 在 Python 3.10 上，几乎所有无返回值的 Qt 函数每调用一次都会把 None 的
    引用计数多减 1，减到 0 时进程直接中止（Fatal Python error: none_dealloc）。
    Python 3.12+ 的 None 引用计数是固定的，不受影响。
    """
    from PySide6.QtGui import QColor
    color = QColor(255, 0, 0)
    color.setAlphaF(0.5)
    before = sys.getrefcount(None)
    for _ in range(calls):
        color.setAlphaF(0.5)
    return sys.getrefcount(None) - before <= -calls


def _ok(msg):
    print(f"  [通过] {msg}")


def _fail(msg):
    print(f"  [失败] {msg}")


def _warn(msg):
    print(f"  [提示] {msg}")


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    root = Path(__file__).resolve().parents[3]
    failed = 0
    print("== RX-78 陪伴机器人 运行环境自检 ==")

    py = sys.version_info
    _ok(f"Python {py.major}.{py.minor}.{py.micro}")

    try:
        import PySide6
        from PySide6.QtWidgets import QApplication
        _ok(f"PySide6 {PySide6.__version__}")
        if pyside_none_leak():
            failed += 1
            _fail(f"PySide6 {PySide6.__version__} 在 Python {py.major}.{py.minor} 上有 None 引用计数 bug，界面运行几分钟后会闪退")
            print("         修复方法：\n" + FIX_COMMANDS)
        else:
            _ok("PySide6 无 None 引用计数问题")

        app = QApplication.instance() or QApplication(sys.argv)
        from PySide6.QtGui import QFontDatabase, QImage
        svg = QImage(str(root / "assets" / "arrow_up.svg"))
        (_ok if not svg.isNull() else _warn)("SVG 图标加载" + ("正常" if not svg.isNull() else "失败（数值框箭头不显示，不影响功能）"))
        families = set(QFontDatabase.families())
        cjk = [f for f in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback") if f in families]
        if cjk:
            _ok(f"中文字体：{cjk[0]}")
        else:
            failed += 1
            _fail("未找到中文字体，界面中文会显示为方块。安装：sudo apt-get install -y fonts-noto-cjk")
        del app
    except ImportError as e:
        failed += 1
        _fail(f"PySide6 未安装：{e}")

    try:
        import cv2
        import numpy
        import onnxruntime
        _ok(f"OpenCV {cv2.__version__} / NumPy {numpy.__version__} / ONNX Runtime {onnxruntime.__version__}")
    except ImportError as e:
        failed += 1
        _fail(f"依赖缺失：{e}")

    model = root / "models" / "yolov8n.onnx"
    (_ok if model.exists() else _fail)(f"模型文件 {model.relative_to(root)}" + ("" if model.exists() else " 不存在"))
    failed += 0 if model.exists() else 1

    # NPU：开发板上应该可用；主机上没有是正常情况（自动用 CPU），只提示不算失败
    try:
        import json

        from usb_person_detector.vision.rknn_backend import RknnModel, rknn_available
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        rknn_path = root / config.get("rknn_model", "models/yolov8n_rk3576_fp16.rknn")
        if config.get("backend", "auto") == "cpu":
            _warn("配置为只用 CPU 推理（backend = cpu）")
        elif not rknn_available():
            _warn("未检测到 NPU 运行库 librknnrt.so，使用 CPU 推理（主机上是正常情况）")
        elif not rknn_path.exists():
            _warn(f"NPU 可用，但没有 RKNN 模型 {rknn_path.relative_to(root)}，将使用 CPU 推理"
                  f"（转换方法见 tools/convert_rknn.py）")
        else:
            npu = RknnModel(str(rknn_path))
            _ok(f"NPU 推理：运行库 {npu.api_version.split()[0]} / 驱动 {npu.driver_version}，"
                f"模型 {rknn_path.relative_to(root)} 可加载")
            npu.release()
    except Exception as e:
        _warn(f"NPU 模型加载失败，将使用 CPU 推理：{e}")

    print("== 结果：" + ("全部通过 ==" if failed == 0 else f"{failed} 项失败 =="))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
