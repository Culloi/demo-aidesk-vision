"""推理方式选择测试（NPU / CPU 自动切换），用模拟对象，主机上不需要 NPU。

运行：python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.vision import detector, rknn_backend  # noqa: E402
from usb_person_detector.vision.detector import create_detector  # noqa: E402


class FakeDetector:
    """代替 YoloDetector：按文件后缀判断推理方式；fail_npu=True 时模拟 NPU 加载失败。"""
    fail_npu = False

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.backend = "npu" if model_path.endswith(".rknn") else "cpu"
        if self.backend == "npu" and FakeDetector.fail_npu:
            raise rknn_backend.RknnError("rknn_init 失败，错误码 -1")


class CreateDetectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rknn = Path(self.tmp.name) / "m.rknn"
        self.rknn.write_bytes(b"fake")
        self.onnx = "models/yolov8n.onnx"
        FakeDetector.fail_npu = False
        patcher = mock.patch.object(detector, "YoloDetector", FakeDetector)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def npu(self, available):
        return mock.patch.object(rknn_backend, "rknn_available", return_value=available)

    def test_auto_uses_npu_on_board(self):
        with self.npu(True):
            det, reason = create_detector(self.onnx, "auto", str(self.rknn), npu_core="0_1")
        self.assertEqual(det.backend, "npu")
        self.assertEqual(det.kwargs["npu_core"], "0_1")
        self.assertIsNone(reason)

    def test_auto_on_host_without_npu_is_silent(self):
        """主机上没有 NPU 运行库是正常情况，不应该提示退回。"""
        with self.npu(False):
            det, reason = create_detector(self.onnx, "auto", str(self.rknn))
        self.assertEqual(det.backend, "cpu")
        self.assertIsNone(reason)

    def test_auto_falls_back_when_model_missing(self):
        with self.npu(True):
            det, reason = create_detector(self.onnx, "auto", str(self.rknn) + ".missing")
        self.assertEqual(det.backend, "cpu")
        self.assertIn("not found", reason)

    def test_auto_falls_back_when_npu_fails(self):
        FakeDetector.fail_npu = True
        with self.npu(True):
            det, reason = create_detector(self.onnx, "auto", str(self.rknn))
        self.assertEqual(det.backend, "cpu")
        self.assertEqual(det.model_path, self.onnx)
        self.assertIn("rknn_init", reason)

    def test_cpu_forced(self):
        with self.npu(True):
            det, reason = create_detector(self.onnx, "cpu", str(self.rknn))
        self.assertEqual(det.backend, "cpu")
        self.assertIsNone(reason)

    def test_npu_forced_raises_on_failure(self):
        with self.npu(False):
            with self.assertRaises(RuntimeError):
                create_detector(self.onnx, "npu", str(self.rknn))
        FakeDetector.fail_npu = True
        with self.npu(True):
            with self.assertRaises(RuntimeError):
                create_detector(self.onnx, "npu", str(self.rknn))

    def test_explicit_rknn_model_path(self):
        with self.npu(True):
            det, reason = create_detector(str(self.rknn), "cpu")
        self.assertEqual(det.backend, "npu")


class RknnHostTest(unittest.TestCase):
    def test_load_library_error_message(self):
        with mock.patch.object(rknn_backend.ctypes, "CDLL", side_effect=OSError("not found")):
            self.assertFalse(rknn_backend.rknn_available())
            with self.assertRaises(rknn_backend.RknnError):
                rknn_backend.load_library()


if __name__ == "__main__":
    unittest.main()
