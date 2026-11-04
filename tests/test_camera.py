"""USB 摄像头自动查找测试（用临时目录模拟 /sys/class/video4linux，不需要真实摄像头）。

运行：python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.vision import detector  # noqa: E402
from usb_person_detector.vision.detector import (  # noqa: E402
    CameraNotFoundError,
    LatestFrameReader,
    find_usb_cameras,
    open_camera,
)


def make_node(root: Path, name: str, driver: str, index: int):
    """模拟 /sys/class/video4linux/<name>，device/driver 是指向驱动目录的符号链接。"""
    node = root / "class" / name
    dev = root / "devices" / name
    drv = root / "drivers" / driver
    for d in (node, dev, drv):
        d.mkdir(parents=True, exist_ok=True)
    (node / "index").write_text(f"{index}\n")
    os.symlink(drv, dev / "driver")
    os.symlink(dev, node / "device")


class FindUsbCameraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "class").mkdir()
        self.sysfs = str(self.root / "class")

    def tearDown(self):
        self.tmp.cleanup()

    def test_rk3576_layout(self):
        """前面是 MIPI 节点（rkcif），USB 摄像头排在后面；只取 index=0 的视频节点。"""
        for i in range(27):
            make_node(self.root, f"video{i}", "rkcif", 0)
        make_node(self.root, "video27", "uvcvideo", 0)   # USB 摄像头：视频
        make_node(self.root, "video28", "uvcvideo", 1)   # USB 摄像头：元数据
        self.assertEqual(find_usb_cameras(self.sysfs), ["/dev/video27"])

    def test_numeric_order_and_multiple_cameras(self):
        make_node(self.root, "video10", "uvcvideo", 0)
        make_node(self.root, "video11", "uvcvideo", 1)
        make_node(self.root, "video9", "uvcvideo", 0)
        self.assertEqual(find_usb_cameras(self.sysfs), ["/dev/video9", "/dev/video10"])

    def test_no_usb_camera(self):
        make_node(self.root, "video0", "rkcif", 0)
        self.assertEqual(find_usb_cameras(self.sysfs), [])

    def test_missing_sysfs(self):
        self.assertEqual(find_usb_cameras(str(self.root / "not_exist")), [])

    def test_open_auto_without_camera_raises(self):
        with mock.patch.object(detector, "find_usb_cameras", return_value=[]):
            for value in ("auto", "AUTO", -1, "-1", None):
                with self.assertRaises(CameraNotFoundError):
                    open_camera(value, 1280, 720)

    def test_open_auto_uses_first_camera(self):
        fake_cap = mock.MagicMock()
        with mock.patch.object(detector, "find_usb_cameras", return_value=["/dev/video27", "/dev/video30"]), \
                mock.patch.object(detector.cv2, "VideoCapture", return_value=fake_cap) as vc:
            cap, source = open_camera("auto", 1280, 720)
        self.assertEqual(source, "/dev/video27")
        self.assertIs(cap, fake_cap)
        self.assertEqual(vc.call_args[0][0], "/dev/video27")

    def test_open_explicit_number(self):
        with mock.patch.object(detector.cv2, "VideoCapture", return_value=mock.MagicMock()) as vc:
            _, source = open_camera("2", 1280, 720)
        self.assertEqual(source, 2)
        self.assertEqual(vc.call_args[0][0], 2)


class FakeCapture:
    """模拟摄像头：每 20 ms 出一帧，帧内容是序号；fail_after 帧之后读帧失败。"""

    def __init__(self, fail_after=None, interval=0.02):
        self.count = 0
        self.fail_after = fail_after
        self.interval = interval

    def read(self):
        time.sleep(self.interval)
        if self.fail_after is not None and self.count >= self.fail_after:
            return False, None
        self.count += 1
        return True, self.count


class LatestFrameReaderTest(unittest.TestCase):
    def test_returns_latest_frame_not_stale(self):
        """处理慢于摄像头时，拿到的应该是最新帧，而不是缓冲里的旧帧。"""
        cap = FakeCapture()
        reader = LatestFrameReader(cap)
        try:
            ok, first = reader.read()
            self.assertTrue(ok)
            time.sleep(0.2)                      # 模拟一次慢推理，期间摄像头又出了约 10 帧
            ok, frame = reader.read()
            self.assertTrue(ok)
            self.assertGreaterEqual(frame, cap.count - 1)   # 最多晚 1 帧
            self.assertGreater(frame, first + 5)
        finally:
            reader.stop()

    def test_never_returns_same_frame_twice(self):
        reader = LatestFrameReader(FakeCapture())
        try:
            frames = [reader.read()[1] for _ in range(5)]
            self.assertEqual(len(set(frames)), 5)
            self.assertEqual(frames, sorted(frames))
        finally:
            reader.stop()

    def test_camera_failure(self):
        reader = LatestFrameReader(FakeCapture(fail_after=2))
        try:
            results = [reader.read(timeout=1.0)[0] for _ in range(4)]
            self.assertFalse(results[-1])        # 摄像头出错后返回 False，不会一直卡住
        finally:
            reader.stop()

    def test_timeout_when_no_frame(self):
        reader = LatestFrameReader(FakeCapture(interval=0.5))
        try:
            start = time.time()
            ok, frame = reader.read(timeout=0.1)
            self.assertFalse(ok)
            self.assertIsNone(frame)
            self.assertLess(time.time() - start, 0.4)
        finally:
            reader.stop()

    def test_stop(self):
        reader = LatestFrameReader(FakeCapture())
        reader.read()
        reader.stop()
        self.assertFalse(reader._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
