"""火焰检测中心裁剪（数字变焦）的单元测试。

核心是要证明：裁剪只影响火焰模型，且坐标还原正确，不会破坏主检测。

运行：python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.vision.detector import fire_roi  # noqa: E402


class TestFireRoi(unittest.TestCase):
    def test_off_mode_returns_same_object(self):
        """mode=off 时直接返回原帧，零拷贝开销。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi, off = fire_roi(frame, mode="off")
        self.assertIs(roi, frame)
        self.assertEqual(off, (0, 0))

    def test_low_resolution_not_cropped(self):
        """短边 <= min_side（如 640x480）时裁剪无收益，应不裁剪。"""
        frame = np.zeros((480, 640, 3), np.uint8)
        roi, off = fire_roi(frame, mode="auto")
        self.assertIs(roi, frame)
        self.assertEqual(off, (0, 0))

    def test_auto_crops_720p(self):
        """1280x720 + ratio 0.75 → 边长 640，偏移居中。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi, (ox, oy) = fire_roi(frame, mode="auto", ratio=0.75, min_side=640)
        self.assertEqual(roi.shape[:2], (640, 640))
        self.assertEqual(ox, (1280 - 640) // 2)
        self.assertEqual(oy, (720 - 640) // 2)

    def test_auto_gives_2x_zoom_on_720p(self):
        """1280 宽裁到 640 → scale 从 0.5 提升到 1.0，正好 2 倍。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi, _ = fire_roi(frame, mode="auto", ratio=0.75, min_side=640)
        full_scale = min(640 / 1280, 640 / 720)
        crop_scale = min(640 / roi.shape[1], 640 / roi.shape[0])
        self.assertAlmostEqual(crop_scale / full_scale, 2.0, places=2)

    def test_1080p_keeps_view_coverage(self):
        """1920x1080 + ratio 0.75 → 边长 810，保留 75% 视野，仍放大 2.37 倍。"""
        frame = np.zeros((1080, 1920, 3), np.uint8)
        roi, _ = fire_roi(frame, mode="auto", ratio=0.75, min_side=640)
        self.assertEqual(roi.shape[:2], (810, 810))
        full_scale = min(640 / 1920, 640 / 1080)
        crop_scale = 640 / 810
        self.assertGreater(crop_scale / full_scale, 2.0)

    def test_center_mode_uses_min_side(self):
        """mode=center 固定裁 min_side，不受 ratio 影响。"""
        frame = np.zeros((1080, 1920, 3), np.uint8)
        roi, off = fire_roi(frame, mode="center", ratio=0.9, min_side=640)
        self.assertEqual(roi.shape[:2], (640, 640))
        self.assertEqual(off, ((1920 - 640) // 2, (1080 - 640) // 2))

    def test_does_not_modify_input_frame(self):
        """裁剪必须只读，绝不能原地修改 frame（否则主检测画面会被裁掉）。"""
        frame = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
        before = frame.copy()
        fire_roi(frame, mode="auto")
        np.testing.assert_array_equal(frame, before)

    def test_offset_restores_coordinates(self):
        """裁剪图上的框 + offset 必须还原到整帧坐标（否则火焰框会画错位置）。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi, (ox, oy) = fire_roi(frame, mode="auto", ratio=0.75, min_side=640)
        # 造一个裁剪图坐标系里的框
        x1, y1, x2, y2 = 100.0, 120.0, 180.0, 200.0
        restored = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        # 还原后必须落在整帧范围内
        self.assertTrue(0 <= restored[0] < restored[2] <= 1280)
        self.assertTrue(0 <= restored[1] < restored[3] <= 720)
        # 验证 ROI 内容确实对应整帧的该区域
        frame[oy + 120:oy + 200, ox + 100:ox + 180] = 255
        np.testing.assert_array_equal(roi[120:200, 100:180], frame[oy + 120:oy + 200, ox + 100:ox + 180])

    def test_portrait_frame(self):
        """竖屏（短边是宽）也要正确居中裁剪。"""
        frame = np.zeros((720, 480, 3), np.uint8)
        roi, (ox, oy) = fire_roi(frame, mode="auto", ratio=0.75, min_side=400)
        self.assertEqual(roi.shape[0], roi.shape[1])       # 正方形
        self.assertEqual(ox, (480 - roi.shape[1]) // 2)

    def test_ratio_clamped(self):
        """ratio 非法值要被夹住，不能产生越界切片。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        for r in (0.0, -1.0, 2.0, 99):
            roi, (ox, oy) = fire_roi(frame, mode="auto", ratio=r, min_side=640)
            self.assertGreaterEqual(roi.shape[0], 1)
            self.assertGreaterEqual(roi.shape[1], 1)
            self.assertGreaterEqual(ox, 0)
            self.assertGreaterEqual(oy, 0)


if __name__ == "__main__":
    unittest.main()
