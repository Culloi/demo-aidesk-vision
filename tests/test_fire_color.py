"""火焰颜色二次校验的单元测试。

解决的问题：摄像头拍到显示器上的纯红色 UI 元素（对话框边框、图标、指示灯）
被火焰模型误判为明火。

判据：真实火焰含橙黄色成分（HSV H 10~35），纯红色 UI 的色相集中在 H 0~10，
橙黄像素比例几乎为 0。这是实测出来的分离度最好的特征。

运行：python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.vision.detector import Detection, fire_color_check  # noqa: E402


def box(x1, y1, x2, y2, score=0.6):
    return Detection(float(x1), float(y1), float(x2), float(y2), score, 0, "fire")


def frame_with(color_bgr, w=640, h=480, rect=(100, 100, 400, 300)):
    """在灰底上画一个指定颜色的实心矩形。"""
    img = np.full((h, w, 3), 50, np.uint8)
    x1, y1, x2, y2 = rect
    img[y1:y2, x1:x2] = color_bgr
    return img, box(x1, y1, x2, y2)


class TestFireColorCheck(unittest.TestCase):
    def test_pure_red_ui_rejected(self):
        """纯红色（H=0）UI 元素应被拒：橙黄比例 0。"""
        img, d = frame_with((0, 0, 255))
        ok, info = fire_color_check(img, d)
        self.assertFalse(ok)
        self.assertEqual(info["orange_ratio"], 0.0)

    def test_dark_red_ui_rejected(self):
        img, d = frame_with((30, 20, 180))
        ok, _ = fire_color_check(img, d)
        self.assertFalse(ok)

    def test_red_border_rejected(self):
        """红色空心边框（模拟对话气泡边框）。"""
        img = np.full((480, 640, 3), 50, np.uint8)
        cv2.rectangle(img, (100, 100), (400, 300), (0, 0, 255), 8)
        ok, _ = fire_color_check(img, box(100, 100, 400, 300))
        self.assertFalse(ok)

    def test_orange_flame_accepted(self):
        """橙黄火焰（H≈18，典型火焰色）应通过。"""
        img, d = frame_with((0, 140, 255))     # BGR -> 橙
        ok, info = fire_color_check(img, d)
        self.assertTrue(ok, f"橙黄应通过，实际 orange_ratio={info['orange_ratio']}")

    def test_yellow_flame_accepted(self):
        img, d = frame_with((0, 230, 250))     # 亮黄
        ok, _ = fire_color_check(img, d)
        self.assertTrue(ok)

    def test_mixed_flame_like_region_accepted(self):
        """火焰典型结构：白黄核心 + 橙色外焰 + 红色边缘，含一定背景。"""
        img = np.full((480, 640, 3), 50, np.uint8)
        cv2.circle(img, (250, 200), 60, (60, 180, 255), -1)    # 橙外焰
        cv2.circle(img, (250, 200), 35, (150, 230, 255), -1)   # 黄
        cv2.circle(img, (250, 200), 15, (230, 245, 255), -1)   # 白核心
        ok, info = fire_color_check(img, box(150, 110, 350, 290))
        self.assertTrue(ok, f"火焰结构应通过，orange_ratio={info['orange_ratio']}")

    def test_threshold_respected(self):
        """阈值可调：提高阈值后，低橙黄比例的框会被拒。"""
        img, d = frame_with((0, 140, 255))
        self.assertTrue(fire_color_check(img, d, min_orange_ratio=0.05)[0])
        # 只在画面里占一小部分橙黄，比例远低于 0.9
        img2 = np.full((480, 640, 3), 50, np.uint8)
        img2[100:130, 100:160] = (0, 140, 255)
        self.assertFalse(fire_color_check(img2, box(100, 100, 400, 300), min_orange_ratio=0.5)[0])

    def test_gray_region_rejected(self):
        """灰色/白色区域没有橙黄成分，应被拒。"""
        img, d = frame_with((200, 200, 200))
        ok, _ = fire_color_check(img, d)
        self.assertFalse(ok)

    def test_blue_flame_rejected_by_default(self):
        """蓝色火焰（燃气灶/防风打火机）默认会被拒——需要时把 fire_color_check 关掉。"""
        img, d = frame_with((255, 120, 0))     # 蓝
        ok, _ = fire_color_check(img, d)
        self.assertFalse(ok)

    def test_empty_box_passes(self):
        """框越界/为空时宁可放过，不制造新的漏报。"""
        img = np.zeros((100, 100, 3), np.uint8)
        ok, info = fire_color_check(img, box(200, 200, 300, 300))
        self.assertTrue(ok)
        self.assertEqual(info, {})

    def test_info_fields_present(self):
        img, d = frame_with((0, 140, 255))
        ok, info = fire_color_check(img, d)
        for k in ("orange_ratio", "v_std", "v_mean", "box"):
            self.assertIn(k, info)

    def test_does_not_modify_frame(self):
        img, d = frame_with((0, 140, 255))
        before = img.copy()
        fire_color_check(img, d)
        np.testing.assert_array_equal(img, before)


if __name__ == "__main__":
    unittest.main()
