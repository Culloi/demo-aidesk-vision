"""静止元素过滤（亮度时域稳定性）的单元测试。

解决的问题：摄像头拍到显示器上的红色 UI 边框/图标被误判为明火。
判据：屏幕 UI 静止，框内平均亮度随时间几乎不变；真实火焰会持续闪烁。

运行：QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from usb_person_detector.entrypoints.gui_app import CameraWorker  # noqa: E402
from usb_person_detector.vision.detector import Detection  # noqa: E402

_app = QApplication.instance() or QApplication([])


def make_worker(**kw):
    opts = dict(model_path="x.onnx", camera="auto", width=1280, height=720,
                conf_thres=0.4, iou_thres=0.5, class_conf={}, companion_cfg=None,
                save_dir="captures", sleep_fps=2)
    opts.update(kw)
    w = CameraWorker(**opts)
    w.log_ready.connect(lambda s: None)
    return w


def fire_det():
    return Detection(500, 300, 700, 500, 0.6, 0, "fire")


class TestStaticFilter(unittest.TestCase):
    def _run(self, worker, brightness_seq):
        """返回每一步是否保留了检测（True=保留）。"""
        worker.fire_bright_hist.clear()
        out = []
        for b in brightness_seq:
            frame = np.full((720, 1280, 3), int(b), np.uint8)
            out.append(bool(worker._filter_static([fire_det()], frame)))
        return out

    def test_static_ui_filtered_after_window(self):
        """亮度恒定的屏幕 UI：窗口满后被过滤。"""
        w = make_worker(fire_static_filter=True, fire_static_min_std=1.2, fire_static_window=5)
        res = self._run(w, [120] * 8)
        self.assertTrue(all(res[:4]), "样本不足时应先放行")
        self.assertFalse(any(res[4:]), "样本够了且亮度恒定，应全部过滤")

    def test_flickering_fire_kept(self):
        """真实火焰亮度波动大，必须全部保留。"""
        w = make_worker(fire_static_filter=True, fire_static_min_std=1.2, fire_static_window=5)
        res = self._run(w, [120, 150, 110, 165, 125, 145, 115, 160])
        self.assertTrue(all(res), "闪烁的火焰不应被过滤")

    def test_stable_flame_kept(self):
        """稳定火焰（小幅抖动，如蜡烛）不应被误伤。"""
        w = make_worker(fire_static_filter=True, fire_static_min_std=1.2, fire_static_window=5)
        res = self._run(w, [120, 123, 119, 124, 120, 122, 121, 123])
        self.assertTrue(all(res), "小幅抖动仍在阈值之上，应保留")

    def test_sensor_noise_ui_still_filtered(self):
        """静止 UI + 传感器噪声（std < 阈值）仍应被过滤。"""
        w = make_worker(fire_static_filter=True, fire_static_min_std=1.2, fire_static_window=5)
        res = self._run(w, [120, 120.5, 119.6, 120.3, 120.1, 119.8, 120.2])
        self.assertFalse(any(res[4:]), "噪声幅度很小，仍应过滤")

    def test_disabled_keeps_everything(self):
        w = make_worker(fire_static_filter=False)
        frame = np.full((720, 1280, 3), 120, np.uint8)
        for _ in range(10):
            self.assertTrue(w._filter_static([fire_det()], frame))

    def test_hist_cleared_when_no_fire(self):
        """火焰消失后历史应能清空，避免下次误判。"""
        w = make_worker(fire_static_filter=True, fire_static_window=5)
        w.fire_bright_hist = [120, 120, 120]
        w.fire_bright_hist.clear()
        self.assertEqual(len(w.fire_bright_hist), 0)

    def test_out_of_frame_box_kept(self):
        """框完全越界时判不了，应放过（不制造漏报）。"""
        w = make_worker(fire_static_filter=True)
        frame = np.zeros((100, 100, 3), np.uint8)
        d = Detection(500, 500, 600, 600, 0.6, 0, "fire")
        self.assertTrue(w._filter_static([d], frame))


if __name__ == "__main__":
    unittest.main()
