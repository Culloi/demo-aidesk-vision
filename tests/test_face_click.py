"""表情脸双击切换全屏的单元测试（offscreen，无需真实屏幕）。

运行：QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from usb_person_detector.ui.face_widget import FaceWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])


class TestFaceDoubleClick(unittest.TestCase):
    def setUp(self):
        self.face = FaceWidget()
        self.face.resize(400, 300)
        self.clicks = 0
        self.face.double_clicked.connect(self._on_double_click)

    def _on_double_click(self):
        self.clicks += 1

    def test_double_click_emits(self):
        QTest.mouseDClick(self.face, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        self.assertEqual(self.clicks, 1)

    def test_two_double_clicks_emit_twice(self):
        QTest.mouseDClick(self.face, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        QTest.mouseDClick(self.face, Qt.LeftButton, Qt.NoModifier, QPoint(120, 120))
        self.assertEqual(self.clicks, 2)

    def test_single_click_does_not_emit(self):
        QTest.mouseClick(self.face, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        self.assertEqual(self.clicks, 0)

    def test_right_double_click_does_not_emit(self):
        QTest.mouseDClick(self.face, Qt.RightButton, Qt.NoModifier, QPoint(100, 100))
        self.assertEqual(self.clicks, 0)


if __name__ == "__main__":
    unittest.main()
