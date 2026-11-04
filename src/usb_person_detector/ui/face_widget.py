"""桌面陪伴机器人表情脸（QPainter 绘制，无图片资源依赖）。

- 表情：idle / normal / happy / curious / alert / worried / sleep
- 目光跟随：set_gaze((x, y))，范围 -1~1；None 时空闲状态下自动四处张望
- 自动眨眼、平滑过渡、字幕气泡、角落信息、画中画摄像头缩略图
"""
import math
import random
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from usb_person_detector.core.companion import (
    EXPR_ALERT, EXPR_CURIOUS, EXPR_HAPPY, EXPR_IDLE, EXPR_NORMAL, EXPR_SLEEP, EXPR_WORRIED,
)
from usb_person_detector.ui import theme

EYE_COLOR = QColor("#EAF2FF")
EYE_GLOW = QColor(61, 123, 255, 70)
PUPIL_COLOR = QColor(theme.NAVY_DEEP)
RED = QColor(theme.RED)
YELLOW = QColor(theme.YELLOW)
BLUE_LIGHT = QColor(theme.BLUE_LIGHT)

# 表情 -> (睁眼程度, 眼睛缩放, 眉毛角度[-1 担心 ~ +1 生气], 眉毛颜色)
EXPRESSION_TARGETS = {
    EXPR_IDLE: (0.85, 0.95, 0.0, None),
    EXPR_NORMAL: (1.0, 1.0, 0.0, None),
    EXPR_HAPPY: (1.0, 1.0, 0.0, None),
    EXPR_CURIOUS: (1.0, 1.15, 0.0, None),
    EXPR_ALERT: (0.5, 1.0, 1.0, RED),
    EXPR_WORRIED: (0.85, 1.0, -1.0, BLUE_LIGHT),
    EXPR_SLEEP: (0.0, 1.0, 0.0, None),
}


def _lerp(a, b, k):
    return a + (b - a) * k


class FaceWidget(QWidget):
    double_clicked = Signal()     # 双击表情脸，用于切换全屏

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FaceWidget")
        self.setMinimumSize(320, 200)
        self.setCursor(Qt.PointingHandCursor)

        self.expression = EXPR_IDLE
        self.target_gaze = None
        self.gaze = [0.0, 0.0]
        self.caption = ""
        self.caption_shown = ""
        self.caption_alpha = 0.0
        self.info_left = ""
        self.info_right = ""
        self.thumbnail = None

        # 动画参数（平滑逼近目标值）
        self.open = 0.85
        self.scale = 0.95
        self.brow = 0.0
        self.brow_alpha = 0.0
        self.brow_color = RED

        self.blink = 0.0
        self.blink_start = None
        self.next_blink = time.monotonic() + random.uniform(2.0, 5.0)
        self.t0 = time.monotonic()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    # ---------- 对外接口 ----------
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def set_expression(self, expr: str):
        if expr in EXPRESSION_TARGETS:
            self.expression = expr

    def set_gaze(self, gaze):
        self.target_gaze = gaze

    def set_caption(self, text: str):
        self.caption = text or ""
        if self.caption:
            self.caption_shown = self.caption

    def set_info(self, left: str = "", right: str = ""):
        self.info_left, self.info_right = left, right

    def set_thumbnail(self, image):
        self.thumbnail = image

    # ---------- 动画 ----------
    def _tick(self):
        now = time.monotonic()
        t = now - self.t0
        open_t, scale_t, brow_t, brow_color = EXPRESSION_TARGETS[self.expression]
        self.open = _lerp(self.open, open_t, 0.2)
        self.scale = _lerp(self.scale, scale_t, 0.2)
        self.brow = _lerp(self.brow, brow_t, 0.2)
        self.brow_alpha = _lerp(self.brow_alpha, 1.0 if brow_color is not None else 0.0, 0.2)
        if brow_color is not None:
            self.brow_color = brow_color

        # 目光：有目标跟随目标；空闲时缓慢四处张望；其他情况回正
        if self.target_gaze is not None:
            gx, gy = self.target_gaze
            k = 0.25
        elif self.expression == EXPR_IDLE:
            gx = 0.7 * math.sin(t * 0.45) * math.cos(t * 0.13)
            gy = 0.25 * math.sin(t * 0.31)
            k = 0.05
        else:
            gx, gy, k = 0.0, 0.0, 0.1
        self.gaze[0] = _lerp(self.gaze[0], gx, k)
        self.gaze[1] = _lerp(self.gaze[1], gy, k)

        # 眨眼（睡觉和开心时不眨）
        if self.expression in (EXPR_SLEEP, EXPR_HAPPY):
            self.blink = 0.0
            self.blink_start = None
        elif self.blink_start is None and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is not None:
            p = (now - self.blink_start) / 0.16
            if p >= 1.0:
                self.blink = 0.0
                self.blink_start = None
                # 偶尔连眨两下
                self.next_blink = now + (0.25 if random.random() < 0.15 else random.uniform(2.5, 5.5))
            else:
                self.blink = math.sin(p * math.pi)

        self.caption_alpha = _lerp(self.caption_alpha, 1.0 if self.caption else 0.0, 0.2)
        self.update()

    # ---------- 绘制 ----------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        t = time.monotonic() - self.t0

        # 背景：驾驶舱深蓝 + 中心微光
        grad = QRadialGradient(QPointF(w / 2, h * 0.45), max(w, h) * 0.65)
        grad.setColorAt(0.0, QColor("#16295E"))
        grad.setColorAt(1.0, QColor(theme.NAVY_DEEP))
        p.fillRect(self.rect(), grad)

        base_w = min(w * 0.15, h * 0.26)
        base_h = base_w * 1.3
        spacing = base_w * 1.05
        cx = w / 2 + self.gaze[0] * w * 0.025     # 整体轻微转头
        cy = h * 0.47 + self.gaze[1] * h * 0.02

        self._draw_crest(p, w / 2, cy - base_h * 0.95, base_w)
        for side in (-1, 1):
            self._draw_eye(p, cx + side * spacing, cy, base_w, base_h, side)
        if self.expression == EXPR_HAPPY:
            for side in (-1, 1):
                self._draw_blush(p, cx + side * (spacing + base_w * 0.35), cy + base_h * 0.55, base_w)
        if self.expression == EXPR_SLEEP:
            self._draw_zzz(p, cx + spacing + base_w * 0.5, cy - base_h * 0.3, base_w, t)

        self._draw_info(p, w)
        thumb_rect = self._draw_thumbnail(p, w, h)
        self._draw_caption(p, w, h, thumb_rect)
        p.end()

    def _draw_crest(self, p, x, y, unit):
        """额头：黄色 V 字天线 + 红色主摄像头。"""
        fin = QPen(YELLOW, max(3.0, unit * 0.09), Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        p.setPen(fin)
        spread, rise = unit * 1.15, unit * 0.55
        p.drawLine(QPointF(x, y), QPointF(x - spread, y - rise))
        p.drawLine(QPointF(x, y), QPointF(x + spread, y - rise))
        p.setPen(Qt.NoPen)
        p.setBrush(RED)
        r = unit * 0.13
        p.drawRoundedRect(QRectF(x - r, y - r * 0.6, r * 2, r * 1.6), r * 0.3, r * 0.3)

    def _draw_eye(self, p, ex, ey, base_w, base_h, side):
        ew = base_w * self.scale
        openness = max(0.0, self.open * (1.0 - self.blink))
        eh = base_h * self.scale * openness

        if self.expression == EXPR_HAPPY:
            # ^ ^ 形状
            pen = QPen(EYE_COLOR, ew * 0.2, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            path = QPainterPath(QPointF(ex - ew * 0.45, ey + base_h * 0.12))
            path.quadTo(QPointF(ex, ey - base_h * 0.55), QPointF(ex + ew * 0.45, ey + base_h * 0.12))
            p.drawPath(path)
            return

        if openness < 0.1:
            # 闭眼：略向下弯的线
            pen = QPen(EYE_COLOR, ew * 0.12, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            path = QPainterPath(QPointF(ex - ew * 0.45, ey))
            path.quadTo(QPointF(ex, ey + base_h * 0.15), QPointF(ex + ew * 0.45, ey))
            p.drawPath(path)
            self._draw_brow(p, ex, ey, ew, base_h, side)
            return

        rect = QRectF(ex - ew / 2, ey - eh / 2, ew, eh)
        radius = min(ew, eh) * 0.35
        # 外发光
        p.setPen(Qt.NoPen)
        p.setBrush(EYE_GLOW)
        g = ew * 0.1
        p.drawRoundedRect(rect.adjusted(-g, -g, g, g), radius + g, radius + g)
        # 眼白
        p.setBrush(EYE_COLOR)
        p.drawRoundedRect(rect, radius, radius)

        # 瞳孔（裁剪在眼眶内）
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        p.save()
        p.setClipPath(clip)
        pr = ew * (0.3 if self.expression == EXPR_CURIOUS else 0.26)
        px = ex + self.gaze[0] * ew * 0.28
        py = ey + self.gaze[1] * base_h * 0.22
        p.setBrush(PUPIL_COLOR)
        p.drawEllipse(QPointF(px, py), pr, pr)
        p.setBrush(QColor(theme.BLUE))
        p.drawEllipse(QPointF(px, py), pr * 0.45, pr * 0.45)
        p.setBrush(QColor(255, 255, 255, 230))
        p.drawEllipse(QPointF(px - pr * 0.35, py - pr * 0.4), pr * 0.22, pr * 0.22)
        p.restore()

        self._draw_brow(p, ex, ey, ew, base_h, side)

    def _draw_brow(self, p, ex, ey, ew, base_h, side):
        if self.brow_alpha < 0.05:
            return
        color = QColor(self.brow_color)
        color.setAlphaF(min(1.0, self.brow_alpha))
        p.setPen(QPen(color, ew * 0.13, Qt.SolidLine, Qt.RoundCap))
        y = ey - base_h * self.scale * 0.5 - base_h * 0.2
        tilt = self.brow * base_h * 0.16
        inner_x = ex - side * ew * 0.5   # 靠近脸中线的一端
        outer_x = ex + side * ew * 0.5
        # brow > 0 生气：内端压低；brow < 0 担心：内端抬高
        p.drawLine(QPointF(inner_x, y + tilt), QPointF(outer_x, y - tilt))

    def _draw_blush(self, p, x, y, unit):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(224, 38, 47, 110))
        p.drawEllipse(QPointF(x, y), unit * 0.28, unit * 0.14)

    def _draw_zzz(self, p, x, y, unit, t):
        for i in range(3):
            phase = (t * 0.35 + i / 3.0) % 1.0
            alpha = int(255 * math.sin(phase * math.pi))
            font = QFont(theme.FONT_TITLE)
            font.setPixelSize(int(unit * (0.25 + phase * 0.3)))
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor(244, 246, 250, alpha))
            p.drawText(QPointF(x + phase * unit * 0.6, y - phase * unit * 1.1), "Z")

    def _draw_info(self, p, w):
        font = QFont(theme.FONT_MONO)
        font.setPixelSize(12)
        p.setFont(font)
        p.setPen(QColor(theme.WHITE_DIM))
        if self.info_left:
            p.drawText(QPointF(14, 22), self.info_left)
        if self.info_right:
            fm = QFontMetrics(font)
            p.drawText(QPointF(w - 14 - fm.horizontalAdvance(self.info_right), 22), self.info_right)

    def _draw_thumbnail(self, p, w, h):
        if self.thumbnail is None or self.thumbnail.isNull():
            return None
        tw = max(120, w * 0.2)
        th = tw * self.thumbnail.height() / max(1, self.thumbnail.width())
        rect = QRectF(w - tw - 14, h - th - 14, tw, th)
        img = self.thumbnail.scaled(int(tw), int(th), Qt.KeepAspectRatio, Qt.FastTransformation)
        p.drawImage(rect.topLeft(), img)
        p.setPen(QPen(QColor(theme.BLUE_LIGHT), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        return rect

    def _draw_caption(self, p, w, h, thumb_rect):
        if self.caption_alpha < 0.03 or not self.caption_shown:
            return
        font = QFont(theme.FONT_BODY)
        font.setPixelSize(max(16, int(min(w, h * 1.6) * 0.03)))
        font.setBold(True)
        fm = QFontMetrics(font)
        bh = fm.height() + 24
        cap_top = h * 0.82 - bh / 2
        # 和画中画在垂直方向重叠时，把字幕上移到小窗上方，而不是压缩宽度（英文句子较长）
        if thumb_rect is not None and cap_top + bh > thumb_rect.top() - 8:
            cap_top = thumb_rect.top() - bh - 12
        max_w = max(200, w - 56)
        text = fm.elidedText(self.caption_shown, Qt.ElideRight, int(max_w - 48))
        bw = fm.horizontalAdvance(text) + 48
        rect = QRectF((w - bw) / 2, cap_top, bw, bh)

        p.setOpacity(self.caption_alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.WHITE))
        p.drawRect(rect)
        p.setBrush(RED)
        p.drawRect(QRectF(rect.left(), rect.top(), 6, rect.height()))
        p.setBrush(QColor(theme.BLUE))
        p.drawRect(QRectF(rect.left(), rect.bottom() - 3, rect.width(), 3))
        p.setFont(font)
        p.setPen(QColor(theme.NAVY_DEEP))
        p.drawText(rect.adjusted(14, 0, -8, 0), Qt.AlignCenter, text)
        p.setOpacity(1.0)
