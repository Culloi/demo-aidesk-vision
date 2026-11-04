import argparse
import json
import os
import sys
import time
from pathlib import Path

from usb_person_detector.audio.speech import SpeechPlayer


def _fix_qt_plugin_path():
    """opencv-python 会把 QT_QPA_PLATFORM_PLUGIN_PATH 指向自带的 Qt 插件，与 PySide6 冲突。"""
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "/cv2/qt/plugins" not in current:
        return

    root = Path(__file__).resolve().parents[3] / ".venv/lib"
    for lib_dir in sorted(root.glob("python*/site-packages/PySide6/Qt/plugins/platforms")):
        if lib_dir.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(lib_dir)
            break


import cv2
_fix_qt_plugin_path()
import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from usb_person_detector.core.companion import (
    CAT, CELL_PHONE, DOG, EXPR_SLEEP, FIRE, NEAR, PERSON, PRESENT, SLEEP,
    CompanionBrain, CompanionConfig, CompanionState,
)
from usb_person_detector.core.i18n import LANGUAGES, get_language, set_language, tr
from usb_person_detector.entrypoints.env_check import FIX_COMMANDS, pyside_none_leak
from usb_person_detector.ui import theme
from usb_person_detector.ui.face_widget import FaceWidget
from usb_person_detector.vision.detector import (
    DEFAULT_CLASSES, CameraNotFoundError, LatestFrameReader, create_detector, draw_hud_overlay,
    fire_color_check, fire_roi, load_config, open_camera,
)


SIDE_PANEL_WIDTH = 420
MIN_WINDOW_SIZE = (960, 560)  # 兼顾桌面机器人常见的 1024x600 小屏

# 视频画面检测标签：类别 -> (翻译键, 底色, 文字色)
LABEL_STYLE = {
    PERSON: ("label.person", theme.RED, theme.WHITE),
    CELL_PHONE: ("label.phone", theme.YELLOW, theme.NAVY_DEEP),
    CAT: ("label.cat", theme.BLUE, theme.WHITE),
    DOG: ("label.dog", theme.BLUE, theme.WHITE),
}


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def draw_labels(image: QImage, detections):
    """用 Qt 在画面上绘制检测标签（支持中文；工作线程中对 QImage 绘图是安全的）。"""
    if not detections:
        return
    w = image.width()
    font = QFont(theme.FONT_BODY)
    font.setPixelSize(max(14, image.height() // 40))
    font.setBold(True)
    fm = QFontMetrics(font)
    counters = {}
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.setFont(font)
    for det in detections:
        key, bg, fg = LABEL_STYLE.get(det.cls_id, (None, theme.WHITE, theme.NAVY_DEEP))
        name = tr(key) if key else det.label
        counters[name] = counters.get(name, 0) + 1
        text = f"{name}-{counters[name]:02d}  {det.score:.2f}" if det.cls_id == PERSON else f"{name}  {det.score:.2f}"
        tag_w, tag_h = fm.horizontalAdvance(text) + 20, fm.height() + 8
        x = min(det.x1, max(0.0, w - tag_w))                # 贴近右边缘时左移
        y = det.y1 - tag_h if det.y1 >= tag_h else det.y1   # 贴近顶部时放到框内
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(bg))
        p.drawRect(QRectF(x, y, tag_w, tag_h))
        p.setBrush(QColor(theme.WHITE if bg == theme.YELLOW else theme.YELLOW))
        p.drawRect(QRectF(x, y, 5, tag_h))
        p.setPen(QColor(fg))
        p.drawText(QRectF(x + 11, y, tag_w - 11, tag_h), Qt.AlignVCenter | Qt.AlignLeft, text)
    p.end()


def repolish(widget: QWidget):
    """动态属性改变后刷新 QSS。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


# ============================================================
# 语音指令工作线程（点击触发一次录音+识别，模型只加载一次常驻在这个线程里）
# ============================================================
class VoiceCommandWorker(QObject):
    listening = Signal()
    result_ready = Signal(object, str)   # VoiceCommand 或 None, 识别到的原始文本
    error = Signal(str)

    def __init__(self, model_dir: str, mic_device: str = None, duration_sec: float = 4.0):
        super().__init__()
        self.model_dir = model_dir
        self.mic_device = mic_device
        self.duration_sec = duration_sec
        self._pipeline = None

    def _ensure_ready(self):
        if self._pipeline is not None:
            return
        from usb_person_detector.audio.voice_command import (
            AudioRecorder,
            SherpaSenseVoiceAsr,
            VoiceCommandPipeline,
        )
        asr = SherpaSenseVoiceAsr(self.model_dir)
        recorder = AudioRecorder(device=self.mic_device)
        self._pipeline = VoiceCommandPipeline(asr=asr, recorder=recorder)

    @Slot()
    def listen_once(self):
        try:
            self._ensure_ready()
            self.listening.emit()
            cmd = self._pipeline.listen_once(self.duration_sec)
            self.result_ready.emit(cmd, cmd.text if cmd else "")
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 采集 + 推理 + 陪伴逻辑 工作线程
# ============================================================
class CameraWorker(QObject):
    frame_ready = Signal(QImage)
    state_ready = Signal(object, float)        # CompanionState, fps
    events_ready = Signal(list)                # List[CompanionEvent]
    log_ready = Signal(str)
    capture_saved = Signal(str)
    error = Signal(str)
    started_ok = Signal(str, str)              # 摄像头名称（#0 / video27）, 分辨率
    backend_ready = Signal(str, str)           # 推理方式（npu / cpu）, 实际加载的模型文件名
    finished = Signal()

    def __init__(self, model_path: str, camera, width: int, height: int, conf_thres: float, iou_thres: float,
                 class_conf: dict, companion_cfg: CompanionConfig, save_dir: str, sleep_fps: float,
                 fourcc: str = "MJPG", num_threads: int = 0, backend: str = "auto", rknn_model: str = None,
                 npu_core: str = "auto", fire_model: str = None, fire_rknn_model: str = None,
                 fire_interval: int = 3, fire_crop: str = "auto", fire_crop_ratio: float = 0.75,
                 fire_color_check_enabled: bool = False, fire_min_orange_ratio: float = 0.02,
                 fire_static_filter: bool = True, fire_static_min_std: float = 1.2,
                 fire_static_window: int = 5):
        super().__init__()
        self.model_path = model_path
        self.fire_model = fire_model
        self.fire_rknn_model = fire_rknn_model
        self.fire_interval = max(1, int(fire_interval))   # 每 N 帧跑一次火焰检测
        # 火焰专用中心裁剪（数字变焦）：只影响火焰模型，主检测始终用整帧
        self.fire_crop = str(fire_crop or "auto").lower()
        self.fire_crop_ratio = float(fire_crop_ratio)
        # 颜色二次校验（可选，默认关）：滤掉纯红 UI。注意会误伤纯红色火焰
        self.fire_color_check = bool(fire_color_check_enabled)
        self.fire_min_orange_ratio = float(fire_min_orange_ratio)
        # 静止元素过滤（默认开）：屏幕 UI 亮度恒定，真火焰一定闪烁
        self.fire_static_filter = bool(fire_static_filter)
        self.fire_static_min_std = float(fire_static_min_std)
        self.fire_static_window = max(2, int(fire_static_window))
        self.fire_bright_hist = []
        self.num_threads = num_threads
        self.backend = backend
        self.rknn_model = rknn_model
        self.npu_core = npu_core
        self.camera = camera
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_conf = class_conf
        self.companion_cfg = companion_cfg
        self.save_dir = Path(save_dir)
        self.sleep_fps = max(0.5, sleep_fps)
        self.running = False
        self.capture_requested = False

    @Slot()
    def run(self):
        cap = None
        reader = None
        try:
            detector, fallback = create_detector(self.model_path, backend=self.backend, rknn_model=self.rknn_model,
                                                 npu_core=self.npu_core, conf_thres=self.conf_thres,
                                                 iou_thres=self.iou_thres, classes=DEFAULT_CLASSES,
                                                 class_conf=self.class_conf, num_threads=self.num_threads)
            if fallback:
                self.log_ready.emit(tr("log.npu_fallback", reason=fallback))
            loaded = Path(self.rknn_model if detector.backend == "npu" else self.model_path).name
            self.backend_ready.emit(detector.backend, loaded)
            self.log_ready.emit(tr("log.backend", backend=tr(f"backend.{detector.backend}"), name=loaded))

            # 火焰检测器（可选）：独立单类模型，和主检测器并行但降频运行
            fire_detector = None
            if self.fire_model:
                try:
                    fire_detector, fire_fb = create_detector(
                        self.fire_model, backend=self.backend, rknn_model=self.fire_rknn_model,
                        npu_core=self.npu_core, conf_thres=0.25, iou_thres=self.iou_thres,
                        classes={0: "fire"}, num_threads=self.num_threads)
                    if fire_fb:
                        self.log_ready.emit(tr("log.npu_fallback", reason=fire_fb))
                    fire_loaded = Path(self.fire_rknn_model if fire_detector.backend == "npu"
                                       else self.fire_model).name
                    self.log_ready.emit(tr("log.fire_ready", name=fire_loaded,
                                            interval=self.fire_interval))
                except Exception as e:
                    self.log_ready.emit(tr("log.fire_load_fail", msg=e))
                    fire_detector = None

            brain = CompanionBrain(self.companion_cfg)
            try:
                cap, source = open_camera(self.camera, self.width, self.height, self.fourcc)
            except CameraNotFoundError:
                raise RuntimeError(tr("err.no_usb_cam"))
            if not cap.isOpened():
                raise RuntimeError(tr("err.cam_open", cam=source))

            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_name = f"#{source}" if isinstance(source, int) else Path(str(source)).name
            self.running = True
            self.started_ok.emit(cam_name, f"{real_w}x{real_h}")
            self.log_ready.emit(tr("log.cam_ok", src=cam_name, res=f"{real_w}x{real_h}"))
            reader = LatestFrameReader(cap)            # 后台取帧，推理总是用最新画面

            fps = 0.0
            last_frame = None
            frame_idx = 0
            self.fire_cycle = 0                        # 火焰检测轮次（auto 裁剪策略交替用）
            fire_dets = []                             # 降频：保留上一轮火焰结果
            while self.running:
                loop_start = time.time()
                ok, frame = reader.read()
                if not ok or frame is None:
                    self.error.emit(tr("err.cam_read"))
                    break

                brain.set_config(self.companion_cfg)   # 支持界面实时修改参数
                detections = detector.infer(frame)

                # 火焰检测降频：每 fire_interval 帧跑一次，中间复用结果
                # 火焰不是毫秒级事件，降频能保住主检测帧率（尤其板子 NPU 双模型并行时）
                if fire_detector is not None:
                    if frame_idx % self.fire_interval == 0:
                        # 裁剪策略：小火焰用裁剪（数字变焦）检出率更高，
                        # 但大火焰/边缘火焰需要整帧。auto 模式轮流用两者，
                        # 配合 CompanionBrain 的峰值聚合窗口自然合并结果，零额外算力。
                        use_crop = self.fire_crop == "center" or (
                            self.fire_crop == "auto" and self.fire_cycle % 2 == 0)
                        self.fire_cycle += 1
                        if use_crop:
                            roi, (ox, oy) = fire_roi(frame, mode="center",
                                                     ratio=self.fire_crop_ratio)
                        else:
                            roi, (ox, oy) = frame, (0, 0)
                        fire_dets = fire_detector.infer(roi)
                        if ox or oy:      # 把裁剪坐标还原回整帧，否则框会画错位置
                            for d in fire_dets:
                                d.x1 += ox; d.x2 += ox
                                d.y1 += oy; d.y2 += oy
                        # 颜色二次校验（可选）：滤掉纯红色 UI 元素
                        if self.fire_color_check:
                            kept = []
                            for d in fire_dets:
                                ok, info = fire_color_check(
                                    frame, d, self.fire_min_orange_ratio)
                                if ok:
                                    kept.append(d)
                                else:
                                    self.log_ready.emit(tr(
                                        "log.fire_reject", score=d.score,
                                        orange=info.get("orange_ratio"),
                                        box=info.get("box")))
                            fire_dets = kept
                        # 静止元素过滤：屏幕 UI 亮度恒定，真火焰一定闪烁
                        if fire_dets:
                            fire_dets = self._filter_static(fire_dets, frame)
                        else:
                            self.fire_bright_hist.clear()
                    for d in fire_dets:
                        d.cls_id = FIRE               # 重映射，避免和 COCO 的 PERSON=0 冲突
                        d.label = "fire"
                    detections = detections + fire_dets
                frame_idx += 1
                # 显示实际每秒处理的帧数（受推理速度和摄像头帧率共同限制），而不是单次推理速度
                now = time.time()
                if last_frame is not None and now > last_frame:
                    current_fps = 1.0 / (now - last_frame)
                    fps = current_fps if fps == 0 else fps * 0.85 + current_fps * 0.15
                last_frame = now

                state, events = brain.update(detections, frame.shape[1], frame.shape[0])
                vis = draw_hud_overlay(frame.copy(), detections, fps, str(source), state.presence,
                                       show_panel=False, draw_labels=False)
                image = to_qimage(vis)
                draw_labels(image, detections)

                if self.capture_requested:
                    self.capture_requested = False
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    path = self.save_dir / f"capture_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    image.save(str(path), "JPG", 92)
                    self.log_ready.emit(tr("log.photo", name=path.name))
                    self.capture_saved.emit(str(path))

                self.state_ready.emit(state, fps)
                if events:
                    self.events_ready.emit(events)
                self.frame_ready.emit(image)

                # 休眠时降低帧率，节省算力和功耗
                if state.low_power:
                    remain = 1.0 / self.sleep_fps - (time.time() - loop_start)
                    if remain > 0:
                        time.sleep(remain)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.running = False
            if reader is not None:
                reader.stop()
            if cap is not None:
                cap.release()
            self.finished.emit()

    def _filter_static(self, fire_dets, frame):
        """用亮度时域稳定性过滤静止元素（屏幕 UI 边框、图标、指示灯等）。

        原理：真实火焰会持续闪烁，框内平均亮度随时间波动；
        而摄像头拍到的屏幕 UI 是静止的，亮度几乎恒定。

        只有当采样窗口满了、且亮度标准差低于阈值时才过滤，
        这样火焰刚出现的头几帧不会被误判。
        """
        if not self.fire_static_filter:
            return fire_dets

        d = max(fire_dets, key=lambda x: x.area)      # 用面积最大的框代表当前火焰
        h, w = frame.shape[:2]
        x1 = max(0, int(d.x1)); y1 = max(0, int(d.y1))
        x2 = min(w, int(d.x2)); y2 = min(h, int(d.y2))
        if x2 <= x1 or y2 <= y1:
            return fire_dets
        roi = frame[y1:y2, x1:x2]
        bright = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())

        self.fire_bright_hist.append(bright)
        if len(self.fire_bright_hist) > self.fire_static_window:
            self.fire_bright_hist.pop(0)

        if len(self.fire_bright_hist) < self.fire_static_window:
            return fire_dets                          # 样本不足，先放过

        std = float(np.std(self.fire_bright_hist))
        if std < self.fire_static_min_std:
            self.log_ready.emit(tr("log.fire_static", std=round(std, 2),
                                   bright=round(bright, 0)))
            return []                                 # 亮度几乎不变 → 静止物体
        return fire_dets

    @Slot()
    def stop(self):
        self.running = False

    @Slot()
    def request_capture(self):
        self.capture_requested = True

    def update_companion_config(self, cfg: CompanionConfig):
        self.companion_cfg = cfg


# ============================================================
# 界面组件
# ============================================================
class Panel(QFrame):
    """带白色装甲标题条的面板；title_row 可追加控件；collapsible=True 时可折叠。"""

    def __init__(self, code: str = "", collapsible: bool = False):
        super().__init__()
        self.setObjectName("Panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.title_bar = QWidget()
        self.title_row = QHBoxLayout(self.title_bar)
        self.title_row.setContentsMargins(0, 0, 10, 0)
        self.title_row.setSpacing(6)
        self.title_label = QLabel()
        self.title_label.setObjectName("PanelTitle")
        self.title_row.addWidget(self.title_label)
        self.title_row.addStretch(1)
        if code:
            code_label = QLabel(code)
            code_label.setObjectName("PanelCode")
            self.title_row.addWidget(code_label)
        self.collapse_btn = None
        if collapsible:
            self.collapse_btn = QPushButton("−")
            self.collapse_btn.setObjectName("ViewBtn")
            self.collapse_btn.setFixedWidth(30)
            self.collapse_btn.setCursor(Qt.PointingHandCursor)
            self.collapse_btn.clicked.connect(lambda: self.set_collapsed(self.body_widget.isVisible()))
            self.title_row.addWidget(self.collapse_btn)
        outer.addWidget(self.title_bar)

        self.body_widget = QWidget()
        self.body = QVBoxLayout(self.body_widget)
        self.body.setContentsMargins(12, 10, 12, 12)
        self.body.setSpacing(8)
        outer.addWidget(self.body_widget, 1)

    def set_collapsed(self, collapsed: bool):
        self.body_widget.setVisible(not collapsed)
        if self.collapse_btn is not None:
            self.collapse_btn.setText("+" if collapsed else "−")


class InfoCard(QFrame):
    """状态卡片：左侧彩色色条 + 标题 + 数值。"""

    def __init__(self, value: str = "--", accent: str = "blue"):
        super().__init__()
        self.setObjectName("InfoCard")
        self.setMinimumHeight(58)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        self.stripe = QFrame()
        self.stripe.setObjectName("CardStripe")
        self.stripe.setFixedWidth(5)
        self.stripe.setProperty("accent", accent)
        layout.addWidget(self.stripe)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 6, 0, 6)
        text_box.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setObjectName("CardTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        text_box.addWidget(self.title_label)
        text_box.addWidget(self.value_label)
        layout.addLayout(text_box, 1)

    def set_value(self, value: str, alert: bool = False):
        if self.value_label.text() != value:
            self.value_label.setText(value)
        if self.value_label.property("alert") != alert:
            self.value_label.setProperty("alert", alert)
            repolish(self.value_label)

    def set_accent(self, accent: str):
        if self.stripe.property("accent") != accent:
            self.stripe.setProperty("accent", accent)
            repolish(self.stripe)


class VideoView(QLabel):
    """视频显示区：尺寸由布局决定，不会被图像撑大。"""

    def __init__(self):
        super().__init__()
        self.setObjectName("VideoLabel")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(480, 270)
        self._pixmap = None
        self.show_placeholder()

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    def show_placeholder(self):
        self._pixmap = None
        self.clear()
        self.setText(tr("video.no_signal"))

    def set_image(self, image: QImage):
        self._pixmap = QPixmap.fromImage(image)
        if self.isVisible():
            self._update_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    voice_listen_requested = Signal()

    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.config = load_config(config_path)
        set_language(self.config.get("language", "zh"))
        self.companion_cfg = CompanionConfig.from_dict(self.config.get("companion", {}))
        self.speech = SpeechPlayer(enabled=self.config.get("voice_enabled", True))
        self.voice_thread = None
        self.voice_worker = None
        self.voice_busy = False
        self.worker_thread = None
        self.worker = None
        self.start_ts = None
        self.robot_mode = False

        # 界面文字绑定：切换语言时逐个刷新
        self._bindings = []
        # 动态内容的最近状态，切换语言时用来重新渲染
        self.mode_key, self.mode_state = "standby", "standby"
        self.model_name = Path(self.config.get("model", "models/yolov8n.onnx")).name
        self.backend = "cpu"                # 实际推理方式，启动检测后由检测线程告知：npu / cpu
        self.cam_status = "offline"         # offline / ok / error
        self.cam_info = ("", "")            # (摄像头编号, 分辨率)
        self.last_state = None
        self.last_fps = 0.0

        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self._build_ui()
        self.setStyleSheet(theme.QSS)
        self._load_defaults()
        self._fit_to_screen()
        self.retranslate()
        self._show_offline_face()
        self._init_voice_command()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick)
        self.clock_timer.start(1000)
        self._tick()

        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self.toggle_robot_mode)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=lambda: self.robot_mode and self.toggle_robot_mode())

        self.log(tr("log.ready"))

    # ---------- 多语言 ----------
    def bind(self, widget, key: str, method: str = "setText"):
        """登记一个需要翻译的控件文字，立即生效，切换语言时自动刷新。"""
        setter = getattr(widget, method)
        apply = lambda: setter(tr(key))
        apply()
        self._bindings.append(apply)
        return widget

    def field_label(self, key: str) -> QLabel:
        label = QLabel()
        label.setObjectName("FieldLabel")
        return self.bind(label, key)

    def toggle_language(self):
        langs = list(LANGUAGES)
        new_lang = langs[(langs.index(get_language()) + 1) % len(langs)]
        set_language(new_lang)
        self.retranslate()
        self._save_language(new_lang)
        self.log(tr("log.lang"))

    def retranslate(self):
        for apply in self._bindings:
            apply()
        self.setWindowTitle(tr("app.title"))
        self.face.setToolTip(tr("face.toggle_fullscreen_tip"))
        self.set_mode(self.mode_key, self.mode_state)
        self._update_footer()
        self._tick()
        if not self.video_view.has_image:
            self.video_view.show_placeholder()
        if self.last_state is not None and self.worker is not None:
            self.on_state(self.last_state, self.last_fps)
        elif self.worker is None:
            self.face.set_info(tr("face.offline"), time.strftime("%H:%M"))

    def _save_language(self, lang: str):
        """只更新 config.json 的 language 字段，保留其他配置。"""
        try:
            data = load_config(self.config_path)
            data["language"] = lang
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.config["language"] = lang
        except OSError as e:
            self.log(tr("log.error", msg=e))

    # ---------- 语音播报 ----------
    def _update_voice_btn(self):
        key = "voice.on" if self.speech.enabled else "voice.off"
        tip_key = "voice.tooltip_on" if self.speech.enabled else "voice.tooltip_off"
        self.voice_btn.setText(tr(key))
        self.voice_btn.setToolTip(tr(tip_key))

    def toggle_voice(self):
        self.speech.set_enabled(not self.speech.enabled)
        self._update_voice_btn()
        self._save_voice_enabled(self.speech.enabled)
        self.log(tr("log.voice_on") if self.speech.enabled else tr("log.voice_off"))

    def _save_voice_enabled(self, enabled: bool):
        try:
            data = load_config(self.config_path)
            data["voice_enabled"] = enabled
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.config["voice_enabled"] = enabled
        except OSError as e:
            self.log(tr("log.error", msg=e))

    # ---------- 布局 ----------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = self._build_header()
        outer.addWidget(self.header)

        self.body_layout = QHBoxLayout()
        self.body_layout.setContentsMargins(14, 14, 14, 14)
        self.body_layout.setSpacing(14)
        self.main_panel = self._build_main_panel()
        self.side_panel = self._build_side_panel()
        self.body_layout.addWidget(self.main_panel, 1)
        self.body_layout.addWidget(self.side_panel)
        outer.addLayout(self.body_layout, 1)

        self.footer = self._build_footer()
        outer.addWidget(self.footer)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(72)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 18, 0)
        layout.setSpacing(16)

        emblem = QLabel("RX-78")
        emblem.setObjectName("Emblem")
        emblem.setAlignment(Qt.AlignCenter)
        emblem.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout.addWidget(emblem)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.setContentsMargins(0, 10, 0, 10)
        title = self.bind(QLabel(), "header.title")
        title.setObjectName("MainTitle")
        subtitle = self.bind(QLabel(), "header.subtitle")
        subtitle.setObjectName("SubTitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch(1)

        self.voice_btn = QPushButton()
        self.voice_btn.setObjectName("LangBtn")
        self.voice_btn.setCursor(Qt.PointingHandCursor)
        self.voice_btn.clicked.connect(self.toggle_voice)
        self._update_voice_btn()
        layout.addWidget(self.voice_btn)

        self.lang_btn = self.bind(QPushButton(), "lang.switch")
        self.bind(self.lang_btn, "lang.tooltip", "setToolTip")
        self.lang_btn.setObjectName("LangBtn")
        self.lang_btn.setCursor(Qt.PointingHandCursor)
        self.lang_btn.clicked.connect(self.toggle_language)
        layout.addWidget(self.lang_btn)

        stripes = QHBoxLayout()
        stripes.setSpacing(4)
        for name, width in (("StripeBlue", 34), ("StripeRed", 22), ("StripeYellow", 10)):
            bar = QFrame()
            bar.setObjectName(name)
            bar.setFixedSize(width, 28)
            stripes.addWidget(bar)
        layout.addLayout(stripes)

        self.mode_label = QLabel()
        self.mode_label.setObjectName("ModeBadge")
        self.mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mode_label)
        return header

    def _build_main_panel(self) -> QWidget:
        panel = Panel()
        self.bind(panel.title_label, "panel.main")
        self.main_panel_obj = panel

        # 标题栏上的视图切换按钮
        self.view_group = QButtonGroup(self)
        self.face_btn = self.bind(QPushButton(), "view.face")
        self.cam_btn = self.bind(QPushButton(), "view.camera")
        for i, btn in enumerate((self.face_btn, self.cam_btn)):
            btn.setObjectName("ViewBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            self.view_group.addButton(btn, i)
            panel.title_row.addWidget(btn)
        self.robot_btn = self.bind(QPushButton(), "view.fullscreen")
        self.bind(self.robot_btn, "view.fullscreen_tip", "setToolTip")
        self.robot_btn.setObjectName("RobotBtn")
        self.robot_btn.setCursor(Qt.PointingHandCursor)
        self.robot_btn.clicked.connect(self.toggle_robot_mode)
        panel.title_row.addWidget(self.robot_btn)
        self.view_group.idClicked.connect(self.switch_view)

        self.stack = QStackedWidget()
        self.stack.setObjectName("Stack")
        self.face = FaceWidget()
        self.face.double_clicked.connect(self.toggle_robot_mode)
        self.video_view = VideoView()
        self.stack.addWidget(self.face)
        self.stack.addWidget(self.video_view)
        panel.body.addWidget(self.stack, 1)
        return panel

    def _build_side_panel(self) -> QWidget:
        """右侧：上方信息区可滚动（小屏不重叠），下方操作按钮固定（任何屏幕都能直接点到）。"""
        container = QWidget()
        container.setFixedWidth(SIDE_PANEL_WIDTH + 12)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setObjectName("SideScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container_layout.addWidget(scroll, 1)

        side = QWidget()
        side.setObjectName("SideContent")
        scroll.setWidget(side)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(12)

        # --- 陪伴状态 ---
        status = Panel("SYS-01")
        self.bind(status.title_label, "panel.status")
        grid = QGridLayout()
        grid.setSpacing(8)
        self.presence_card = InfoCard("--", "blue")
        self.distance_card = InfoCard("--", "white")
        self.person_card = InfoCard("0", "red")
        self.phone_card = InfoCard("--", "yellow")
        self.pet_card = InfoCard("--", "blue")
        # 在场状态最重要，独占第一行；其余四张两两一行
        cards = [(self.presence_card, "card.presence", 0, 0, 2), (self.distance_card, "card.distance", 1, 0, 1),
                 (self.person_card, "card.persons", 1, 1, 1), (self.phone_card, "card.phone", 2, 0, 1),
                 (self.pet_card, "card.pet", 2, 1, 1)]
        for card, key, row, col, span in cards:
            self.bind(card.title_label, key)
            grid.addWidget(card, row, col, 1, span)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        status.body.addLayout(grid)
        layout.addWidget(status)

        # --- 陪伴参数（运行中可实时调整） ---
        comp = Panel("CMP-02")
        self.bind(comp.title_label, "panel.rules")
        self.phone_spin = QSpinBox()
        self.phone_spin.setRange(3, 600)
        self.sleep_spin = QSpinBox()
        self.sleep_spin.setRange(10, 3600)
        self.sleep_spin.setSingleStep(10)
        rules = QGridLayout()
        rules.setHorizontalSpacing(8)
        rules.setVerticalSpacing(4)
        rules.addWidget(self.field_label("rule.phone"), 0, 0)
        rules.addWidget(self.field_label("rule.sleep"), 0, 1)
        rules.addWidget(self.phone_spin, 1, 0)
        rules.addWidget(self.sleep_spin, 1, 1)
        for col in range(2):
            rules.setColumnStretch(col, 1)
        comp.body.addLayout(rules)
        layout.addWidget(comp)

        # --- 检测参数（运行时锁定并自动折叠） ---
        control = Panel("CFG-03", collapsible=True)
        self.bind(control.title_label, "panel.detection")
        self.detection_panel = control
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self.model_edit = QLineEdit()
        self.browse_btn = self.bind(QPushButton(), "btn.browse")
        self.browse_btn.setObjectName("BtnBrowse")
        self.browse_btn.clicked.connect(self.browse_model)
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(self.browse_btn)
        control.body.addWidget(self.field_label("field.model"))
        control.body.addLayout(model_row)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(-1, 63)              # -1 = 自动查找 USB 摄像头
        self.bind(self.camera_spin, "field.camera_auto", "setSpecialValueText")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setSingleStep(160)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setSingleStep(90)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)

        params = QGridLayout()
        params.setHorizontalSpacing(8)
        params.setVerticalSpacing(4)
        params.addWidget(self.field_label("field.camera"), 0, 0)
        params.addWidget(self.field_label("field.width"), 0, 1)
        params.addWidget(self.field_label("field.height"), 0, 2)
        params.addWidget(self.camera_spin, 1, 0)
        params.addWidget(self.width_spin, 1, 1)
        params.addWidget(self.height_spin, 1, 2)
        params.addWidget(self.field_label("field.conf"), 2, 0)
        params.addWidget(self.field_label("field.iou"), 2, 1)
        params.addWidget(self.conf_spin, 3, 0)
        params.addWidget(self.iou_spin, 3, 1)
        for col in range(3):
            params.setColumnStretch(col, 1)
        control.body.addLayout(params)
        layout.addWidget(control)

        self.param_widgets = [self.model_edit, self.browse_btn, self.camera_spin, self.width_spin,
                              self.height_spin, self.conf_spin, self.iou_spin]

        # --- 日志 ---
        log_panel = Panel("LOG-04")
        self.bind(log_panel.title_label, "panel.log")
        self.log_list = QListWidget()
        self.log_list.setWordWrap(True)
        self.log_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_list.setMinimumHeight(110)
        log_panel.body.addWidget(self.log_list, 1)
        layout.addWidget(log_panel, 1)

        # --- 操作按钮（滚动区外，始终可见） ---
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 12, 0)
        actions.setSpacing(8)
        self.start_btn = self.bind(QPushButton(), "btn.start")
        self.start_btn.setObjectName("BtnStart")
        self.stop_btn = self.bind(QPushButton(), "btn.stop")
        self.stop_btn.setObjectName("BtnStop")
        self.capture_btn = self.bind(QPushButton(), "btn.capture")
        self.capture_btn.setObjectName("BtnCapture")
        self.voice_cmd_btn = self.bind(QPushButton(), "btn.voice_cmd")
        self.voice_cmd_btn.setObjectName("BtnCapture")
        for btn in (self.start_btn, self.stop_btn, self.capture_btn, self.voice_cmd_btn):
            btn.setMinimumHeight(46)
            btn.setCursor(Qt.PointingHandCursor)
            actions.addWidget(btn, 1)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.voice_cmd_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.capture_btn.clicked.connect(self.capture_frame)
        self.voice_cmd_btn.clicked.connect(self.on_voice_cmd_clicked)
        container_layout.addLayout(actions)
        return container

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        footer.setFixedHeight(28)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(14, 0, 14, 0)
        self.footer_left = QLabel()
        self.footer_left.setObjectName("FooterText")
        self.footer_mid = QLabel()
        self.footer_mid.setObjectName("FooterText")
        self.footer_right = QLabel()
        self.footer_right.setObjectName("FooterText")
        layout.addWidget(self.footer_left)
        layout.addStretch(1)
        layout.addWidget(self.footer_mid)
        layout.addSpacing(24)
        layout.addWidget(self.footer_right)
        return footer

    def _fit_to_screen(self):
        want_w = int(self.config.get("gui_window_width", 1480))
        want_h = int(self.config.get("gui_window_height", 920))
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            want_w = min(want_w, int(avail.width() * 0.95))
            want_h = min(want_h, int(avail.height() * 0.92))
        self.resize(max(want_w, self.minimumWidth()), max(want_h, self.minimumHeight()))
        if screen is not None:  # 居中显示
            avail = screen.availableGeometry()
            self.move(avail.x() + (avail.width() - self.width()) // 2,
                      avail.y() + (avail.height() - self.height()) // 2)

    def _load_defaults(self):
        self.model_edit.setText(self.config.get("model", "models/yolov8n.onnx"))
        camera = str(self.config.get("camera", "auto")).strip().lower()
        self.camera_spin.setValue(-1 if camera in ("auto", "-1") else int(camera))
        self.width_spin.setValue(int(self.config.get("width", 1280)))
        self.height_spin.setValue(int(self.config.get("height", 720)))
        self.conf_spin.setValue(float(self.config.get("class_conf", {}).get("person", self.config.get("conf_thres", 0.4))))
        self.iou_spin.setValue(float(self.config.get("iou_thres", 0.5)))

        self.phone_spin.setValue(int(self.companion_cfg.phone_alert_sec))
        self.sleep_spin.setValue(int(self.companion_cfg.sleep_after_sec))
        for spin in (self.phone_spin, self.sleep_spin):
            spin.valueChanged.connect(self.on_rules_changed)

        default_view = 0 if self.config.get("default_view", "face") == "face" else 1
        self.view_group.button(default_view).setChecked(True)
        self.stack.setCurrentIndex(default_view)

    # ---------- 视图 / 全屏模式 ----------
    @Slot(int)
    def switch_view(self, index: int):
        self.stack.setCurrentIndex(index)

    def toggle_robot_mode(self):
        """全屏模式：只显示表情脸，即实际产品在机器人屏幕上的形态。"""
        self.robot_mode = not self.robot_mode
        for w in (self.header, self.side_panel, self.footer, self.main_panel_obj.title_bar):
            w.setVisible(not self.robot_mode)
        margin = 0 if self.robot_mode else 14
        self.body_layout.setContentsMargins(margin, margin, margin, margin)
        self.main_panel_obj.body.setContentsMargins(*((0, 0, 0, 0) if self.robot_mode else (12, 10, 12, 12)))
        if self.robot_mode:
            self.stack.setCurrentIndex(0)
            self.showFullScreen()
        else:
            self.stack.setCurrentIndex(self.view_group.checkedId())
            self.showNormal()

    # ---------- 工具 ----------
    def set_mode(self, key: str, state: str):
        """key: standby / booting / running / stopping / sleep / error"""
        self.mode_key, self.mode_state = key, state
        self.mode_label.setText(tr(f"mode.{key}"))
        if self.mode_label.property("state") != state:
            self.mode_label.setProperty("state", state)
            repolish(self.mode_label)

    def set_running_ui(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.capture_btn.setEnabled(running)
        for w in self.param_widgets:
            w.setEnabled(not running)
        self.detection_panel.set_collapsed(running)

    def log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_list.insertItem(0, QListWidgetItem(f"{stamp}  {text}"))
        while self.log_list.count() > 200:
            self.log_list.takeItem(self.log_list.count() - 1)

    def _update_footer(self):
        self.footer_left.setText(tr("footer.system", backend=tr(f"backend.{self.backend}"), model=self.model_name))
        if self.cam_status == "error":
            self.footer_mid.setText(tr("footer.cam_error"))
        elif self.cam_status == "offline":
            self.footer_mid.setText(tr("footer.cam_offline"))
        else:
            cam = tr("footer.cam", cam=self.cam_info[0], res=self.cam_info[1])
            low_power = self.last_state is not None and self.last_state.low_power
            fps = tr("footer.low_power") if low_power else f"{self.last_fps:.1f} FPS"
            self.footer_mid.setText(f"{cam}  ·  {fps}")

    def _tick(self):
        uptime = "--:--:--"
        if self.start_ts is not None:
            s = int(time.time() - self.start_ts)
            uptime = f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
        self.footer_right.setText(f"{tr('footer.runtime', uptime=uptime)}   ·   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _show_offline_face(self):
        self.face.set_expression(EXPR_SLEEP)
        self.face.set_gaze(None)
        self.face.set_caption("")
        self.face.set_thumbnail(None)
        self.face.set_info(tr("face.offline"), time.strftime("%H:%M"))

    def _current_rules(self) -> CompanionConfig:
        cfg = CompanionConfig(**vars(self.companion_cfg))
        cfg.phone_alert_sec = float(self.phone_spin.value())
        cfg.sleep_after_sec = float(self.sleep_spin.value())
        return cfg

    @Slot()
    def on_rules_changed(self):
        self.companion_cfg = self._current_rules()
        if self.worker is not None:
            self.worker.update_companion_config(self.companion_cfg)

    # ---------- 操作 ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("dlg.choose_model"), self.model_edit.text(), tr("dlg.onnx_filter"))
        if path:
            self.model_edit.setText(path)

    def start_detection(self):
        if self.worker_thread is not None:
            return

        model_path = Path(self.model_edit.text().strip())
        if not model_path.exists():
            QMessageBox.critical(self, tr("dlg.model_missing_title"), tr("dlg.model_missing", path=model_path))
            return

        class_conf = dict(self.config.get("class_conf", {}))
        class_conf["person"] = self.conf_spin.value()

        # 火焰检测：开关打开时传模型路径，否则不启用
        fire_model = fire_rknn = None
        if self.config.get("fire_enabled", False):
            fire_model = self.config.get("fire_model", "models/fire_yolov8n.onnx")
            fire_rknn = self.config.get("fire_rknn_model", "models/fire_rk3576_fp16.rknn")
            if not Path(fire_model).exists() and not (fire_rknn and Path(fire_rknn).exists()):
                self.log(tr("log.fire_load_fail", msg=tr("err.fire_model_missing", path=fire_model)))
                fire_model = fire_rknn = None

        self.worker_thread = QThread(self)
        self.worker = CameraWorker(
            model_path=str(model_path),
            camera="auto" if self.camera_spin.value() < 0 else self.camera_spin.value(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            conf_thres=self.conf_spin.value(),
            iou_thres=self.iou_spin.value(),
            class_conf=class_conf,
            companion_cfg=self._current_rules(),
            save_dir=self.config.get("save_dir", "captures"),
            sleep_fps=float(self.config.get("sleep_fps", 2)),
            fourcc=self.config.get("camera_fourcc", "MJPG"),
            num_threads=int(self.config.get("cpu_threads", 4)),
            backend=self.config.get("backend", "auto"),
            rknn_model=self.config.get("rknn_model"),
            npu_core=str(self.config.get("npu_core", "auto")),
            fire_model=fire_model,
            fire_rknn_model=fire_rknn,
            fire_interval=int(self.config.get("fire_interval", 3)),
            fire_crop=str(self.config.get("fire_crop", "auto")),
            fire_crop_ratio=float(self.config.get("fire_crop_ratio", 0.75)),
            fire_color_check_enabled=bool(self.config.get("fire_color_check", False)),
            fire_min_orange_ratio=float(self.config.get("fire_min_orange_ratio", 0.02)),
            fire_static_filter=bool(self.config.get("fire_static_filter", True)),
            fire_static_min_std=float(self.config.get("fire_static_min_std", 1.2)),
            fire_static_window=int(self.config.get("fire_static_window", 5)),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_ready.connect(self.on_state)
        self.worker.events_ready.connect(self.on_events)
        self.worker.log_ready.connect(self.log)
        self.worker.capture_saved.connect(self.on_capture_saved)
        self.worker.error.connect(self.on_worker_error)
        self.worker.started_ok.connect(self.on_started_ok)
        self.worker.backend_ready.connect(self.on_backend_ready)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

        self.set_running_ui(True)
        self.capture_btn.setEnabled(False)
        self.model_name = model_path.name
        self._update_footer()
        self.set_mode("booting", "booting")

    def stop_detection(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.set_mode("stopping", "booting")

    def capture_frame(self):
        if self.worker is not None:
            self.worker.request_capture()

    # ---------- 语音指令 ----------
    def _init_voice_command(self):
        model_dir = Path(self.config.get("asr_model_dir",
                                          "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
        ready = (model_dir / "model.int8.onnx").exists() and (model_dir / "tokens.txt").exists()
        if ready:
            try:
                import sherpa_onnx  # noqa: F401
            except ImportError:
                ready = False
        if not ready:
            self.voice_cmd_btn.setEnabled(False)
            self.voice_cmd_btn.setToolTip(tr("btn.voice_cmd_unavailable"))
            self.log(tr("log.voice_unavailable"))
            return

        duration = float(self.config.get("voice_cmd_duration_sec", 4.0))
        mic_device = self.config.get("mic_device") or None
        self.voice_thread = QThread(self)
        self.voice_worker = VoiceCommandWorker(str(model_dir), mic_device=mic_device, duration_sec=duration)
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_listen_requested.connect(self.voice_worker.listen_once)
        self.voice_worker.listening.connect(self.on_voice_listening)
        self.voice_worker.result_ready.connect(self.on_voice_result)
        self.voice_worker.error.connect(self.on_voice_error)
        self.voice_thread.start()
        self.voice_cmd_btn.setEnabled(True)

    def on_voice_cmd_clicked(self):
        if self.voice_busy or self.voice_worker is None:
            return
        self.voice_busy = True
        self.voice_cmd_btn.setEnabled(False)
        self.log(tr("log.voice_listening"))
        self.face.set_caption(tr("caption.voice_listening"))
        self.voice_listen_requested.emit()

    @Slot()
    def on_voice_listening(self):
        pass  # 预留：录音开始时的额外反馈（比如闪烁图标）

    @Slot(object, str)
    def on_voice_result(self, cmd, text: str):
        self.voice_busy = False
        self.voice_cmd_btn.setEnabled(True)
        self.face.set_caption("")
        if not text:
            self.log(tr("log.voice_no_speech"))
            return
        self.log(tr("log.voice_heard", text=text))
        if cmd is None:
            self.log(tr("log.voice_no_match"))
            return
        self._run_voice_intent(cmd.intent)

    @Slot(str)
    def on_voice_error(self, message: str):
        self.voice_busy = False
        self.voice_cmd_btn.setEnabled(True)
        self.face.set_caption("")
        self.log(tr("log.voice_error", msg=message))

    def _run_voice_intent(self, intent: str):
        if intent == "capture":
            if self.worker is not None:
                self.capture_frame()
            else:
                self.log(tr("log.voice_need_camera"))
        elif intent == "lang_switch":
            self.toggle_language()
        elif intent == "mute":
            if self.speech.enabled:
                self.toggle_voice()
        elif intent == "unmute":
            if not self.speech.enabled:
                self.toggle_voice()
        elif intent == "fullscreen_on":
            if not self.robot_mode:
                self.toggle_robot_mode()
        elif intent == "fullscreen_off":
            if self.robot_mode:
                self.toggle_robot_mode()
        elif intent == "greet":
            self.log(tr("log.voice_greet"))

    # ---------- 信号回调 ----------
    @Slot(str, str)
    def on_backend_ready(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name
        self._update_footer()

    @Slot(str, str)
    def on_started_ok(self, cam_name: str, resolution: str):
        self.start_ts = time.time()
        self.cam_status = "ok"
        self.cam_info = (cam_name, resolution)
        self._update_footer()
        self.capture_btn.setEnabled(True)
        self.set_mode("running", "running")

    @Slot(QImage)
    def on_frame(self, image: QImage):
        self.video_view.set_image(image)
        self.face.set_thumbnail(image if self.config.get("face_pip", True) else None)

    @Slot(object, float)
    def on_state(self, st: CompanionState, fps: float):
        self.last_state, self.last_fps = st, fps
        present = st.presence == PRESENT
        presence_text = tr(f"presence.{st.presence}")
        distance_text = tr(f"distance.{st.distance}") if st.distance != "--" else ""

        self.presence_card.set_value(presence_text, alert=present)
        self.presence_card.set_accent({PRESENT: "red", SLEEP: "white"}.get(st.presence, "blue"))
        self.distance_card.set_value(f"{distance_text}  {st.distance_ratio:.2f}" if distance_text else "--",
                                     alert=st.distance == NEAR)
        self.phone_card.set_value(tr("phone.in_use", sec=int(st.phone_seconds)) if st.phone_in_use else "--",
                                  alert=st.phone_in_use)
        self.pet_card.set_value(st.pet or "--")
        self.person_card.set_value(str(st.person_count), alert=st.person_count > 0)
        self._update_footer()

        if st.presence == SLEEP:
            if self.mode_key != "sleep":
                self.set_mode("sleep", "standby")
        elif self.mode_key != "running":
            self.set_mode("running", "running")

        self.face.set_expression(st.expression)
        self.face.set_gaze(st.gaze)
        self.face.set_caption(st.caption)
        right = time.strftime("%H:%M")
        left = tr("face.presence_distance", presence=presence_text, distance=distance_text) if distance_text \
            else presence_text
        self.face.set_info(left, right)

    @Slot(list)
    def on_events(self, events):
        for ev in events:
            tag = tr(f"tag.{ev.kind}")
            self.log(f"{tag if tag != f'tag.{ev.kind}' else tr('tag.event')} {ev.message}")
            self.speech.speak_event(ev, get_language())

    @Slot(str)
    def on_capture_saved(self, path: str):
        self.face.set_caption(tr("caption.photo"))
        QTimer.singleShot(2500, lambda: self.face.set_caption(""))

    @Slot(str)
    def on_worker_error(self, message: str):
        self.log(tr("log.error", msg=message))
        self.set_mode("error", "error")
        self.cam_status = "error"
        self._update_footer()
        QMessageBox.warning(self, tr("dlg.error_title"), message)

    @Slot()
    def on_worker_finished(self):
        self.start_ts = None
        self.last_state = None
        if self.mode_state != "error":
            self.set_mode("standby", "standby")
            self.cam_status = "offline"
        self._update_footer()
        self.set_running_ui(False)
        self.video_view.show_placeholder()
        for card, value in ((self.presence_card, "--"), (self.distance_card, "--"), (self.phone_card, "--"),
                            (self.pet_card, "--"), (self.person_card, "0")):
            card.set_value(value)
        self.presence_card.set_accent("blue")
        self._show_offline_face()
        self.log(tr("log.stopped"))

    @Slot()
    def cleanup_worker(self):
        if self.worker is not None:
            self.worker.deleteLater()
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        if self.voice_thread is not None:
            self.voice_thread.quit()
            self.voice_thread.wait(2000)
        self.speech.shutdown()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="RX-78 桌面陪伴机器人 / RX-78 Desktop Companion")
    parser.add_argument("--config", default="config.json", help="配置文件路径 / config file")
    parser.add_argument("--robot", action="store_true", help="启动即进入全屏模式 / start in fullscreen")
    parser.add_argument("--lang", choices=list(LANGUAGES), help="界面语言，覆盖配置文件 / UI language")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    set_language(args.lang or load_config(config_path).get("language", "zh"))
    app = QApplication(sys.argv)
    # 有问题的 PySide6 版本会在运行几分钟后闪退，启动时就拦住并给出修复方法
    if pyside_none_leak():
        import PySide6
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        msg = tr("dlg.pyside_msg", version=PySide6.__version__, py=py, commands=FIX_COMMANDS)
        print(msg, file=sys.stderr)
        QMessageBox.critical(None, tr("dlg.pyside_title"), msg)
        sys.exit(1)
    window = MainWindow(config_path)
    if args.lang:  # 命令行指定的语言优先于配置文件
        set_language(args.lang)
        window.retranslate()
    window.show()
    if args.robot:
        window.toggle_robot_mode()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
