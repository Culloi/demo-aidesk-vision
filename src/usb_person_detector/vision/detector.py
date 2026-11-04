import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# COCO 类别中与桌面陪伴相关的类别
PERSON, CAT, DOG, CELL_PHONE = 0, 15, 16, 67
DEFAULT_CLASSES = {PERSON: "person", CAT: "cat", DOG: "dog", CELL_PHONE: "cell phone"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_id: int = PERSON
    label: str = "person"

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class YoloDetector:
    """YOLOv8/YOLO11 ONNX 检测器（输出 [1, 4+num_classes, N]，无内置 NMS）。

    classes:    {类别 id: 名称}，只保留这些类别
    class_conf: {类别名称: 阈值}，可为小目标（如手机）单独设置更低阈值
    """

    def __init__(self, model_path: str, conf_thres: float = 0.4, iou_thres: float = 0.5,
                 classes: dict = None, class_conf: dict = None, num_threads: int = 0, npu_core: str = "auto"):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = dict(classes or {PERSON: "person"})
        class_conf = class_conf or {}

        self.class_ids = np.array(sorted(self.classes), dtype=np.int64)
        self.class_thres = np.array(
            [float(class_conf.get(self.classes[c], conf_thres)) for c in self.class_ids], dtype=np.float32
        )

        # .rknn 模型在 NPU 上推理（开发板），其他用 ONNX Runtime 在 CPU 上推理
        self.rknn = None
        if str(model_path).lower().endswith(".rknn"):
            from usb_person_detector.vision.rknn_backend import RknnModel
            self.rknn = RknnModel(str(model_path), core_mask=npu_core)
            self.backend = "npu"
            self.input_height = self.rknn.input_height
            self.input_width = self.rknn.input_width
            return

        # num_threads: 推理线程数，0 = ONNX Runtime 默认（用满所有核）。
        # RK3576 上用满 8 核时 CPU 占用约 720%，界面和系统会卡；4 线程只慢约 17%，但能空出一半的核。
        self.backend = "cpu"
        options = ort.SessionOptions()
        if num_threads and num_threads > 0:
            options.intra_op_num_threads = int(num_threads)
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        shape = self.input_meta.shape
        self.input_height = self._normalize_dim(shape[2], 640)
        self.input_width = self._normalize_dim(shape[3], 640)

    @staticmethod
    def _normalize_dim(value, default):
        return value if isinstance(value, int) else default

    def letterbox(self, image, color=(114, 114, 114)):
        h, w = image.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), color, dtype=np.uint8)
        left = int(round((self.input_width - nw) / 2 - 0.1))
        top = int(round((self.input_height - nh) / 2 - 0.1))
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def preprocess(self, image):
        padded, scale, pad_x, pad_y = self.letterbox(image)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        return blob, scale, pad_x, pad_y

    def infer(self, image):
        if self.rknn is not None:
            # NPU：直接送 uint8 RGB，/255 归一化已写进模型（见 tools/convert_rknn.py）
            padded, scale, pad_x, pad_y = self.letterbox(image)
            outputs = self.rknn.run(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            return self.postprocess(outputs[0], image.shape[:2], scale, pad_x, pad_y)
        blob, scale, pad_x, pad_y = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs[0], image.shape[:2], scale, pad_x, pad_y)

    def postprocess(self, pred, orig_shape, scale, pad_x, pad_y):
        orig_h, orig_w = orig_shape
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"Unsupported output shape: {pred.shape}")
        if pred.shape[0] < pred.shape[1]:  # [4+C, N] -> [N, 4+C]
            pred = pred.T
        if pred.shape[1] < 5:
            raise RuntimeError(f"Unexpected output shape after transpose: {pred.shape}")

        # 只取关心的类别列，向量化计算
        valid_ids = self.class_ids[self.class_ids < pred.shape[1] - 4]
        if valid_ids.size == 0:
            return []
        scores_all = pred[:, 4 + valid_ids]                   # [N, K]
        best = scores_all.argmax(axis=1)                       # 每个候选框的最佳类别
        best_score = scores_all[np.arange(len(pred)), best]
        thres = self.class_thres[: valid_ids.size][best]
        keep = best_score >= thres
        if not np.any(keep):
            return []

        sel = pred[keep]
        sel_scores = best_score[keep]
        sel_cls = valid_ids[best[keep]]
        cx, cy, w, h = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, orig_w - 1)
        y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, orig_h - 1)
        x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, orig_w - 1)
        y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, orig_h - 1)

        ok = (x2 > x1) & (y2 > y1)
        if not np.any(ok):
            return []
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)[ok]
        scores = sel_scores[ok].astype(float)
        cls = sel_cls[ok]

        # 按类别分组做 NMS，避免手机框被人框抑制
        indices = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), cls.astype(int).tolist(),
                                          float(min(self.class_thres)), self.iou_thres)
        detections = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = boxes[idx]
            c = int(cls[idx])
            detections.append(Detection(float(bx), float(by), float(bx + bw), float(by + bh),
                                        float(scores[idx]), c, self.classes.get(c, str(c))))
        return detections


def create_detector(model_path, backend: str = "auto", rknn_model: str = None, npu_core: str = "auto", **kwargs):
    """按配置创建检测器，返回 (detector, fallback_reason)。

    backend:
      "auto"（默认）有 NPU 运行库且 rknn_model 存在时用 NPU，否则用 CPU；NPU 加载失败自动退回 CPU
      "npu"  只用 NPU，失败直接报错
      "cpu"  只用 CPU（model_path 指向的 ONNX 模型）
    fallback_reason: 本应该用 NPU 却退回了 CPU 时的原因；没有退回（包括主机上本来就没有 NPU）时为 None。
    """
    from usb_person_detector.vision.rknn_backend import rknn_available

    backend = (backend or "auto").lower()
    if str(model_path).lower().endswith(".rknn"):          # 直接指定了 .rknn 模型
        return YoloDetector(str(model_path), npu_core=npu_core, **kwargs), None

    reason = None
    if backend in ("auto", "npu"):
        if not rknn_available():
            reason = "no librknnrt.so" if backend == "npu" else None   # 主机上没有 NPU 是正常情况
        elif not rknn_model or not Path(rknn_model).exists():
            reason = f"model not found: {rknn_model}"
        else:
            try:
                return YoloDetector(str(rknn_model), npu_core=npu_core, **kwargs), None
            except Exception as e:  # 模型和运行库版本不匹配、NPU 驱动异常等
                reason = str(e)
        if backend == "npu":
            raise RuntimeError(f"NPU: {reason}")
    return YoloDetector(str(model_path), **kwargs), reason


def load_config(config_path: Path):
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CameraNotFoundError(RuntimeError):
    """camera="auto" 时没有找到 USB 摄像头。"""


def fire_roi(frame, mode="auto", ratio: float = 0.75, min_side: int = 640):
    """为火焰检测计算中心裁剪区域（数字变焦，提升小火焰的有效像素）。

    背景：摄像头 1280x720 送进 640x640 时横向被缩小 0.5 倍，桌面上的打火机火苗
    可能只剩十几像素。中心裁剪后再送，不增加任何 NPU 计算量，却能让火苗像素翻倍。

    参数：
        frame:    原始帧（BGR）
        mode:     "off" 不裁剪；"center" 固定裁 min_side；"auto" 自适应
        ratio:    auto 模式下裁剪区域至少占画面短边的比例（保视野）
        min_side: 目标裁剪边长（通常等于模型输入边长）

    返回 (roi, (offset_x, offset_y))：roi 是原帧的视图，offset 用于把检测框坐标还原回整帧。
    不修改传入的 frame。
    """
    h, w = frame.shape[:2]
    short = min(h, w)

    if mode == "off" or short <= min_side:
        # 短边本来就不大于模型输入边长：裁剪不会有任何放大收益
        return frame, (0, 0)

    if mode == "center":
        side = min(min_side, short)
    else:  # auto
        # 至少保留 ratio 比例的画面，避免视野过窄漏检画面边缘的火焰
        side = max(min_side, int(round(short * max(0.1, min(1.0, ratio)))))
        side = min(side, short)

    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return frame[y1:y1 + side, x1:x1 + side], (x1, y1)


def fire_color_check(frame, det, min_orange_ratio: float = 0.05,
                     min_saturation: int = 80, min_value: int = 80):
    """火焰框内的颜色校验：过滤掉纯红色 UI 元素（屏幕对话框边框、图标、指示灯等）。

    原理（实测数据）：真实火焰含有橙黄色成分（HSV 色相 H 10~35），
    而纯红色 UI 元素的色相都集中在 H 0~10，橙黄像素比例几乎为 0：

        真实火焰   orange_ratio = 0.20 ~ 0.91
        纯红/渐变/白字/边框/对话框   = 0.000

    注意不能用亮度方差（v_std）做判据：红块加白字的 v_std 比火焰还高。

    参数：
        frame: 整帧 BGR（坐标需与 det 一致）
        det:   火焰检测框
        min_orange_ratio: 橙黄像素比例下限，低于此值判为“不像火焰”
        min_saturation / min_value: 只统计“够饱和且够亮”的像素，避免灰色背景干扰

    返回 (是否像火焰, 特征字典)。无法判定时（框为空）返回 True，宁可放过不漏报。
    """
    h_img, w_img = frame.shape[:2]
    x1 = max(0, int(det.x1)); y1 = max(0, int(det.y1))
    x2 = min(w_img, int(det.x2)); y2 = min(h_img, int(det.y2))
    if x2 <= x1 or y2 <= y1:
        return True, {}

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return True, {}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    # 橙黄（H 10~35）= 火焰的典型色调；纯红 UI 落在 H 0~10，不计入
    orange = (hue >= 10) & (hue <= 35) & (sat >= min_saturation) & (val >= min_value)
    orange_ratio = float(orange.mean())
    info = {
        "orange_ratio": round(orange_ratio, 3),
        "v_std": round(float(val.std()), 1),
        "v_mean": round(float(val.mean()), 0),
        "box": f"{x2 - x1}x{y2 - y1}",
    }
    return orange_ratio >= min_orange_ratio, info


def find_usb_cameras(sysfs_root: str = "/sys/class/video4linux"):
    """返回 USB（UVC）摄像头的视频采集节点，如 ['/dev/video27']，按编号排序。

    RK3576 等开发板上 /dev/video0~N 常被 MIPI 摄像头接口（rkcif/rkisp）占用，USB 摄像头编号不固定，
    所以按驱动（uvcvideo）查找。每个 UVC 摄像头通常注册两个节点（视频 + 元数据），
    取 index 为 0 的视频采集节点。
    """
    root = Path(sysfs_root)
    found = []
    for node in root.glob("video*"):
        if not node.name[5:].isdigit():
            continue
        try:
            driver = os.path.basename(os.readlink(node / "device" / "driver"))
        except OSError:
            continue
        if driver != "uvcvideo":
            continue
        index_file = node / "index"
        if index_file.exists() and index_file.read_text().strip() != "0":
            continue
        found.append(node.name)
    return [f"/dev/{n}" for n in sorted(found, key=lambda n: int(n[5:]))]


def open_camera(camera_value, width, height, fourcc="MJPG"):
    """打开摄像头。

    camera_value: "auto"（自动查找第一个 USB 摄像头）/ 编号（如 0）/ 设备路径（如 /dev/video27）
    fourcc: 像素格式。USB 2.0 摄像头用默认的 YUYV 时，720p 通常只有 10 FPS；
            MJPG 是压缩格式，可达 30 FPS。传 None 或空字符串则保持摄像头默认格式。
    """
    if camera_value is None or str(camera_value).strip().lower() in ("auto", "-1"):
        cameras = find_usb_cameras()
        if not cameras:
            raise CameraNotFoundError("no USB camera found")
        source = cameras[0]
    else:
        try:
            source = int(camera_value)
        except (TypeError, ValueError):
            source = camera_value

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if fourcc:  # 需要在设置分辨率之前设置
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 减小缓冲，避免低帧率待机时读到旧画面
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, source


class LatestFrameReader:
    """后台线程持续取帧，只保留最新一帧。

    同步循环「读一帧 → 推理 → 再读一帧」会读到在驱动缓冲区里放了一个推理周期的旧帧。
    开发板上 CPU 推理约 400 ms，实测画面要晚约 0.4 秒；休眠降帧时会晚得更多。
    用了本类后，处理的画面最多晚 1 帧。

    （OpenCV 的摄像头对象在两个线程里同时调用 grab/retrieve 不安全，所以由后台线程独占并解码。）
    """

    def __init__(self, cap):
        self.cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0          # 已取到的帧序号
        self._taken = 0        # 已交给调用方的帧序号
        self._failed = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok or frame is None:
                    self._failed = True
                else:
                    self._frame = frame
                    self._seq += 1
                self._cond.notify_all()
            if not ok or frame is None:
                break

    def read(self, timeout: float = 3.0):
        """返回 (ok, frame)：最新的、还没返回过的一帧；没有新帧时等待，超时或摄像头出错返回 (False, None)。"""
        with self._cond:
            has_new = self._cond.wait_for(lambda: self._seq != self._taken or self._failed, timeout)
            if not has_new or self._seq == self._taken:
                return False, None
            self._taken = self._seq
            return True, self._frame

    def stop(self, timeout: float = 2.0):
        self._running = False
        self._thread.join(timeout)


def save_frame(frame, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = save_dir / f"capture_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    return path


# ============================================================
# 视频叠加层（红白蓝高达配色，BGR）
# ============================================================
_RED = (47, 38, 224)
_WHITE = (250, 246, 244)
_BLUE = (200, 86, 30)
_YELLOW = (0, 196, 245)
_NAVY = (69, 31, 17)

# 不同类别的框颜色 / 标签前缀
_CLASS_STYLE = {
    PERSON: (_RED, "TARGET"),
    CELL_PHONE: (_YELLOW, "PHONE"),
    CAT: (_BLUE, "CAT"),
    DOG: (_BLUE, "DOG"),
}


def _corner_brackets(frame, x1, y1, x2, y2, color, length=18, thickness=3):
    """在检测框四角绘制瞄准角标。"""
    length = max(6, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def draw_hud_overlay(frame, detections, fps, camera_label, presence_text=None, show_panel=True, draw_labels=True):
    """绘制目标锁定框。

    show_panel=True：左上角绘制信息面板（CLI 版使用）
    draw_labels=False：只画框不画文字标签。OpenCV 自带字体不支持中文，GUI 版改用 Qt 绘制中文标签
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    counters = {}

    for det in detections:
        color, prefix = _CLASS_STYLE.get(det.cls_id, (_WHITE, det.label.upper()))
        counters[prefix] = counters.get(prefix, 0) + 1
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _corner_brackets(frame, x1, y1, x2, y2, _YELLOW if det.cls_id == PERSON else _WHITE)
        if not draw_labels:
            continue

        label = f"{prefix}-{counters[prefix]:02d}  {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        tag_w, tag_h = tw + 18, th + 12
        ty1 = y1 - tag_h
        if ty1 < 0:  # 框贴近顶部时把标签放到框内
            ty1 = y1
        tx1 = min(x1, max(0, w - tag_w))  # 框贴近右边缘时把标签左移，避免被截断
        text_color = _NAVY if color == _YELLOW else _WHITE
        cv2.rectangle(frame, (tx1, ty1), (tx1 + tag_w, ty1 + tag_h), color, -1)
        cv2.rectangle(frame, (tx1, ty1), (tx1 + 5, ty1 + tag_h), _WHITE if color == _YELLOW else _YELLOW, -1)
        cv2.putText(frame, label, (tx1 + 11, ty1 + th + 5), font, 0.55, text_color, 2, cv2.LINE_AA)

    if not show_panel:
        return frame

    persons = [d for d in detections if d.cls_id == PERSON]
    panel_w, panel_h = 330, 128
    x0, y0 = 16, 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _NAVY, -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 26), _WHITE, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + 6, y0 + 26), _RED, -1)
    cv2.putText(frame, "RX-78 PERSON SENSOR", (x0 + 14, y0 + 19), font, 0.55, _NAVY, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), _BLUE, 1)

    presence = presence_text or ("PRESENT" if persons else "IDLE")
    lines = [
        (f"CAMERA    {camera_label}", _WHITE),
        (f"PERSON    {len(persons)}", _RED if persons else _WHITE),
        (f"FPS       {fps:.1f}", _WHITE),
        (f"PRESENCE  {presence}", _RED if presence == "PRESENT" else _YELLOW),
    ]
    y = y0 + 48
    for text, color in lines:
        cv2.putText(frame, text, (x0 + 14, y), font, 0.52, color, 1, cv2.LINE_AA)
        y += 22
    return frame
