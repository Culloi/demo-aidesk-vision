"""桌面陪伴机器人 P1 行为逻辑（纯 Python，不依赖 Qt，便于单元测试与移植）。

输入：每帧检测结果 + 画面尺寸 + 当前时间
输出：CompanionState（给 UI 显示）+ 事件列表（给日志 / 字幕 / 后续语音播报）

功能：
1. 有人/无人状态机（去抖动）+ 无人超时休眠
2. 距离感知（近 / 中 / 远）
3. 目光跟随（主目标头部位置 -> 归一化坐标）
4. 玩手机提醒
5. 宠物识别（猫 / 狗）
6. 表情决策
"""
import time
from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple

from usb_person_detector.core.i18n import tr

PERSON, CAT, DOG, CELL_PHONE = 0, 15, 16, 67
# 火焰用独立模型（单类，模型内部类别是 0）。但 COCO 里 PERSON 也是 0，
# 两个模型的结果合并到同一个列表时会冲突，所以这里用一个 COCO 范围外的哨兵 ID，
# 由 detector 在合并结果时把火焰模型的 class 0 重映射成 FIRE。
FIRE = 1000
PET_KEYS = {CAT: "cat", DOG: "dog"}   # 显示名称见 i18n.py 的 pet.* / ev.pet_*

# 在场状态
SLEEP = "SLEEP"
ABSENT = "ABSENT"
PRESENT = "PRESENT"

# 距离
NEAR, MID, FAR, NONE = "NEAR", "MID", "FAR", "--"

# 表情
EXPR_SLEEP = "sleep"
EXPR_IDLE = "idle"
EXPR_NORMAL = "normal"
EXPR_HAPPY = "happy"
EXPR_CURIOUS = "curious"
EXPR_ALERT = "alert"
EXPR_WORRIED = "worried"   # 目前没有逻辑触发，保留给后续提醒类功能使用
# 临时表情优先级（前高后低）
EXPR_PRIORITY = [EXPR_ALERT, EXPR_WORRIED, EXPR_HAPPY, EXPR_CURIOUS]


@dataclass
class CompanionConfig:
    # 在场状态机
    enter_sec: float = 1.0           # 连续检测到人多久算"有人"
    leave_sec: float = 5.0           # 连续无人多久算"离开"
    sleep_after_sec: float = 60.0    # 离开后多久进入休眠
    # 距离（人框高度 / 画面高度）
    near_ratio: float = 0.75
    far_ratio: float = 0.35
    # 玩手机
    phone_alert_sec: float = 10.0    # 持续看手机多久提醒
    phone_grace_sec: float = 2.0     # 手机短暂没检测到的容忍时间
    phone_cooldown_sec: float = 60.0
    # 手机框几何合理性（挡掉打火机等小物件被误识别为手机）
    phone_min_aspect: float = 0.25   # 宽/高比下限（竖持手机约 0.45~0.7）
    phone_max_aspect: float = 3.0    # 宽/高比上限（横持手机约 1.4~2.9）
    phone_min_height_ratio: float = 0.05   # 框高 / 画面高 的下限
    # 宠物
    pet_confirm_sec: float = 1.0
    pet_leave_sec: float = 5.0
    pet_cooldown_sec: float = 30.0
    # 明火告警（独立模型，检测结果通过 cls_id == FIRE 传入）
    # 小目标火焰的置信度会在阈值附近抖动，单帧“连续过线”可能永远不触发，
    # 所以用滑动窗口内的峰值分数 + 最少命中次数判定，而不是单帧连续。
    fire_confirm_sec: float = 1.5     # 至少观察这么久才可能告警（同时也是最短告警延时）
    fire_clear_sec: float = 3.0       # 火焰消失多久后解除告警状态
    fire_cooldown_sec: float = 60.0   # 两次语音/事件告警之间的最小间隔
    fire_min_score: float = 0.33      # 火焰最低置信度（小目标火焰实测峰值常只有 0.4~0.5）
    fire_window_sec: float = 2.0      # 峰值聚合窗口：统计最近这么久的最高分
    fire_min_hits: int = 3            # 窗口内至少这么多次命中才告警（压一闪而过的误报）
    # 表现
    happy_sec: float = 3.0           # 欢迎表情持续时长
    caption_sec: float = 6.0         # 字幕显示时长
    mirror_gaze: bool = True         # 摄像头画面与屏幕左右相反，目光需要镜像

    @classmethod
    def from_dict(cls, data: dict):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class CompanionEvent:
    kind: str        # arrived / left / sleep / phone_alert / pet_arrived / pet_left
    message: str
    ts: float
    voice_key: str = ""   # 对应 i18n.STRINGS 的 key，供语音播报查表用（见 speech.py）
    voice_param: Optional[int] = None  # 语音分档参数（目前只有 phone_alert 的秒数）


@dataclass
class CompanionState:
    presence: str = ABSENT
    person_count: int = 0
    distance: str = NONE
    distance_ratio: float = 0.0
    phone_in_use: bool = False
    phone_seconds: float = 0.0
    pet: str = ""                                   # "猫咪" / "狗狗" / ""
    fire: bool = False                              # 是否处于明火告警状态
    fire_score: float = 0.0                         # 告警时的火焰置信度
    gaze: Optional[Tuple[float, float]] = None      # (-1~1, -1~1)，None 表示没有目标
    expression: str = EXPR_IDLE
    caption: str = ""
    low_power: bool = False


class Debounce:
    """布尔信号去抖：持续 on_sec 才变 True，持续 off_sec 才变 False。"""

    def __init__(self, on_sec: float, off_sec: float):
        self.on_sec = on_sec
        self.off_sec = off_sec
        self.state = False
        self.score = 0.0        # 最近一次触发时的信号强度（火焰用置信度）
        self._on_since = None
        self._off_since = None
        self.changed_at = None

    def update(self, raw: bool, now: float) -> bool:
        """返回本次是否发生状态切换。"""
        if raw:
            self._off_since = None
            if not self.state:
                if self._on_since is None:
                    self._on_since = now
                if now - self._on_since >= self.on_sec:
                    self.state = True
                    self.changed_at = now
                    return True
        else:
            self._on_since = None
            if self.state:
                if self._off_since is None:
                    self._off_since = now
                if now - self._off_since >= self.off_sec:
                    self.state = False
                    self.changed_at = now
                    return True
        return False


def _boxes_overlap(a, b, expand: float = 1.0) -> bool:
    """两个框是否重叠（expand>1 时把 a 按比例放大后再判，容忍框不完整）。"""
    aw = a.width * (expand - 1.0) / 2.0
    ah = a.height * (expand - 1.0) / 2.0
    return not (a.x2 + aw < b.x1 or a.x1 - aw > b.x2 or
                a.y2 + ah < b.y1 or a.y1 - ah > b.y2)


def _looks_like_phone(det, frame_h: int, cfg) -> bool:
    """几何合理性检查：排除细长条和过小的框（打火机、香烟、钥匙等）。"""
    if det.height <= 0 or frame_h <= 0:
        return False
    aspect = det.width / det.height
    if not (cfg.phone_min_aspect <= aspect <= cfg.phone_max_aspect):
        return False
    # 相对画面的高度比，过小的框不像手机（打火机高度通常不到画面 1/8）
    return det.height / frame_h >= cfg.phone_min_height_ratio


def _point_in_box(px, py, det, margin=0.15):
    mx, my = det.width * margin, det.height * margin
    return det.x1 - mx <= px <= det.x2 + mx and det.y1 - my <= py <= det.y2 + my


class CompanionBrain:
    def __init__(self, config: CompanionConfig = None, clock=time.monotonic):
        self.config = config or CompanionConfig()
        self.clock = clock
        cfg = self.config

        self.presence = Debounce(cfg.enter_sec, cfg.leave_sec)
        self.phone = Debounce(0.5, cfg.phone_grace_sec)
        self.pet = Debounce(cfg.pet_confirm_sec, cfg.pet_leave_sec)
        self.fire = Debounce(cfg.fire_confirm_sec, cfg.fire_clear_sec)

        start = clock()
        self.sleeping = False
        self.last_person_seen = None
        self.absent_since = start
        self.arrived_at = None
        self.phone_start = None
        self.last_phone_alert = None
        self.last_pet_event = None
        self.pet_key = ""
        self.last_fire_alert = None
        self.fire_score = 0.0
        self.fire_hits = []            # 峰值聚合窗口：[(时间, 分数), ...]
        self.fire_alert_since = None   # 首次满足告警条件的时间（用于 fire_confirm_sec 延时）
        self.distance_ratio = 0.0
        self.caption = ""
        self.caption_until = 0.0
        self.expr_override = None      # (表情, 截止时间)

    # ---------- 对外接口 ----------
    def set_config(self, config: CompanionConfig):
        self.config = config

    def update(self, detections, frame_w: int, frame_h: int, now: float = None):
        now = self.clock() if now is None else now
        cfg = self.config
        events: List[CompanionEvent] = []

        # 参数可能被实时修改，同步到去抖器
        self.presence.on_sec, self.presence.off_sec = cfg.enter_sec, cfg.leave_sec
        self.phone.off_sec = cfg.phone_grace_sec
        self.pet.on_sec, self.pet.off_sec = cfg.pet_confirm_sec, cfg.pet_leave_sec

        persons = [d for d in detections if d.cls_id == PERSON]
        phones = [d for d in detections if d.cls_id == CELL_PHONE]
        pets = [d for d in detections if d.cls_id in PET_KEYS]
        fires = [d for d in detections if d.cls_id == FIRE]
        main = max(persons, key=lambda d: d.area) if persons else None

        if persons:
            self.last_person_seen = now

        self._update_presence(bool(persons), now, events)
        self._update_distance(main, frame_h)
        self._update_phone(persons, phones, fires, frame_h, now, events)
        self._update_pet(pets, now, events)
        self._update_fire(fires, now, events)

        for ev in events:
            self._show_caption(ev.message, now)

        gaze_target = main or (max(pets, key=lambda d: d.area) if pets else None)
        state = CompanionState(
            presence=SLEEP if self.sleeping else (PRESENT if self.presence.state else ABSENT),
            person_count=len(persons),
            distance=self._distance_label() if main else NONE,
            distance_ratio=self.distance_ratio if main else 0.0,
            phone_in_use=self.phone.state,
            phone_seconds=(now - self.phone_start) if self.phone_start is not None else 0.0,
            pet=tr(f"pet.{self.pet_key}") if self.pet.state else "",
            fire=self.fire.state,
            fire_score=self.fire.score if self.fire.state else 0.0,
            gaze=self._gaze(gaze_target, frame_w, frame_h),
            caption=self.caption if now < self.caption_until else "",
            low_power=self.sleeping,
        )
        state.expression = self._expression(state, now)
        return state, events

    # ---------- 各子模块 ----------
    def _update_presence(self, raw: bool, now: float, events):
        cfg = self.config
        if self.presence.update(raw, now):
            if self.presence.state:
                was_sleeping = self.sleeping
                self.sleeping = False
                self.arrived_at = now
                self.absent_since = None
                key = "ev.arrived_wake" if was_sleeping else "ev.arrived"
                events.append(CompanionEvent("arrived", tr(key), now, voice_key=key))
                self._set_expr(EXPR_HAPPY, now + cfg.happy_sec, now)
            else:
                self.absent_since = now
                events.append(CompanionEvent("left", tr("ev.left"), now, voice_key="ev.left"))

        if not self.presence.state and not self.sleeping and self.absent_since is not None:
            if now - self.absent_since >= cfg.sleep_after_sec:
                self.sleeping = True
                events.append(CompanionEvent("sleep", tr("ev.sleep"), now, voice_key="ev.sleep"))

    def _update_distance(self, main, frame_h: int):
        if main is None or frame_h <= 0:
            return
        ratio = main.height / frame_h
        # 指数平滑，避免框抖动导致距离状态来回跳
        self.distance_ratio = ratio if self.distance_ratio == 0 else self.distance_ratio * 0.7 + ratio * 0.3

    def _distance_label(self):
        if self.distance_ratio >= self.config.near_ratio:
            return NEAR
        if self.distance_ratio <= self.config.far_ratio:
            return FAR
        return MID

    def _update_phone(self, persons, phones, fires, frame_h, now: float, events):
        cfg = self.config
        # 打火机被误识别为手机的抑制：两个条件任一满足就排除
        # 1) 与火焰框重叠——点燃的打火机不是手机（跨模型互斥）
        # 2) 几何不像手机——细长条/过小的框（打火机侧持时宽高比很小）
        phones = [p for p in phones
                  if not any(_boxes_overlap(p, f, 1.4) for f in fires)
                  and _looks_like_phone(p, frame_h, cfg)]

        # 只统计"在人身上"的手机，桌上放着的手机不算
        in_use = any(_point_in_box(*p.center, person) for p in phones for person in persons)
        if self.phone.update(in_use, now) and self.phone.state:
            self.phone_start = now
        if not self.phone.state:
            self.phone_start = None
            return

        used = now - self.phone_start
        cooled = self.last_phone_alert is None or now - self.last_phone_alert >= cfg.phone_cooldown_sec
        if used >= cfg.phone_alert_sec and cooled:
            self.last_phone_alert = now
            events.append(CompanionEvent("phone_alert", tr("ev.phone_alert", sec=int(used)), now,
                                          voice_key="ev.phone_alert", voice_param=int(cfg.phone_alert_sec)))
            self._set_expr(EXPR_ALERT, now + cfg.caption_sec, now)

    def _update_pet(self, pets, now: float, events):
        cfg = self.config
        if pets:
            self.pet_key = PET_KEYS[max(pets, key=lambda d: d.score).cls_id]
        if self.pet.update(bool(pets), now):
            if self.pet.state:
                if self.last_pet_event is None or now - self.last_pet_event >= cfg.pet_cooldown_sec:
                    self.last_pet_event = now
                    key = f"ev.pet_arrived.{self.pet_key}"
                    events.append(CompanionEvent("pet_arrived", tr(key), now, voice_key=key))
                    self._set_expr(EXPR_CURIOUS, now + cfg.happy_sec, now)
            else:
                key = f"ev.pet_left.{self.pet_key}"
                events.append(CompanionEvent("pet_left", tr(key), now, voice_key=key))

    def _update_fire(self, fires, now: float, events):
        """明火告警（峰值聚合 + 命中次数，替代原来的单帧连续去抖）。

        为什么要改：小目标火焰（桌面打火机）的置信度往往在 0.4~0.5 之间振荡，
        而原实现要求“连续 fire_confirm_sec 都过线”，每次跌落都会重置计时器，
        结果可能永远不告警。改成滑动窗口峰值聚合后，只要窗口内命中足够多次
        且峰值过线就告警，对抖动免疫。

        四层防护（前两层针对灵敏度，后两层针对误报）：
        1) fire_min_score：单帧过线门槛（默认 0.33，适配小火焰偏低的分数）
        2) 峰值聚合：统计最近 fire_window_sec 内的最高分，抵抗单帧抖动
        3) fire_min_hits：窗口内至少命中这么多次，压掉一闪而过的误报
        4) fire_cooldown_sec：告警后一段时间不重复播报（状态仍保持）
        """
        cfg = self.config
        window = max(0.1, cfg.fire_window_sec)

        # 1) 收集本帧过线分数，并清理窗口外记录
        frame_peak = max((d.score for d in fires if d.score >= cfg.fire_min_score), default=0.0)
        self.fire_hits = [(t, s) for t, s in self.fire_hits if now - t <= window]
        if frame_peak > 0:
            self.fire_hits.append((now, frame_peak))

        # 2) 窗口峰值 + 命中次数
        peak = max((s for _, s in self.fire_hits), default=0.0)
        hit_count = len(self.fire_hits)
        detected = peak >= cfg.fire_min_score and hit_count >= max(1, cfg.fire_min_hits)

        if detected:
            if self.fire_alert_since is None:
                self.fire_alert_since = now
            self.fire.score = peak
        else:
            self.fire_alert_since = None

        # 3) 告警延时：满足条件并持续 fire_confirm_sec 后才置为告警状态
        ready = (detected and self.fire_alert_since is not None
                 and now - self.fire_alert_since >= cfg.fire_confirm_sec)

        was_on = self.fire.state
        if ready and not was_on:
            self.fire.state = True
            self.fire.changed_at = now
            cooled = (self.last_fire_alert is None
                      or now - self.last_fire_alert >= cfg.fire_cooldown_sec)
            if cooled:
                self.last_fire_alert = now
                events.append(CompanionEvent("fire_alert", tr("ev.fire_alert"), now,
                                             voice_key="ev.fire_alert"))
            self._set_expr(EXPR_ALERT, now + cfg.caption_sec, now)
            return

        # 4) 解除告警：窗口内长时间无命中，且超过 fire_clear_sec
        if was_on:
            last_hit = self.fire_hits[-1][0] if self.fire_hits else None
            silent_for = now - last_hit if last_hit is not None else float("inf")
            if not detected and silent_for >= cfg.fire_clear_sec:
                self.fire.state = False
                self.fire.changed_at = now
                events.append(CompanionEvent("fire_clear", tr("ev.fire_clear"), now,
                                             voice_key="ev.fire_clear"))

    def _gaze(self, target, frame_w: int, frame_h: int):
        if target is None or frame_w <= 0 or frame_h <= 0:
            return None
        cx = (target.x1 + target.x2) / 2
        # 人看头部（框上部 20%），宠物看中心
        cy = target.y1 + target.height * (0.2 if target.cls_id == PERSON else 0.5)
        gx = cx / frame_w * 2 - 1
        gy = cy / frame_h * 2 - 1
        if self.config.mirror_gaze:
            gx = -gx
        return max(-1.0, min(1.0, gx)), max(-1.0, min(1.0, gy))

    # ---------- 表情 / 字幕 ----------
    def _set_expr(self, expr: str, until: float, now: float):
        """设置临时表情；仍在生效中的更高优先级表情不会被覆盖。"""
        if self.expr_override is not None:
            cur, cur_until = self.expr_override
            if now < cur_until and EXPR_PRIORITY.index(cur) < EXPR_PRIORITY.index(expr):
                return
        self.expr_override = (expr, until)

    def _show_caption(self, text: str, now: float):
        self.caption = text
        self.caption_until = now + self.config.caption_sec

    def _expression(self, state: CompanionState, now: float) -> str:
        # 明火告警优先级最高：即使人在休眠/离开也要显示告警
        if state.fire:
            return EXPR_ALERT
        if state.presence == SLEEP:
            return EXPR_SLEEP
        if self.expr_override is not None:
            expr, until = self.expr_override
            if now < until:
                return expr
            self.expr_override = None
        if state.phone_in_use and state.phone_seconds >= self.config.phone_alert_sec:
            return EXPR_ALERT
        if state.pet:
            return EXPR_CURIOUS
        if state.presence == PRESENT:
            return EXPR_CURIOUS if state.distance == NEAR else EXPR_NORMAL
        return EXPR_IDLE
