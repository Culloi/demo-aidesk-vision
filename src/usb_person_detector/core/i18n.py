"""中英文切换（纯 Python，不依赖 Qt；界面和 core.companion 共用）。

用法：
    from usb_person_detector.core.i18n import set_language, tr
    set_language("en")
    tr("btn.start")                     -> "START"
    tr("ev.phone_alert", sec=10)        -> "10 seconds on your phone. Let's focus~"

新增语言：在 LANGUAGES 中登记，并给 STRINGS 每一条加上对应的译文。
缺少译文时回退到中文；键不存在时返回键名本身，方便发现遗漏。
"""

LANGUAGES = {"zh": "中文", "en": "English"}
DEFAULT_LANGUAGE = "zh"
_current = DEFAULT_LANGUAGE

STRINGS = {
    # ---------- 标题栏 ----------
    "app.title": {"zh": "RX-78 桌面陪伴机器人", "en": "RX-78 Desktop Companion"},
    "header.title": {"zh": "桌面陪伴机器人", "en": "COMPANION UNIT"},
    "header.subtitle": {"zh": "RX-78 视觉系统  ·  USB 摄像头  ·  本地离线推理",
                        "en": "MOBILE SUIT VISION SYSTEM  ·  USB CAMERA  ·  LOCAL INFERENCE"},
    "lang.switch": {"zh": "EN", "en": "中文"},          # 按钮显示"要切换到的语言"
    "lang.tooltip": {"zh": "切换到英文 / Switch to English", "en": "切换到中文 / Switch to Chinese"},
    "voice.on": {"zh": "🔊", "en": "🔊"},
    "voice.off": {"zh": "🔇", "en": "🔇"},
    "voice.tooltip_on": {"zh": "语音播报：开（点击关闭）", "en": "Voice: ON (click to mute)"},
    "voice.tooltip_off": {"zh": "语音播报：关（点击开启）", "en": "Voice: OFF (click to unmute)"},

    # ---------- 运行状态徽章 ----------
    "mode.standby": {"zh": "待机", "en": "STANDBY"},
    "mode.booting": {"zh": "启动中", "en": "BOOTING"},
    "mode.running": {"zh": "运行中", "en": "ACTIVE"},
    "mode.stopping": {"zh": "停止中", "en": "STOPPING"},
    "mode.sleep": {"zh": "休眠中", "en": "SLEEP"},
    "mode.error": {"zh": "错误", "en": "ERROR"},

    # ---------- 主视图 ----------
    "panel.main": {"zh": "陪伴  //  主视图", "en": "COMPANION  //  MAIN VIEW"},
    "view.face": {"zh": "表情", "en": "FACE"},
    "view.camera": {"zh": "画面", "en": "CAMERA"},
    "view.fullscreen": {"zh": "全屏模式", "en": "FULLSCREEN"},
    "view.fullscreen_tip": {"zh": "全屏只显示表情脸（F11），Esc 退出",
                            "en": "Show only the face in fullscreen (F11), Esc to exit"},
    "video.no_signal": {"zh": "无信号\n\n— 请按「启动」—", "en": "NO SIGNAL\n\n— PRESS START —"},

    # ---------- 状态卡片 ----------
    "panel.status": {"zh": "陪伴状态", "en": "COMPANION STATUS"},
    "card.presence": {"zh": "在场状态", "en": "PRESENCE"},
    "card.distance": {"zh": "距离", "en": "DISTANCE"},
    "card.phone": {"zh": "手机", "en": "PHONE"},
    "card.pet": {"zh": "宠物", "en": "PET"},
    "card.persons": {"zh": "人数", "en": "PERSON COUNT"},
    "presence.PRESENT": {"zh": "有人", "en": "PRESENT"},
    "presence.ABSENT": {"zh": "无人", "en": "ABSENT"},
    "presence.SLEEP": {"zh": "休眠", "en": "SLEEP"},
    "distance.NEAR": {"zh": "近", "en": "NEAR"},
    "distance.MID": {"zh": "中", "en": "MID"},
    "distance.FAR": {"zh": "远", "en": "FAR"},
    "phone.in_use": {"zh": "使用中 {sec} 秒", "en": "IN USE {sec}s"},

    # ---------- 陪伴规则 / 检测参数 ----------
    "panel.rules": {"zh": "陪伴规则", "en": "COMPANION RULES"},
    "rule.phone": {"zh": "手机提醒（秒）", "en": "PHONE ALERT (S)"},
    "rule.sleep": {"zh": "无人休眠（秒）", "en": "SLEEP AFTER (S)"},
    "panel.detection": {"zh": "检测参数", "en": "DETECTION"},
    "field.model": {"zh": "模型文件", "en": "MODEL FILE"},
    "field.camera": {"zh": "摄像头编号", "en": "CAMERA"},
    "field.camera_auto": {"zh": "自动", "en": "AUTO"},
    "field.width": {"zh": "宽度", "en": "WIDTH"},
    "field.height": {"zh": "高度", "en": "HEIGHT"},
    "field.conf": {"zh": "人员置信度", "en": "PERSON CONF"},
    "field.iou": {"zh": "重叠阈值 IoU", "en": "IOU THRES"},
    "btn.browse": {"zh": "浏览", "en": "BROWSE"},

    # ---------- 日志 / 按钮 ----------
    "panel.log": {"zh": "事件日志", "en": "EVENT LOG"},
    "btn.start": {"zh": "启动", "en": "START"},
    "btn.stop": {"zh": "停止", "en": "STOP"},
    "btn.capture": {"zh": "拍照", "en": "CAPTURE"},
    "btn.voice_cmd": {"zh": "语音指令", "en": "VOICE"},
    "btn.voice_cmd_unavailable": {"zh": "未找到语音识别模型，功能不可用",
                                 "en": "ASR model not found, feature unavailable"},

    # ---------- 底部状态栏 ----------
    "footer.system": {"zh": "E.F.S.F. 视觉系统  ·  {backend}  ·  {model}",
                      "en": "E.F.S.F. VISION OS  ·  {backend}  ·  {model}"},
    "backend.npu": {"zh": "NPU 加速（RKNN）", "en": "NPU (RKNN)"},
    "backend.cpu": {"zh": "CPU 推理（ONNX Runtime）", "en": "CPU (ONNX RUNTIME)"},
    "footer.cam": {"zh": "摄像头 {cam} {res}", "en": "CAM {cam} {res}"},
    "footer.cam_offline": {"zh": "摄像头离线", "en": "CAM OFFLINE"},
    "footer.cam_error": {"zh": "摄像头错误", "en": "CAM ERROR"},
    "footer.low_power": {"zh": "省电模式", "en": "LOW POWER"},
    "footer.runtime": {"zh": "运行时长 {uptime}", "en": "RUN TIME {uptime}"},

    # ---------- 表情脸角落信息 ----------
    "face.offline": {"zh": "离线", "en": "OFFLINE"},
    "face.toggle_fullscreen_tip": {"zh": "双击表情进入 / 退出全屏（也可用 F11，Esc 退出）",
                                   "en": "Double-click the face to enter/exit fullscreen (F11, Esc to exit)"},
    "face.presence_distance": {"zh": "{presence} · {distance}距离", "en": "{presence} · {distance}"},
    "caption.photo": {"zh": "咔嚓！照片已保存", "en": "Click! Photo saved."},
    "caption.voice_listening": {"zh": "我在听...", "en": "Listening..."},

    # ---------- 视频画面检测标签 ----------
    "label.person": {"zh": "人员", "en": "TARGET"},
    "label.phone": {"zh": "手机", "en": "PHONE"},
    "label.cat": {"zh": "猫咪", "en": "CAT"},
    "label.dog": {"zh": "狗狗", "en": "DOG"},
    "label.fire": {"zh": "火焰", "en": "FIRE"},

    # ---------- 陪伴事件（字幕 + 日志） ----------
    "pet.cat": {"zh": "猫咪", "en": "Cat"},
    "pet.dog": {"zh": "狗狗", "en": "Dog"},
    "ev.arrived": {"zh": "欢迎回来！", "en": "Welcome back!"},
    "ev.arrived_wake": {"zh": "你回来啦，我刚刚打了个盹～", "en": "You're back! I just took a little nap~"},
    "ev.left": {"zh": "待会见～", "en": "See you later~"},
    "ev.sleep": {"zh": "没人陪我，先睡一会儿… Zzz", "en": "Nobody around... time for a nap. Zzz"},
    "ev.phone_alert": {"zh": "已经看手机 {sec} 秒了，先专注一下吧～",
                       "en": "{sec} seconds on your phone. Let's focus~"},
    "ev.pet_arrived.cat": {"zh": "发现一只猫咪！你好呀～", "en": "A kitty! Hello there~"},
    "ev.pet_arrived.dog": {"zh": "发现一只狗狗！你好呀～", "en": "A doggy! Hello there~"},
    "ev.pet_left.cat": {"zh": "猫咪走掉了", "en": "The kitty left"},
    "ev.pet_left.dog": {"zh": "狗狗走掉了", "en": "The doggy left"},
    "ev.fire_alert": {"zh": "检测到明火！请注意安全！", "en": "Fire detected! Please stay safe!"},
    "ev.fire_clear": {"zh": "明火已消失", "en": "The fire is gone"},

    # 日志前缀
    "tag.arrived": {"zh": "[到达]", "en": "[ARRIVE]"},
    "tag.left": {"zh": "[离开]", "en": "[LEAVE]"},
    "tag.sleep": {"zh": "[休眠]", "en": "[SLEEP]"},
    "tag.phone_alert": {"zh": "[手机]", "en": "[PHONE]"},
    "tag.pet_arrived": {"zh": "[宠物]", "en": "[PET]"},
    "tag.pet_left": {"zh": "[宠物]", "en": "[PET]"},
    "tag.fire_alert": {"zh": "[火焰]", "en": "[FIRE]"},
    "tag.fire_clear": {"zh": "[火焰]", "en": "[FIRE]"},
    "tag.event": {"zh": "[事件]", "en": "[EVENT]"},

    # ---------- 系统日志 ----------
    "log.ready": {"zh": "系统就绪，按「启动」开始；「全屏模式」或 F11 进入全屏，Esc 退出",
                  "en": "System ready. Press START to begin; FULLSCREEN or F11 for fullscreen, Esc to exit"},
    "log.backend": {"zh": "推理方式：{backend}，模型 {name}", "en": "Inference: {backend}, model {name}"},
    "log.npu_fallback": {"zh": "NPU 不可用，改用 CPU：{reason}", "en": "NPU unavailable, using CPU: {reason}"},
    "log.cam_ok": {"zh": "摄像头 {src} 已连接 {res}", "en": "Camera {src} connected {res}"},
    "log.photo": {"zh": "已拍照 {name}", "en": "Photo saved {name}"},
    "log.stopped": {"zh": "检测已停止", "en": "Detection stopped"},
    "log.error": {"zh": "错误：{msg}", "en": "Error: {msg}"},
    "log.lang": {"zh": "已切换为中文", "en": "Language switched to English"},
    "log.voice_on": {"zh": "语音播报已开启", "en": "Voice announcements enabled"},
    "log.voice_unavailable": {"zh": "未找到语音指令模型，语音指令功能已禁用（见 README 语音交互说明）",
                           "en": "ASR model not found, voice command disabled (see README)"},
    "log.voice_listening": {"zh": "🎤 请说话……", "en": "🎤 Listening, please speak..."},
    "log.voice_heard": {"zh": "听到：{text}", "en": "Heard: {text}"},
    "log.voice_no_speech": {"zh": "没有听清楚，请再说一次", "en": "Didn't catch that, please try again"},
    "log.voice_no_match": {"zh": "没有匹配到已知指令", "en": "No matching command"},
    "log.voice_need_camera": {"zh": "请先启动摄像头再拍照", "en": "Please start the camera before taking a photo"},
    "log.voice_greet": {"zh": "你好呀！我在呢～", "en": "Hi there! I'm here~"},
    "log.voice_error": {"zh": "语音指令出错：{msg}", "en": "Voice command error: {msg}"},
    "log.fire_ready": {"zh": "火焰检测已启用：{name}（每 {interval} 帧检测一次）",
                       "en": "Fire detection enabled: {name} (every {interval} frames)"},
    "log.fire_load_fail": {"zh": "火焰模型加载失败，已跳过：{msg}",
                           "en": "Failed to load fire model, skipped: {msg}"},
    "log.fire_reject": {"zh": "火焰框被颜色校验过滤（置信 {score:.2f}，橙黄占比 {orange}，尺寸 {box}）——若是真实火焰请关闭 fire_color_check",
                        "en": "Fire box rejected by color check (score {score:.2f}, orange ratio {orange}, box {box})"},
    "log.fire_static": {"zh": "火焰框被静止元素过滤（亮度波动 {std}，均值 {bright}）——静止的红框多来自屏幕 UI；若误伤真实火焰请调低 fire_static_min_std",
                        "en": "Fire box rejected as static object (brightness std {std}, mean {bright})"},
    "log.voice_off": {"zh": "语音播报已关闭", "en": "Voice announcements disabled"},

    # ---------- 错误 / 对话框 ----------
    "err.cam_open": {"zh": "无法打开摄像头: {cam}", "en": "Cannot open camera: {cam}"},
    "err.fire_model_missing": {"zh": "火焰模型不存在: {path}", "en": "Fire model not found: {path}"},
    "err.no_usb_cam": {"zh": "未找到 USB 摄像头，请检查是否已插好；也可以在「检测参数」中手动指定摄像头编号",
                       "en": "No USB camera found. Check the connection, or set the camera number in DETECTION"},
    "err.cam_read": {"zh": "读取摄像头帧失败", "en": "Failed to read camera frame"},
    "dlg.model_missing_title": {"zh": "模型不存在", "en": "Model not found"},
    "dlg.model_missing": {"zh": "未找到模型文件:\n{path}", "en": "Model file not found:\n{path}"},
    "dlg.choose_model": {"zh": "选择模型", "en": "Select model"},
    "dlg.onnx_filter": {"zh": "模型 (*.onnx *.rknn)", "en": "Model (*.onnx *.rknn)"},
    "dlg.error_title": {"zh": "运行错误", "en": "Runtime error"},
    "dlg.pyside_title": {"zh": "PySide6 版本不兼容", "en": "Incompatible PySide6 version"},
    "dlg.pyside_msg": {
        "zh": "当前 PySide6 {version} 在 Python {py} 上有已知 bug，界面运行几分钟后会闪退。\n\n请在终端执行：\n{commands}",
        "en": "PySide6 {version} has a known bug on Python {py}: the UI will crash after a few minutes.\n\n"
              "Run in a terminal:\n{commands}",
    },
}


def set_language(lang: str) -> str:
    """切换语言；不支持的语言回退到默认语言。返回实际生效的语言。"""
    global _current
    _current = lang if lang in LANGUAGES else DEFAULT_LANGUAGE
    return _current


def get_language() -> str:
    return _current


def tr(key: str, **kwargs) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(_current) or entry[DEFAULT_LANGUAGE]
    return text.format(**kwargs) if kwargs else text
