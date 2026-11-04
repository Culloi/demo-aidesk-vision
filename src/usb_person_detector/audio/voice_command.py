"""语音指令识别（纯 Python，不依赖 Qt）。

该模块位于 audio 包中，可独立测试和替换。

架构（和 speech.py 对称，speech.py 是"播"，这里是"听"）：
    麦克风 -> AudioRecorder（固定时长录音，arecord 子进程）
           -> AsrEngine（语音转文字，SenseVoice via sherpa-onnx）
           -> IntentMatcher（关键词匹配，不做开放式 NLU）
           -> VoiceCommand（intent + 原始文本，交给 gui_app.py 执行动作）

第一版只做"点击说话"（push-to-talk），不做热词唤醒；指令集固定、关键词匹配，
不依赖联网、不占用 NPU（ASR 走 CPU 推理）。

用法：
    engine = SherpaSenseVoiceAsr(model_dir="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    recorder = AudioRecorder()
    matcher = IntentMatcher()

    wav_path = recorder.record_once(duration_sec=4)
    text = engine.transcribe(wav_path)
    cmd = matcher.match(text)   # VoiceCommand(intent="capture", text=text) 或 None
"""
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 16000


@dataclass
class VoiceCommand:
    intent: str
    text: str


# ---------- 录音 ----------
class AudioRecorder:
    """用 arecord 录固定时长的音频到临时 wav 文件（16kHz / 单声道 / 16bit，SenseVoice 要求的格式）。"""

    def __init__(self, device: Optional[str] = None, sample_rate: int = SAMPLE_RATE):
        self.device = device
        self.sample_rate = sample_rate

    def record_once(self, duration_sec: float = 4.0) -> Path:
        out_path = Path(tempfile.gettempdir()) / f"voice_cmd_{int(time.time() * 1000)}.wav"
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1",
               "-d", str(int(round(duration_sec)))]
        if self.device:
            cmd += ["-D", self.device]
        cmd += [str(out_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path


# ---------- 语音识别 ----------
class AsrEngine:
    """接口：子类实现 transcribe。方便测试时用假引擎替换，不依赖真实模型文件。"""

    def transcribe(self, wav_path: Path) -> str:
        raise NotImplementedError


class SherpaSenseVoiceAsr(AsrEngine):
    """用 sherpa-onnx 加载 SenseVoice-small（int8 量化版，CPU 推理）。

    模型文件（不入库，需单独下载/从板子对应目录获取）：
        models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
            model.int8.onnx
            tokens.txt
    """

    def __init__(self, model_dir, num_threads: int = 2, use_itn: bool = True):
        import sherpa_onnx  # 延迟导入：没装这个包时，不用到语音指令功能的代码路径不受影响

        model_dir = Path(model_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            use_itn=use_itn,
        )

    def transcribe(self, wav_path: Path) -> str:
        import numpy as np
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sr, pcm)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


# ---------- 意图匹配 ----------
# 每个 intent 对应一组中英文关键词（子串匹配，不区分大小写）。
# 指令数量少，用关键词表足够准确，不需要上 NLU 模型。
INTENT_KEYWORDS = {
    "capture":        ["拍照", "拍张照", "照张相", "take a photo", "take a picture"],
    "lang_switch":     ["切换英文", "切换中文", "切换到英文", "切换到中文", "切英文", "切中文",
                        "switch to english", "switch to chinese", "change language"],
    "mute":            ["安静点", "别说话", "闭嘴", "别说了", "mute", "be quiet", "stop talking"],
    "unmute":          ["说话吧", "可以说话", "unmute", "you can talk", "start talking"],
    "fullscreen_on":   ["进入全屏", "全屏模式", "enter fullscreen", "go fullscreen", "full screen"],
    "fullscreen_off":  ["退出全屏", "关闭全屏", "exit fullscreen", "leave fullscreen"],
    "greet":           ["你好", "在吗", "hello there", "are you there"],
}

_PUNCT_RE = re.compile(r"[，。！？,.!?~\s]")


class IntentMatcher:
    def __init__(self, keywords=None):
        self.keywords = keywords or INTENT_KEYWORDS

    def match(self, text: str) -> Optional[VoiceCommand]:
        if not text:
            return None
        norm = _PUNCT_RE.sub("", text.lower())
        for intent, phrases in self.keywords.items():
            for phrase in phrases:
                if _PUNCT_RE.sub("", phrase.lower()) in norm:
                    return VoiceCommand(intent=intent, text=text)
        return None


# ---------- 端到端封装 ----------
class VoiceCommandPipeline:
    """串起 录音 -> 识别 -> 匹配，供 gui_app.py 的后台线程调用一次性执行。"""

    def __init__(self, asr: AsrEngine, recorder: Optional[AudioRecorder] = None,
                 matcher: Optional[IntentMatcher] = None):
        self.asr = asr
        self.recorder = recorder or AudioRecorder()
        self.matcher = matcher or IntentMatcher()

    def listen_once(self, duration_sec: float = 4.0) -> Optional[VoiceCommand]:
        wav_path = self.recorder.record_once(duration_sec)
        try:
            text = self.asr.transcribe(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)
        return self.matcher.match(text)
