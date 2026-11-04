"""固定播报语音（纯 Python，不依赖 Qt）。

素材由 tools/gen_voice.py 离线预合成，存放在 assets/voice/<lang>/<key>.wav。
运行时只做"查表 -> 排队 -> 子进程播放"，不做任何实时语音合成，
不占用留给 YOLO 推理的 CPU / NPU 资源。

用法：
    player = SpeechPlayer(enabled=True)
    player.speak_event(event, lang="zh")   # event: companion.CompanionEvent
    player.shutdown()
"""
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

PHONE_ALERT_SEC_BUCKETS = [5, 10, 15, 20, 30, 45, 60]

DEFAULT_VOICE_DIR = Path(__file__).resolve().parents[3] / "assets" / "voice"


def _nearest_bucket(sec: int, buckets=PHONE_ALERT_SEC_BUCKETS) -> int:
    return min(buckets, key=lambda b: abs(b - sec))


def resolve_voice_path(voice_dir: Path, lang: str, voice_key: str,
                        voice_param: Optional[int] = None) -> Optional[Path]:
    """把事件的 voice_key(+voice_param) 映射到具体 wav 文件路径；找不到返回 None。"""
    if not voice_key:
        return None
    key = voice_key
    if voice_param is not None:
        bucket = _nearest_bucket(int(voice_param))
        key = f"{voice_key}.{bucket}"
    path = voice_dir / lang / f"{key}.wav"
    if path.exists():
        return path
    # 回退到中文素材（例如英文语音还没生成时）
    fallback = voice_dir / "zh" / f"{key}.wav"
    return fallback if fallback.exists() else None


def _find_player_cmd() -> Optional[list]:
    """挑一个本机可用的播放命令，优先 paplay（PipeWire/PulseAudio），其次 aplay。"""
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    return None


class SpeechPlayer:
    """事件驱动的语音播放：单独线程串行播放，避免阻塞摄像头/检测线程。"""

    def __init__(self, voice_dir: Path = DEFAULT_VOICE_DIR, enabled: bool = True,
                 player_cmd: Optional[list] = None):
        self.voice_dir = Path(voice_dir)
        self.enabled = enabled
        self._player_cmd = player_cmd if player_cmd is not None else _find_player_cmd()
        self._queue: "queue.Queue[Optional[Path]]" = queue.Queue()
        self._proc = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stopped = False
        self._thread.start()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self._drain_queue()

    def speak_event(self, event, lang: str):
        """event: companion.CompanionEvent（鸭子类型，只用到 voice_key / voice_param）"""
        if not self.enabled or self._player_cmd is None:
            return
        path = resolve_voice_path(self.voice_dir, lang,
                                   getattr(event, "voice_key", ""),
                                   getattr(event, "voice_param", None))
        if path is None:
            return
        self._queue.put(path)

    def _drain_queue(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _run(self):
        while not self._stopped:
            path = self._queue.get()
            if path is None:
                break
            if not self.enabled:
                continue
            try:
                with self._lock:
                    self._proc = subprocess.Popen(
                        self._player_cmd + [str(path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                self._proc.wait()
            except OSError:
                pass
            finally:
                with self._lock:
                    self._proc = None

    def shutdown(self):
        self._stopped = True
        with self._lock:
            if self._proc is not None:
                self._proc.terminate()
        self._queue.put(None)
        self._thread.join(timeout=2)
