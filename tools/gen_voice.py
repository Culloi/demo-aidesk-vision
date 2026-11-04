#!/usr/bin/env python3
"""离线预合成固定播报语音素材（主机联网时跑一次即可，产物入库）。

依赖（仅主机 / CI，板子运行时不需要）：
    pip install edge-tts
    系统需要 gst-launch-1.0（GStreamer，用来把 mp3 转成 wav；GNOME 桌面一般自带）

用法：
    python3 tools/gen_voice.py                 # 生成全部缺失的音频
    python3 tools/gen_voice.py --force         # 全部重新生成

产物：assets/voice/<lang>/<key>.wav（<key> 对应 i18n.STRINGS 的键，
phone_alert 这类带参数的事件按 PHONE_ALERT_SEC_BUCKETS 生成若干档）。
"""
import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from usb_person_detector.core.i18n import STRINGS  # noqa: E402

VOICE_DIR = ROOT / "assets" / "voice"

# 每种语言用哪个 edge-tts 音色朗读
TTS_VOICE = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AnaNeural",
}

# 固定播报的事件（key -> 是否带 {sec} 参数）
EVENT_KEYS = [
    "ev.arrived",
    "ev.arrived_wake",
    "ev.left",
    "ev.sleep",
    "ev.pet_arrived.cat",
    "ev.pet_arrived.dog",
    "ev.pet_left.cat",
    "ev.pet_left.dog",
    "ev.fire_alert",
    "ev.fire_clear",
]
PHONE_ALERT_KEY = "ev.phone_alert"
PHONE_ALERT_SEC_BUCKETS = [5, 10, 15, 20, 30, 45, 60]


def check_tools():
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("缺少 edge-tts：pip install edge-tts", file=sys.stderr)
        sys.exit(1)
    if shutil.which("gst-launch-1.0") is None:
        print("缺少 gst-launch-1.0（GStreamer），无法把 mp3 转 wav", file=sys.stderr)
        sys.exit(1)


async def synth_mp3(text: str, voice: str, out_mp3: Path):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_mp3))


def mp3_to_wav(mp3_path: Path, wav_path: Path):
    subprocess.run(
        ["gst-launch-1.0", "-q", "filesrc", f"location={mp3_path}", "!",
         "decodebin", "!", "audioconvert", "!", "audioresample", "!",
         "wavenc", "!", "filesink", f"location={wav_path}"],
        check=True,
    )


def gen_one(key: str, lang: str, text: str, force: bool):
    out_dir = VOICE_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{key}.wav"
    if wav_path.exists() and not force:
        print(f"跳过（已存在）：{wav_path.relative_to(ROOT)}")
        return
    mp3_path = out_dir / f"{key}.mp3"
    voice = TTS_VOICE[lang]
    print(f"合成：[{lang}] {key} -> {text!r}")
    asyncio.run(synth_mp3(text, voice, mp3_path))
    mp3_to_wav(mp3_path, wav_path)
    mp3_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略已存在的文件，全部重新生成")
    args = parser.parse_args()

    check_tools()

    for key in EVENT_KEYS:
        entry = STRINGS[key]
        for lang, text in entry.items():
            gen_one(key, lang, text, args.force)

    phone_entry = STRINGS[PHONE_ALERT_KEY]
    for sec in PHONE_ALERT_SEC_BUCKETS:
        bucket_key = f"{PHONE_ALERT_KEY}.{sec}"
        for lang, template in phone_entry.items():
            gen_one(bucket_key, lang, template.format(sec=sec), args.force)

    print("完成。")


if __name__ == "__main__":
    main()
