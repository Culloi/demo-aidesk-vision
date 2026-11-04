"""语音播报单元测试：不依赖真实音频设备（mock 掉 subprocess）。

运行：python -m unittest discover -s tests -v
"""
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.audio.speech import SpeechPlayer, resolve_voice_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "assets" / "voice"


@dataclass
class FakeEvent:
    kind: str
    voice_key: str
    voice_param = None


class TestResolveVoicePath(unittest.TestCase):
    def test_simple_event(self):
        path = resolve_voice_path(VOICE_DIR, "zh", "ev.arrived")
        self.assertIsNotNone(path)
        self.assertTrue(path.name.endswith("ev.arrived.wav"))

    def test_phone_alert_bucket(self):
        path = resolve_voice_path(VOICE_DIR, "zh", "ev.phone_alert", voice_param=11)
        self.assertIsNotNone(path)
        self.assertEqual(path.name, "ev.phone_alert.10.wav")   # 11 离 10 最近

    def test_unknown_key_returns_none(self):
        path = resolve_voice_path(VOICE_DIR, "zh", "ev.not_exist")
        self.assertIsNone(path)

    def test_missing_lang_falls_back_to_zh(self):
        path = resolve_voice_path(VOICE_DIR, "fr", "ev.arrived")
        self.assertIsNotNone(path)
        self.assertIn("/zh/", str(path))


class TestSpeechPlayer(unittest.TestCase):
    def _make_player(self, enabled=True):
        popen_mock = MagicMock()
        popen_mock.return_value.wait.return_value = 0
        with patch("usb_person_detector.audio.speech.subprocess.Popen", popen_mock):
            player = SpeechPlayer(voice_dir=VOICE_DIR, enabled=enabled,
                                   player_cmd=["fake-player"])
        return player, popen_mock

    def test_speak_event_triggers_playback(self):
        player, popen_mock = self._make_player()
        try:
            with patch("usb_person_detector.audio.speech.subprocess.Popen", popen_mock):
                player.speak_event(FakeEvent("arrived", "ev.arrived"), lang="zh")
                time.sleep(0.3)
            popen_mock.assert_called_once()
            args = popen_mock.call_args[0][0]
            self.assertIn("fake-player", args)
            self.assertTrue(args[-1].endswith("ev.arrived.wav"))
        finally:
            player.shutdown()

    def test_disabled_does_not_play(self):
        player, popen_mock = self._make_player(enabled=False)
        try:
            with patch("usb_person_detector.audio.speech.subprocess.Popen", popen_mock):
                player.speak_event(FakeEvent("arrived", "ev.arrived"), lang="zh")
                time.sleep(0.2)
            popen_mock.assert_not_called()
        finally:
            player.shutdown()

    def test_unknown_voice_key_does_not_play(self):
        player, popen_mock = self._make_player()
        try:
            with patch("usb_person_detector.audio.speech.subprocess.Popen", popen_mock):
                player.speak_event(FakeEvent("x", "ev.does_not_exist"), lang="zh")
                time.sleep(0.2)
            popen_mock.assert_not_called()
        finally:
            player.shutdown()

    def test_set_enabled_false_drains_pending_queue(self):
        player, popen_mock = self._make_player()
        try:
            for _ in range(5):
                player._queue.put(VOICE_DIR / "zh" / "ev.arrived.wav")
            player.set_enabled(False)
            self.assertTrue(player._queue.empty())
        finally:
            player.shutdown()


if __name__ == "__main__":
    unittest.main()
