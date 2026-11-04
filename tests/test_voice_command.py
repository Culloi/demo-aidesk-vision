"""语音指令单元测试：不依赖真实麦克风 / ASR 模型（mock 掉录音和识别）。

运行：python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.audio.voice_command import (  # noqa: E402
    AudioRecorder, IntentMatcher, VoiceCommand, VoiceCommandPipeline,
)


class FakeAsr:
    def __init__(self, text):
        self.text = text

    def transcribe(self, wav_path):
        return self.text


class TestIntentMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = IntentMatcher()

    def test_capture_zh(self):
        cmd = self.matcher.match("帮我拍张照吧")
        self.assertEqual(cmd.intent, "capture")

    def test_capture_en(self):
        cmd = self.matcher.match("Please take a photo for me")
        self.assertEqual(cmd.intent, "capture")

    def test_lang_switch(self):
        cmd = self.matcher.match("切换英文界面")
        self.assertEqual(cmd.intent, "lang_switch")

    def test_mute(self):
        cmd = self.matcher.match("你能不能安静点")
        self.assertEqual(cmd.intent, "mute")

    def test_punctuation_ignored(self):
        cmd = self.matcher.match("你好！在吗？")
        self.assertEqual(cmd.intent, "greet")

    def test_no_match_returns_none(self):
        self.assertIsNone(self.matcher.match("今天天气怎么样"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(self.matcher.match(""))
        self.assertIsNone(self.matcher.match(None))

    def test_case_insensitive(self):
        cmd = self.matcher.match("MUTE please")
        self.assertEqual(cmd.intent, "mute")


class TestAudioRecorder(unittest.TestCase):
    def test_record_once_calls_arecord_with_expected_args(self):
        recorder = AudioRecorder(device="hw:3,0", sample_rate=16000)
        with patch("usb_person_detector.audio.voice_command.subprocess.run") as run_mock:
            path = recorder.record_once(duration_sec=3.0)
        args = run_mock.call_args[0][0]
        self.assertIn("arecord", args)
        self.assertIn("-D", args)
        self.assertIn("hw:3,0", args)
        self.assertIn("-r", args)
        self.assertIn("16000", args)
        self.assertIn("-d", args)
        self.assertIn("3", args)
        self.assertTrue(str(path).endswith(".wav"))


class TestVoiceCommandPipeline(unittest.TestCase):
    def test_full_pipeline_matches_intent(self):
        fake_asr = FakeAsr("帮我拍张照")
        fake_recorder = MagicMock()
        fake_recorder.record_once.return_value = Path("/tmp/fake.wav")
        with patch("usb_person_detector.audio.voice_command.Path.unlink"):
            pipeline = VoiceCommandPipeline(asr=fake_asr, recorder=fake_recorder)
            cmd = pipeline.listen_once(duration_sec=2.0)
        self.assertIsInstance(cmd, VoiceCommand)
        self.assertEqual(cmd.intent, "capture")
        fake_recorder.record_once.assert_called_once_with(2.0)

    def test_pipeline_returns_none_when_no_match(self):
        fake_asr = FakeAsr("今天天气不错")
        fake_recorder = MagicMock()
        fake_recorder.record_once.return_value = Path("/tmp/fake.wav")
        with patch("usb_person_detector.audio.voice_command.Path.unlink"):
            pipeline = VoiceCommandPipeline(asr=fake_asr, recorder=fake_recorder)
            cmd = pipeline.listen_once()
        self.assertIsNone(cmd)


if __name__ == "__main__":
    unittest.main()
