"""中英文翻译测试：防止漏翻、占位符不一致、代码引用了不存在的键。

运行：python -m unittest discover -s tests -v
"""
import re
import string
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from usb_person_detector.core import i18n  # noqa: E402
from usb_person_detector.core.companion import CAT, PET_KEYS  # noqa: E402
from usb_person_detector.core.i18n import LANGUAGES, STRINGS, set_language, tr  # noqa: E402
from test_companion import Sim, person, pet, phone  # noqa: E402


def placeholders(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


class TranslationTableTest(unittest.TestCase):
    def test_every_key_has_all_languages(self):
        missing = [(k, lang) for k, v in STRINGS.items() for lang in LANGUAGES if not v.get(lang)]
        self.assertEqual(missing, [], f"缺少译文: {missing}")

    def test_placeholders_match_across_languages(self):
        bad = []
        for key, entry in STRINGS.items():
            sets = {lang: placeholders(entry[lang]) for lang in LANGUAGES}
            if len({frozenset(s) for s in sets.values()}) > 1:
                bad.append((key, sets))
        self.assertEqual(bad, [], f"占位符不一致: {bad}")

    def test_literal_keys_used_in_code_exist(self):
        pattern = re.compile(r"""\btr\(\s*["']([a-z_]+\.[A-Za-z0-9_.]+)["']""")
        used = set()
        sources = (
            SRC / "usb_person_detector" / "entrypoints" / "gui_app.py",
            SRC / "usb_person_detector" / "core" / "companion.py",
        )
        for source in sources:
            used |= set(pattern.findall(source.read_text(encoding="utf-8")))
        bind = re.compile(r"""bind\([^,]+,\s*["']([a-z_]+\.[A-Za-z0-9_.]+)["']""")
        field = re.compile(r"""field_label\(\s*["']([a-z_]+\.[A-Za-z0-9_.]+)["']""")
        src = sources[0].read_text(encoding="utf-8")
        used |= set(bind.findall(src)) | set(field.findall(src))
        unknown = sorted(k for k in used if k not in STRINGS)
        self.assertGreater(len(used), 40)
        self.assertEqual(unknown, [], f"代码引用了不存在的键: {unknown}")

    def test_dynamic_key_families_exist(self):
        """代码中用 f-string 拼出来的键。"""
        families = {
            "mode": ["standby", "booting", "running", "stopping", "sleep", "error"],
            "presence": ["PRESENT", "ABSENT", "SLEEP"],
            "distance": ["NEAR", "MID", "FAR"],
            "tag": ["arrived", "left", "sleep", "phone_alert", "pet_arrived", "pet_left", "event"],
            "pet": list(PET_KEYS.values()),
            "ev.pet_arrived": list(PET_KEYS.values()),
            "ev.pet_left": list(PET_KEYS.values()),
            "label": ["person", "phone", "cat", "dog"],
            "backend": ["npu", "cpu"],
        }
        missing = [f"{p}.{s}" for p, items in families.items() for s in items if f"{p}.{s}" not in STRINGS]
        self.assertEqual(missing, [])


class SwitchTest(unittest.TestCase):
    def tearDown(self):
        set_language("zh")

    def test_switch_and_fallback(self):
        self.assertEqual(set_language("en"), "en")
        self.assertEqual(tr("btn.start"), "START")
        self.assertEqual(set_language("fr"), "zh")        # 不支持的语言回退中文
        self.assertEqual(tr("btn.start"), "启动")
        self.assertEqual(tr("no.such.key"), "no.such.key")

    def test_format(self):
        set_language("en")
        self.assertEqual(tr("phone.in_use", sec=12), "IN USE 12s")
        set_language("zh")
        self.assertEqual(tr("phone.in_use", sec=12), "使用中 12 秒")

    def test_companion_messages_follow_language(self):
        set_language("en")
        s = Sim(phone_alert_sec=3, pet_confirm_sec=0.5)
        s.run(2, [person()])
        s.run(4, [person(), phone()])
        st = s.run(1, [person(), phone(), pet(CAT)])
        messages = [e.message for e in s.events]
        self.assertIn("Welcome back!", messages)
        self.assertTrue(any("seconds on your phone" in m for m in messages))
        self.assertIn("A kitty! Hello there~", messages)
        self.assertEqual(st.pet, "Cat")
        self.assertFalse(any(re.search(r"[\u4e00-\u9fff]", m) for m in messages), "英文模式下不应出现中文")

    def test_language_switch_mid_run(self):
        """运行中切换语言：之后产生的文案使用新语言。"""
        s = Sim(leave_sec=1)
        s.run(2, [person()])
        set_language("en")
        s.run(2, [])
        self.assertEqual(s.events[0].message, "欢迎回来！")
        self.assertEqual(s.events[-1].message, "See you later~")


if __name__ == "__main__":
    unittest.main()
