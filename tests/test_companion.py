"""陪伴逻辑单元测试（模拟时钟，无需摄像头 / 模型）。

运行：python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usb_person_detector.core.companion import (  # noqa: E402
    ABSENT, CAT, CELL_PHONE, DOG, EXPR_ALERT, EXPR_CURIOUS, EXPR_HAPPY, EXPR_IDLE, EXPR_NORMAL,
    EXPR_SLEEP, FAR, FIRE, MID, NEAR, PERSON, PRESENT, SLEEP, CompanionBrain, CompanionConfig,
)
from usb_person_detector.vision.detector import Detection  # noqa: E402

W, H = 1280, 720


def person(x1=500, y1=200, x2=800, y2=700, score=0.9):
    return Detection(x1, y1, x2, y2, score, PERSON, "person")


def phone(x1=600, y1=400, x2=660, y2=480, score=0.6):
    return Detection(x1, y1, x2, y2, score, CELL_PHONE, "cell phone")


def pet(cls_id=CAT, x1=100, y1=500, x2=300, y2=700):
    return Detection(x1, y1, x2, y2, 0.8, cls_id, "cat" if cls_id == CAT else "dog")


def fire(score=0.8, x1=200, y1=200, x2=400, y2=400):
    return Detection(x1, y1, x2, y2, score, FIRE, "fire")


class Sim:
    """以固定帧率推进模拟时间。"""

    def __init__(self, **cfg):
        self.t = 1000.0
        self.brain = CompanionBrain(CompanionConfig(**cfg), clock=lambda: self.t)
        self.events = []
        self.state = None

    def run(self, seconds, detections=(), fps=10):
        for _ in range(int(seconds * fps)):
            self.t += 1.0 / fps
            self.state, evs = self.brain.update(list(detections), W, H)
            self.events.extend(evs)
        return self.state

    def kinds(self):
        return [e.kind for e in self.events]


class PresenceTest(unittest.TestCase):
    def test_initial_absent(self):
        s = Sim()
        st = s.run(0.5)
        self.assertEqual(st.presence, ABSENT)
        self.assertEqual(st.expression, EXPR_IDLE)

    def test_arrive_after_debounce(self):
        s = Sim(enter_sec=1.0)
        self.assertEqual(s.run(0.5, [person()]).presence, ABSENT)   # 不到 1 秒不算到达
        st = s.run(1.0, [person()])
        self.assertEqual(st.presence, PRESENT)
        self.assertEqual(s.kinds().count("arrived"), 1)
        self.assertEqual(st.expression, EXPR_HAPPY)                  # 欢迎表情
        self.assertIn("欢迎回来", st.caption)

    def test_brief_dropout_does_not_leave(self):
        s = Sim(leave_sec=5.0)
        s.run(2, [person()])
        s.run(1, [])                                                 # 短暂丢失 1 秒
        st = s.run(1, [person()])
        self.assertEqual(st.presence, PRESENT)
        self.assertNotIn("left", s.kinds())

    def test_leave_then_sleep_then_wake(self):
        s = Sim(leave_sec=5.0, sleep_after_sec=10.0)
        s.run(2, [person()])
        self.assertEqual(s.run(6, []).presence, ABSENT)
        self.assertIn("left", s.kinds())
        st = s.run(11, [])
        self.assertEqual(st.presence, SLEEP)
        self.assertTrue(st.low_power)
        self.assertEqual(st.expression, EXPR_SLEEP)
        st = s.run(2, [person()])                                    # 回来后唤醒
        self.assertEqual(st.presence, PRESENT)
        self.assertIn("打了个盹", s.events[-1].message)

    def test_normal_expression_after_happy(self):
        s = Sim(happy_sec=3.0)
        st = s.run(6, [person()])
        self.assertEqual(st.expression, EXPR_NORMAL)

    def test_long_sitting_no_reminder(self):
        """久坐提醒已删除：连续坐 3 小时只有一次到达事件，状态里也没有已坐时长。"""
        s = Sim()
        st = s.run(3 * 3600, [person()], fps=1)
        self.assertEqual(s.kinds(), ["arrived"])
        self.assertEqual(st.expression, EXPR_NORMAL)
        self.assertFalse(hasattr(st, "sit_seconds"))


class DistanceGazeTest(unittest.TestCase):
    def test_distance_levels(self):
        self.assertEqual(Sim().run(3, [person(y1=100, y2=700)]).distance, NEAR)   # 高度比 0.83
        self.assertEqual(Sim().run(3, [person(y1=300, y2=700)]).distance, MID)    # 0.56
        self.assertEqual(Sim().run(3, [person(y1=500, y2=700)]).distance, FAR)    # 0.28

    def test_near_shows_curious(self):
        st = Sim(happy_sec=0.5).run(4, [person(y1=50, y2=710)])
        self.assertEqual(st.expression, EXPR_CURIOUS)

    def test_gaze_mirrored(self):
        # 人在画面左侧 -> 镜像后屏幕上的眼睛看向右侧（gx > 0）
        st = Sim(mirror_gaze=True).run(1, [person(x1=0, x2=200)])
        self.assertGreater(st.gaze[0], 0.5)
        st = Sim(mirror_gaze=False).run(1, [person(x1=0, x2=200)])
        self.assertLess(st.gaze[0], -0.5)

    def test_gaze_none_without_target(self):
        self.assertIsNone(Sim().run(1, []).gaze)


class PhoneTest(unittest.TestCase):
    def test_phone_alert(self):
        s = Sim(phone_alert_sec=5, phone_cooldown_sec=60)
        s.run(2, [person()])
        st = s.run(7, [person(), phone()])
        self.assertTrue(st.phone_in_use)
        self.assertEqual(s.kinds().count("phone_alert"), 1)
        self.assertEqual(st.expression, EXPR_ALERT)
        s.run(20, [person(), phone()])                               # 冷却期内不重复提醒
        self.assertEqual(s.kinds().count("phone_alert"), 1)

    def test_phone_flicker_tolerated(self):
        s = Sim(phone_alert_sec=5, phone_grace_sec=2)
        s.run(2, [person()])
        for _ in range(4):                                           # 手机检测时有时无
            s.run(1.5, [person(), phone()])
            s.run(0.5, [person()])
        self.assertIn("phone_alert", s.kinds())

    def test_phone_on_desk_ignored(self):
        s = Sim(phone_alert_sec=3)
        desk_phone = phone(x1=1100, y1=650, x2=1180, y2=700)          # 远离人框
        st = s.run(10, [person(x1=300, x2=600), desk_phone])
        self.assertFalse(st.phone_in_use)
        self.assertNotIn("phone_alert", s.kinds())

    def test_phone_without_person_ignored(self):
        s = Sim(phone_alert_sec=3)
        s.run(10, [phone()])
        self.assertNotIn("phone_alert", s.kinds())

    def test_lighter_overlapping_fire_not_phone(self):
        """点燃的打火机：同时被识别为手机和火焰时，不能算作玩手机。"""
        s = Sim(phone_alert_sec=3, fire_confirm_sec=10)   # 火焰长时间不告警，只看手机
        lighter = phone(x1=620, y1=420, x2=680, y2=490)   # 手持位置的小框
        flame = fire(score=0.5, x1=630, y1=430, x2=670, y2=470)  # 与手机框重叠
        st = s.run(10, [person(), lighter, flame])
        self.assertFalse(st.phone_in_use)
        self.assertNotIn("phone_alert", s.kinds())

    def test_thin_object_not_phone(self):
        """细长条不像手机（竖持打火机宽高比很小），应被几何过滤掉。"""
        s = Sim(phone_alert_sec=3)
        thin = phone(x1=640, y1=400, x2=655, y2=540)      # 15x140，宽高比 0.107
        st = s.run(10, [person(), thin])
        self.assertFalse(st.phone_in_use)
        self.assertNotIn("phone_alert", s.kinds())

    def test_tiny_object_not_phone(self):
        """过小的框不像手机（高度不足画面 5%）。"""
        s = Sim(phone_alert_sec=3)
        tiny = phone(x1=640, y1=400, x2=660, y2=430)      # 高端只有 30px / 720
        st = s.run(10, [person(), tiny])
        self.assertFalse(st.phone_in_use)
        self.assertNotIn("phone_alert", s.kinds())

    def test_normal_phone_still_detected(self):
        """几何过滤不能误伤真实手机（竖持 / 横持都要能识别）。"""
        for box in ((600, 400, 660, 480),     # 竖持 60x80
                    (520, 430, 720, 500)):    # 横持 200x70
            s = Sim(phone_alert_sec=3)
            st = s.run(6, [person(), phone(*box)])
            self.assertTrue(st.phone_in_use, f"{box} 应被识别为手机")
            self.assertIn("phone_alert", s.kinds(), f"{box} 应触发提醒")


class PetTest(unittest.TestCase):
    def test_cat_arrive_and_leave(self):
        s = Sim(pet_confirm_sec=1, pet_leave_sec=3)
        st = s.run(2, [pet(CAT)])
        self.assertEqual(st.pet, "猫咪")
        self.assertEqual(st.expression, EXPR_CURIOUS)
        self.assertIsNotNone(st.gaze)                                # 没人时看向宠物
        self.assertIn("pet_arrived", s.kinds())
        s.run(4, [])
        self.assertIn("pet_left", s.kinds())

    def test_pet_cooldown(self):
        s = Sim(pet_confirm_sec=1, pet_leave_sec=1, pet_cooldown_sec=30)
        s.run(2, [pet(DOG)])
        s.run(2, [])
        s.run(2, [pet(DOG)])                                         # 冷却期内再次出现不重复问候
        self.assertEqual(s.kinds().count("pet_arrived"), 1)


class FireTest(unittest.TestCase):
    def test_fire_alert_after_debounce(self):
        s = Sim(fire_confirm_sec=1.5, fire_cooldown_sec=60)
        s.run(1.0, [fire()])                       # 还没到去抖时间，不告警
        self.assertEqual(s.kinds().count("fire_alert"), 0)
        st = s.run(1.0, [fire()])                  # 累计 2 秒 > 1.5 秒，触发告警
        self.assertEqual(s.kinds().count("fire_alert"), 1)
        self.assertTrue(st.fire)
        self.assertEqual(st.expression, EXPR_ALERT)

    def test_oscillating_score_still_alerts(self):
        """置信度在阈值附近抖动（小目标火焰的真实情况）也必须能告警。

        这是峰值聚合要解决的核心问题：原“单帧连续过线”实现在这里会永远不触发。
        """
        s = Sim(fire_confirm_sec=1.0, fire_min_score=0.33, fire_min_hits=3, fire_window_sec=2.0)
        scores = [0.45, 0.28, 0.38, 0.31, 0.44, 0.27, 0.36, 0.30] * 2   # 反复跌到阈值下
        for sc in scores:
            s.run(0.1, [fire(score=sc)])
        self.assertEqual(s.kinds().count("fire_alert"), 1)
        self.assertTrue(s.state.fire)

    def test_alternating_crop_scores_still_alert(self):
        """模拟 auto 裁剪策略：奇数轮整帧分数跌破阈值、偶数轮裁剪分数过线，
        峰值聚合必须能把两者合并并告警。

        （实测：64px 火苗整帧 0.326 < 0.33 会漏检，裁剪后 0.410 可以检出）
        """
        s = Sim(fire_confirm_sec=1.0, fire_min_score=0.33, fire_min_hits=3, fire_window_sec=2.0)
        for _ in range(5):
            s.run(0.2, [fire(score=0.41)])      # 裁剪轮（过线）
            s.run(0.2, [fire(score=0.25)])      # 整帧轮（跌破阈值）
        self.assertEqual(s.kinds().count("fire_alert"), 1)
        self.assertTrue(s.state.fire)

    def test_min_hits_blocks_isolated_spike(self):
        """零星一两帧高分不能告警（需凑够 fire_min_hits 次命中）。"""
        s = Sim(fire_confirm_sec=0.1, fire_min_score=0.33, fire_min_hits=3,
                fire_window_sec=2.0, fire_clear_sec=1.0)
        s.run(0.1, [fire(score=0.9)])      # 只命中 1 次
        s.run(0.1, [fire(score=0.9)])      # 第 2 次
        st = s.run(0.2, [])                # 不再命中，不够 3 次
        self.assertFalse(st.fire)
        self.assertEqual(s.kinds().count("fire_alert"), 0)

    def test_single_frame_flicker_does_not_alert(self):
        """单帧误报不能告警——这是这个功能能不能用的关键。"""
        s = Sim(fire_confirm_sec=1.5)
        for _ in range(5):
            s.run(0.1, [fire()])       # 每次只出现 0.1 秒
            s.run(1.0, [])             # 然后消失 1 秒
        self.assertEqual(s.kinds().count("fire_alert"), 0)
        self.assertFalse(s.state.fire)

    def test_low_score_ignored(self):
        s = Sim(fire_confirm_sec=1.0, fire_min_score=0.45)
        st = s.run(3, [fire(score=0.3)])           # 置信度低于阈值，忽略
        self.assertFalse(st.fire)
        self.assertEqual(s.kinds().count("fire_alert"), 0)

    def test_fire_cooldown(self):
        s = Sim(fire_confirm_sec=0.5, fire_clear_sec=0.5, fire_cooldown_sec=60)
        s.run(2, [fire()])
        s.run(2, [])                               # 火焰消失
        s.run(2, [fire()])                         # 冷却期内再次着火，不重复告警
        self.assertEqual(s.kinds().count("fire_alert"), 1)

    def test_fire_clear_event(self):
        s = Sim(fire_confirm_sec=0.5, fire_clear_sec=0.5)
        s.run(2, [fire()])
        st = s.run(2, [])
        self.assertFalse(st.fire)
        self.assertIn("fire_clear", s.kinds())

    def test_fire_alert_even_when_absent(self):
        """没人在场时着火也必须告警。"""
        s = Sim(fire_confirm_sec=1.0)
        st = s.run(3, [fire()])
        self.assertTrue(st.fire)
        self.assertEqual(st.expression, EXPR_ALERT)
        self.assertEqual(s.kinds().count("fire_alert"), 1)

    def test_fire_not_confused_with_person(self):
        """火焰和人共用一张检测列表时不能混淆（FIRE 哨兵 ID 与 COCO PERSON=0 不冲突）。"""
        s = Sim(fire_confirm_sec=1.0)
        st = s.run(3, [person()])                 # 只有人，没有火
        self.assertFalse(st.fire)
        self.assertEqual(s.kinds().count("fire_alert"), 0)
        self.assertTrue(st.person_count)

    def test_peak_score_recorded(self):
        """窗口内取峰值分数，而不是最后一帧的分数。"""
        s = Sim(fire_confirm_sec=0.3, fire_min_score=0.30, fire_min_hits=2, fire_window_sec=3.0)
        s.run(0.4, [fire(score=0.62)])
        st = s.run(0.5, [fire(score=0.35)])        # 后续帧分更低，峰值应保留
        self.assertTrue(st.fire)
        self.assertAlmostEqual(st.fire_score, 0.62, places=2)


class PriorityTest(unittest.TestCase):
    def test_alert_not_overridden_by_pet(self):
        s = Sim(phone_alert_sec=3, pet_confirm_sec=0.5)
        s.run(2, [person()])
        s.run(4, [person(), phone()])
        st = s.run(1, [person(), phone(), pet(CAT)])
        self.assertEqual(st.expression, EXPR_ALERT)

    def test_config_from_dict_ignores_unknown(self):
        # 旧版本配置文件里的久坐参数（已删除的功能）应被忽略，不报错
        cfg = CompanionConfig.from_dict({"phone_alert_sec": 5, "sit_limit_min": 45, "unknown_key": 1})
        self.assertEqual(cfg.phone_alert_sec, 5)
        self.assertFalse(hasattr(cfg, "sit_limit_min"))


if __name__ == "__main__":
    unittest.main()
