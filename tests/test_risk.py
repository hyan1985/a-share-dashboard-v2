"""双轨风险模型的评分边界与归一化测试。"""
from __future__ import annotations

import unittest

from src.kit import risk as rk


class DimensionScoringTest(unittest.TestCase):
    def test_concentration_bounds(self):
        self.assertEqual(rk.score_concentration(0.40), 0.0)
        self.assertEqual(rk.score_concentration(0.45), 0.0)
        self.assertAlmostEqual(rk.score_concentration(0.485), 10.0)
        self.assertEqual(rk.score_concentration(0.55), 20.0)
        self.assertIsNone(rk.score_concentration(None))

    def test_industry_bounds(self):
        self.assertEqual(rk.score_industry(0.10), 0.0)
        self.assertAlmostEqual(rk.score_industry(0.30), 10.0)
        self.assertEqual(rk.score_industry(0.50), 20.0)

    def test_stress_bounds(self):
        self.assertEqual(rk.score_stress(0.01), 0.0)
        self.assertAlmostEqual(rk.score_stress(0.07), 12.5)
        self.assertEqual(rk.score_stress(0.20), 25.0)

    def test_breadth_inverted(self):
        # 上涨家数越多越安全
        self.assertEqual(rk.score_breadth(0.60), 0.0)
        self.assertEqual(rk.score_breadth(0.10), 15.0)
        self.assertAlmostEqual(rk.score_breadth(0.30), 7.5)

    def test_limit_takes_worse_of_two(self):
        self.assertAlmostEqual(rk.score_limit(0.30, 0), 0.0)
        self.assertAlmostEqual(rk.score_limit(0.70, 0), 10.0)
        self.assertAlmostEqual(rk.score_limit(0.30, 60), 10.0)
        self.assertIsNone(rk.score_limit(None, None))


class RenormalisationTest(unittest.TestCase):
    def test_missing_dimension_does_not_dilute(self):
        """缺维度时不应被稀释：满分的集中度应拿到结构轨满分的一半权重。"""
        # 只有资金集中度可用，且打满 → 结构轨应为满分 50
        result = rk.evaluate(concentration=0.60)
        self.assertAlmostEqual(result.structure, 50.0, places=1)

    def test_two_dimensions_share_weight(self):
        result = rk.evaluate(concentration=0.60, industry=0.50)
        self.assertAlmostEqual(result.structure, 50.0, places=1)

    def test_no_data_yields_zero(self):
        result = rk.evaluate()
        self.assertEqual(result.structure, 0.0)
        self.assertEqual(result.breakdown, 0.0)


class EvaluateTest(unittest.TestCase):
    def test_calm_market_is_safe(self):
        result = rk.evaluate(concentration=0.42, industry=0.22,
                             severe_ratio=0.01, up_ratio=0.60,
                             broken_ratio=0.20, limit_down=2)
        self.assertLess(result.total, 20)
        self.assertEqual(result.level, "安全")

    def test_stressed_market_is_danger(self):
        result = rk.evaluate(concentration=0.55, industry=0.45,
                             severe_ratio=0.15, up_ratio=0.10,
                             broken_ratio=0.80, limit_down=80)
        self.assertGreaterEqual(result.total, 60)
        self.assertEqual(result.level, "危险")

    def test_total_never_exceeds_100(self):
        result = rk.evaluate(concentration=1.0, industry=1.0, severe_ratio=1.0,
                             up_ratio=0.0, broken_ratio=1.0, limit_down=1000)
        self.assertLessEqual(result.total, 100.0)

    def test_bull_trend_caps_total(self):
        # 结构拥挤打满 + 破位分低（<20）→ 主升期封顶，避免上涨途中天天喊风险
        stressed = dict(concentration=0.55, industry=0.45, severe_ratio=0.06,
                        up_ratio=0.50, broken_ratio=0.10, limit_down=0)
        plain = rk.evaluate(**stressed, bull_trend=False)
        capped = rk.evaluate(**stressed, bull_trend=True)
        self.assertLessEqual(capped.total, 55.0)
        self.assertLess(capped.total, plain.total)
        self.assertTrue(capped.capped)

    def test_signals_only_for_high_dimensions(self):
        result = rk.evaluate(concentration=0.60, industry=0.10,
                             severe_ratio=0.01, up_ratio=0.60)
        labels = " ".join(result.signals)
        self.assertIn("资金集中度", labels)
        self.assertNotIn("恐慌扩散", labels)

    def test_missing_dimensions_reported_in_note(self):
        result = rk.evaluate(concentration=0.5, industry=0.3)
        payload = result.as_dict()
        self.assertTrue(all(d["available"] is not None for d in payload["dimensions"]))
        self.assertIn("归一化", payload["note"])


if __name__ == "__main__":
    unittest.main()
