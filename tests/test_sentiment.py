"""情绪强度合成的边界与归一化测试。"""
from __future__ import annotations

import unittest

from src.kit import sentiment as st


class ComponentTest(unittest.TestCase):
    def test_breadth_bounds(self):
        self.assertEqual(st.breadth_score(0.10), 0.0)
        self.assertEqual(st.breadth_score(0.20), 0.0)
        self.assertAlmostEqual(st.breadth_score(0.50), 50.0)
        self.assertEqual(st.breadth_score(0.90), 100.0)
        self.assertIsNone(st.breadth_score(None))

    def test_profit_effect(self):
        # 涨停远多于跌停 → 高分
        self.assertGreater(st.profit_score(80, 5), 80)
        # 跌停多于涨停 → 低分
        self.assertLess(st.profit_score(5, 80), 20)
        # 均衡 → 50
        self.assertAlmostEqual(st.profit_score(10, 10), 50.0)

    def test_ladder_bounds(self):
        self.assertEqual(st.ladder_score(1), 0.0)
        self.assertEqual(st.ladder_score(2), 0.0)
        self.assertAlmostEqual(st.ladder_score(6), 50.0)
        self.assertEqual(st.ladder_score(12), 100.0)

    def test_seal_inverted(self):
        self.assertEqual(st.seal_score(0.0), 100.0)
        self.assertEqual(st.seal_score(1.0), 0.0)
        self.assertAlmostEqual(st.seal_score(0.4), 60.0)

    def test_volume_passthrough(self):
        self.assertEqual(st.volume_score(80), 80.0)
        self.assertIsNone(st.volume_score(None))


class CompositeTest(unittest.TestCase):
    def test_missing_components_do_not_dilute(self):
        """只给宽度且打满时，合成结果应为 100 而不是被其他缺失项拉低。"""
        self.assertAlmostEqual(st.composite({"breadth": 100.0}), 100.0)

    def test_weighted_average(self):
        # 宽度(25) 和 赚钱(25) 各 100，其余缺失 → 100
        self.assertAlmostEqual(st.composite({"breadth": 100, "profit": 100}), 100.0)
        # 一项 100、一项 0，等权 → 50
        self.assertAlmostEqual(st.composite({"breadth": 100, "profit": 0}), 50.0)

    def test_all_missing_returns_none(self):
        self.assertIsNone(st.composite({"breadth": None, "profit": None}))
        self.assertIsNone(st.composite({}))


class SummarizeTest(unittest.TestCase):
    def test_shape_and_tone(self):
        result = st.summarize(
            {"breadth": 90, "profit": 90, "ladder": 90, "seal": 90, "volume": 90},
            details={"breadth": "上涨家数占比 70%"},
            percentile=88.0,
        )
        self.assertGreater(result["score"], 80)
        self.assertEqual(result["tone"], "hot")
        self.assertEqual(result["percentile"], 88.0)
        self.assertEqual(len(result["components"]), len(st.COMPONENTS))
        self.assertIn("非股吧文本情绪", result["note"])

    def test_cold_tone(self):
        result = st.summarize({"breadth": 5, "profit": 5, "ladder": 0,
                               "seal": 10, "volume": 5})
        self.assertEqual(result["tone"], "cold")

    def test_missing_components_reported(self):
        result = st.summarize({"breadth": 60})
        missing = [c for c in result["components"] if not c["available"]]
        self.assertEqual(len(missing), len(st.COMPONENTS) - 1)
        self.assertIn("缺", result["note"])

    def test_all_missing(self):
        result = st.summarize({})
        self.assertIsNone(result["score"])
        self.assertIn("缺", result["note"])


if __name__ == "__main__":
    unittest.main()
