"""regime 判定与分位数解析的回归测试。

这里锁的是一个真实修复过的 bug：单看炸板率就把
「涨 3301 / 跌 2128、指数 +0.24%」判成退潮。
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit.util import parse_pct
from src.modules.overview import OverviewModule


def history(values):
    return pd.DataFrame({"up": values})


class ParsePctTest(unittest.TestCase):
    def test_percentile_expression(self):
        series = pd.Series(range(1, 101))
        # pandas 线性插值：pos = (n-1)*q
        self.assertAlmostEqual(parse_pct("p20", series), 20.8, places=1)
        self.assertAlmostEqual(parse_pct("p90", series), 90.1, places=1)

    def test_literal_passthrough(self):
        self.assertEqual(parse_pct(3.5, pd.Series([1, 2])), 3.5)

    def test_empty_series_falls_back(self):
        self.assertIsNone(parse_pct("p20", pd.Series(dtype=float)))


class RegimeTest(unittest.TestCase):
    def setUp(self):
        self.mod = OverviewModule({}, {"overview": {
            "limit_up_cold": "p20",
            "limit_up_hot": "p80",
            "amount_pct_hot": "p90",
        }})
        self.limit_hist = history(list(range(20, 120)))
        self.amount_hist = pd.Series(range(100, 400))

    def test_broad_rally_with_high_broken_ratio_is_not_ebb(self):
        """核心回归：炸板率高但宽度强，不能判退潮。"""
        regime = self.mod._regime(
            breadth={"up_down_ratio": 1.55},
            limit_today={"up": 61, "down": 10, "broken_ratio": 0.58, "ladder": 4},
            limit_hist=self.limit_hist,
            amount_hist=self.amount_hist,
            amount_today=300,
            index_row={"pct_chg": 0.24},
        )
        self.assertNotEqual(regime["code"], "ebb")
        self.assertEqual(regime["code"], "mild")

    def test_narrow_rally_with_high_broken_ratio_is_ebb(self):
        # 涨停家数仍在常态区间（不触发冰点），但宽度转弱 + 炸板率高 → 退潮
        regime = self.mod._regime(
            breadth={"up_down_ratio": 0.6},
            limit_today={"up": 60, "down": 20, "broken_ratio": 0.5, "ladder": 2},
            limit_hist=self.limit_hist,
            amount_hist=self.amount_hist,
            amount_today=200,
            index_row={"pct_chg": -0.5},
        )
        self.assertEqual(regime["code"], "ebb")

    def test_few_limit_ups_weak_breadth_is_ice_not_ebb(self):
        """涨停极少 + 宽度弱 + 指数跌 → 冰点优先于退潮。"""
        regime = self.mod._regime(
            breadth={"up_down_ratio": 0.5},
            limit_today={"up": 8, "down": 40, "broken_ratio": 0.5, "ladder": 1},
            limit_hist=self.limit_hist,
            amount_hist=self.amount_hist,
            amount_today=150,
            index_row={"pct_chg": -1.0},
        )
        self.assertEqual(regime["code"], "ice")

    def test_strong_index_with_ladder_is_accel(self):
        regime = self.mod._regime(
            breadth={"up_down_ratio": 2.0},
            limit_today={"up": 80, "down": 5, "broken_ratio": 0.2, "ladder": 6},
            limit_hist=self.limit_hist,
            amount_hist=self.amount_hist,
            amount_today=300,
            index_row={"pct_chg": 1.2},
        )
        self.assertEqual(regime["code"], "accel")

    def test_cold_breadth_is_ice(self):
        regime = self.mod._regime(
            breadth={"up_down_ratio": 0.3},
            limit_today={"up": 5, "down": 40, "broken_ratio": 0.2, "ladder": 1},
            limit_hist=self.limit_hist,
            amount_hist=self.amount_hist,
            amount_today=100,
            index_row={"pct_chg": -1.5},
        )
        self.assertEqual(regime["code"], "ice")

    def test_regime_always_has_reason(self):
        regime = self.mod._regime(
            breadth={}, limit_today={}, limit_hist={},
            amount_hist=pd.Series(dtype=float), amount_today=0,
            index_row={},
        )
        self.assertTrue(regime["reasons"])
        self.assertIn(regime["code"], {"ice", "ebb", "overheat", "accel",
                                       "mild", "repair", "neutral"})


if __name__ == "__main__":
    unittest.main()
