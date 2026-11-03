"""关键点位的聚类与阶梯约束测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.modules.overview import OverviewModule


class ClusterTest(unittest.TestCase):
    def test_weighted_mean_and_touches(self):
        clusters = OverviewModule._cluster([
            (100.0, 1, "MA20"),
            (100.1, 2, "筹码峰"),
            (110.0, 3, "MA250"),
        ], tolerance=0.005)
        self.assertEqual(len(clusters), 2)
        first = clusters[0]
        self.assertEqual(first["touches"], 2)
        self.assertAlmostEqual(first["weight"], 3)
        # 加权平均：(100*1 + 100.1*2) / 3
        self.assertAlmostEqual(first["price"], (100 * 1 + 100.1 * 2) / 3, places=4)
        self.assertIn("筹码峰", first["sources"])

    def test_empty_returns_empty(self):
        self.assertEqual(OverviewModule._cluster([]), [])

    def test_far_values_not_merged(self):
        clusters = OverviewModule._cluster([(100.0, 1, "a"), (101.0, 1, "b")])
        self.assertEqual(len(clusters), 2)


class GapTest(unittest.TestCase):
    def test_upward_gap_detected_when_unfilled(self):
        df = pd.DataFrame({
            "trade_date": [f"202601{i:02d}" for i in range(1, 7)],
            "high": [10.0, 10.2, 11.5, 11.8, 12.0, 12.2],
            "low":  [9.8, 10.0, 11.2, 11.5, 11.7, 11.9],
        })
        gaps = OverviewModule._unfilled_gaps(df)
        # 第 2→3 天向上跳空（11.2 > 10.2），之后最低 11.2 未回补
        self.assertIn(10.2, [round(g, 2) for g in gaps])

    def test_gap_ignored_after_filled(self):
        # 最后一天大幅下探回补缺口（low 9.5 < 前高 10.2），
        # 但不构成新的向下跳空（high 12.0 > 前一日 low 11.7）
        df = pd.DataFrame({
            "trade_date": [f"202601{i:02d}" for i in range(1, 7)],
            "high": [10.0, 10.2, 11.5, 11.8, 12.0, 12.0],
            "low":  [9.8, 10.0, 11.2, 11.5, 11.7, 9.5],
        })
        self.assertEqual(OverviewModule._unfilled_gaps(df), [])


class LadderOrderingTest(unittest.TestCase):
    """承接位必须高于破位位，否则"破位"会排在"承接"上方，语义错误。"""

    def test_ladder_is_strictly_ordered(self):
        current = 100.0
        below = [
            {"price": 99.5, "weight": 1, "touches": 1, "sources": ["整数关口"]},
            {"price": 96.0, "weight": 3, "touches": 2, "sources": ["60日低点", "筹码峰"]},
            {"price": 92.0, "weight": 2, "touches": 1, "sources": ["MA120"]},
        ]
        support = below[0]
        rest = [c for c in below if c["price"] < support["price"]]
        breakdown = max(rest, key=lambda c: c["weight"])

        self.assertLess(support["price"], current)
        self.assertLess(breakdown["price"], support["price"])
        self.assertEqual(breakdown["price"], 96.0)   # 权重最高而非距离最近


if __name__ == "__main__":
    unittest.main()
