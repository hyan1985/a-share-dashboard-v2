"""回测引擎的计算口径测试。

重点是「买入口径与基准窗口必须对齐」——这曾经是个真实 bug：
次日开盘买入，却用当日收盘算基准，导致超额收益整体偏乐观。
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.backtest import Pick, PricePanel, evaluate, sample_dates


def make_panel():
    dates = ["20260101", "20260102", "20260103", "20260104", "20260105", "20260106"]
    close = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "B": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)
    open_ = pd.DataFrame({
        "A": [10.0, 10.5, 11.5, 12.5, 13.5, 14.5],
        "B": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)
    return PricePanel(close, open_)


class PricePanelTest(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()

    def test_entry_next_open(self):
        self.assertAlmostEqual(self.panel.entry_price("20260101", "A", "next_open"), 10.5)

    def test_entry_close(self):
        self.assertAlmostEqual(self.panel.entry_price("20260101", "A", "close"), 10.0)

    def test_exit_aligned_with_entry(self):
        # horizon=2 且次日开盘买入 → pos 0 + 1 + 2 = 3 → 收盘 13
        self.assertAlmostEqual(
            self.panel.exit_price("20260101", "A", 2, offset=1), 13.0)
        # 当日收盘买入 → pos 0 + 0 + 2 = 2 → 收盘 12
        self.assertAlmostEqual(
            self.panel.exit_price("20260101", "A", 2, offset=0), 12.0)

    def test_exit_returns_none_past_end(self):
        self.assertIsNone(self.panel.exit_price("20260106", "A", 5, offset=1))

    def test_benchmark_matches_entry_style(self):
        # 次日开盘买、持有 2 日：A=(13/10.5-1)，B=(10/10-1)，等权平均
        bench = self.panel.benchmark(2, mode="next_open")
        expected = ((13 / 10.5 - 1) + 0.0) / 2
        self.assertAlmostEqual(float(bench.loc["20260101"]), expected, places=6)


class EvaluateTest(unittest.TestCase):
    def test_excess_against_benchmark(self):
        panel = make_panel()
        picks = [Pick(date="20260101", code="A", score=90.0)]
        stats = evaluate(picks, panel, horizons=(2,), entry="next_open")
        stat = stats[0]
        self.assertEqual(stat.samples, 1)
        self.assertAlmostEqual(stat.mean, 13 / 10.5 - 1, places=6)
        self.assertAlmostEqual(stat.win_rate, 1.0)
        self.assertGreater(stat.excess, 0)      # A 跑赢等权基准

    def test_missing_price_is_skipped(self):
        panel = make_panel()
        picks = [Pick(date="20260106", code="A", score=90.0)]
        stats = evaluate(picks, panel, horizons=(5,), entry="next_open")
        self.assertEqual(stats[0].samples, 0)


class SampleDatesTest(unittest.TestCase):
    def test_future_dates_excluded(self):
        class FakeStore:
            pass

        # 交易日历里含未来日期时，抽样必须裁到 cutoff
        from src.data import calendar as cal

        original = cal.open_dates
        cal.open_dates = lambda store, start=None, end=None: [
            f"2026{m:02d}{d:02d}" for m in range(1, 3) for d in (1, 2)
        ]
        try:
            dates = sample_dates(FakeStore(), days=10, step=1, warmup=2,
                                 end="20260102")
            self.assertTrue(all(d <= "20260102" for d in dates))
        finally:
            cal.open_dates = original


if __name__ == "__main__":
    unittest.main()
