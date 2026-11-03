"""分红历史解析测试。

tushare 的 dividend 每只股票每年有多条记录（预案 / 股东大会通过 / 实施），
只有「实施」且现金分红 > 0 才算真正派钱。这里的测试锁住这个过滤。
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit import dividend as dk


def frame(rows):
    return pd.DataFrame(rows, columns=[
        "ts_code", "end_date", "ann_date", "div_proc", "cash_div", "cash_div_tax"])


class ImplementedTest(unittest.TestCase):
    def test_only_implemented_rows_kept(self):
        df = frame([
            ("A", "20241231", "20250401", "预案", 0.0, 0.30),
            ("A", "20241231", "20250601", "股东大会通过", 0.0, 0.30),
            ("A", "20241231", "20250701", "实施", 0.30, 0.30),
        ])
        rows = dk.implemented(df)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows["_cash"].iloc[0]), 0.30)

    def test_zero_cash_dividend_excluded(self):
        df = frame([("A", "20241231", "20250701", "实施", 0.0, 0.0)])
        self.assertTrue(dk.implemented(df).empty)

    def test_falls_back_to_taxed_column(self):
        df = frame([("A", "20241231", "20250701", "实施", None, 0.25)])
        rows = dk.implemented(df)
        self.assertAlmostEqual(float(rows["_cash"].iloc[0]), 0.25)


class StreakTest(unittest.TestCase):
    def test_consecutive_years(self):
        df = frame([("A", f"202{y}1231", "2025", "实施", 0.1, 0.1)
                    for y in range(1, 6)])
        self.assertEqual(dk.streak(df), 5)

    def test_gap_breaks_streak(self):
        years = [2021, 2022, 2024, 2025]
        df = frame([("A", f"{y}1231", "2025", "实施", 0.1, 0.1) for y in years])
        # 从 2025 往回：2025-2024 连续 2，2024-2022 断开
        self.assertEqual(dk.streak(df), 2)

    def test_no_dividend(self):
        self.assertEqual(dk.streak(frame([])), 0)

    def test_interim_dividend_counts_as_year(self):
        df = frame([
            ("A", "20251231", "2026", "实施", 0.2, 0.2),
            ("A", "20260630", "2026", "实施", 0.1, 0.1),
        ])
        self.assertEqual(sorted(dk.years_paid(df)), [2025, 2026])


class LatestAnnualTest(unittest.TestCase):
    def test_only_calendar_year_end_used(self):
        df = frame([
            ("A", "20241231", "2025", "实施", 0.20, 0.20),
            ("A", "20250630", "2025", "实施", 0.05, 0.05),
            ("A", "20251231", "2026", "实施", 0.25, 0.25),
        ])
        self.assertAlmostEqual(dk.latest_annual_dps(df), 0.25)

    def test_none_when_no_annual(self):
        df = frame([("A", "20250630", "2025", "实施", 0.05, 0.05)])
        self.assertIsNone(dk.latest_annual_dps(df))


class TrailingDpsTest(unittest.TestCase):
    def test_includes_interim_within_12_months(self):
        # 年报 0.30 + 中期 0.10，滚动 12 个月应为 0.40
        df = frame([
            ("A", "20241231", "2025", "实施", 0.30, 0.30),
            ("A", "20250630", "2025", "实施", 0.10, 0.10),
        ])
        self.assertAlmostEqual(dk.trailing_dps(df), 0.40)

    def test_excludes_older_than_a_year(self):
        df = frame([
            ("A", "20231231", "2024", "实施", 0.20, 0.20),
            ("A", "20250630", "2025", "实施", 0.10, 0.10),
        ])
        self.assertAlmostEqual(dk.trailing_dps(df), 0.10)

    def test_none_when_empty(self):
        self.assertIsNone(dk.trailing_dps(pd.DataFrame()))


class SummaryTest(unittest.TestCase):
    def test_summary_shape(self):
        df = frame([("A", f"{y}1231", "2026", "实施", 0.1, 0.1)
                    for y in (2021, 2022, 2023, 2024, 2025)])
        s = dk.payout_summary(df)
        self.assertEqual(s["streak"], 5)
        self.assertEqual(s["first_year"], 2021)
        self.assertEqual(s["last_year"], 2025)
        self.assertAlmostEqual(s["latest_annual_dps"], 0.1)
        self.assertIn("trailing_dps", s)

    def test_empty_summary(self):
        s = dk.payout_summary(pd.DataFrame())
        self.assertEqual(s["streak"], 0)
        self.assertIsNone(s["latest_annual_dps"])


if __name__ == "__main__":
    unittest.main()
