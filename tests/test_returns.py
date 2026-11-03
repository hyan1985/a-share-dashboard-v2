"""价格指数守门人的测试。锁住「不许拿价格指数算红利收益」这个约束。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit import returns as ret


class GuardTest(unittest.TestCase):
    def test_price_index_is_rejected(self):
        with self.assertRaises(ret.PriceIndexError):
            ret.guard("000922.CSI")

    def test_etf_passes(self):
        ret.guard("512890.SH")          # 不抛异常即通过

    def test_allow_price_explicitly(self):
        ret.guard("000300.SH", allow_price=True)

    def test_message_mentions_dividend(self):
        try:
            ret.guard("000922.CSI")
        except ret.PriceIndexError as exc:
            self.assertIn("股息", str(exc))

    def test_known_indices_registered(self):
        for code in ("000300.SH", "000922.CSI", "000905.SH"):
            self.assertTrue(ret.is_price_only(code))
        self.assertFalse(ret.is_price_only("512890.SH"))


class MetricsTest(unittest.TestCase):
    def test_cagr_doubles_in_one_year(self):
        n = 243
        s = pd.Series([1.0 * (2 ** (i / (n - 1))) for i in range(n)])
        self.assertAlmostEqual(ret.cagr(s), 1.0, places=2)

    def test_max_drawdown(self):
        s = pd.Series([1.0, 1.2, 0.9, 1.1])
        self.assertAlmostEqual(ret.max_drawdown(s), 0.9 / 1.2 - 1, places=6)

    def test_forward_return(self):
        s = pd.Series([1.0, 1.1, 1.21])
        f = ret.forward_return(s, 1)
        self.assertAlmostEqual(f.iloc[0], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
