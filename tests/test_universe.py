"""股票池工具：金融业识别与行业分散。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit import universe as uni


class FinancialTest(unittest.TestCase):
    def test_banks_are_financial(self):
        self.assertTrue(uni.is_financial("银行"))
        self.assertTrue(uni.is_financial("保险"))
        self.assertTrue(uni.is_financial("证券"))

    def test_non_financial(self):
        self.assertFalse(uni.is_financial("白酒"))
        self.assertFalse(uni.is_financial("普钢"))

    def test_empty(self):
        self.assertFalse(uni.is_financial(None))
        self.assertFalse(uni.is_financial(""))


class DiversifyTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "name": ["a", "b", "c", "d", "e", "f"],
            "industry": ["银行", "银行", "银行", "银行", "白酒", "白酒"],
            "score": [99, 98, 97, 96, 50, 49],
        }, index=["1", "2", "3", "4", "5", "6"])

    def test_caps_per_industry(self):
        result = uni.diversify(self.frame(), 2)
        self.assertEqual(len(result), 4)                       # 2 银行 + 2 白酒
        self.assertEqual(list(result["industry"]), ["银行", "银行", "白酒", "白酒"])

    def test_preserves_order(self):
        result = uni.diversify(self.frame(), 2)
        self.assertEqual(list(result.index), ["1", "2", "5", "6"])

    def test_zero_cap_passthrough(self):
        result = uni.diversify(self.frame(), 0)
        self.assertEqual(len(result), 6)

    def test_empty_frame(self):
        self.assertTrue(uni.diversify(pd.DataFrame(), 2).empty)

    def test_missing_industry_treated_as_other(self):
        frame = pd.DataFrame({"score": [3, 2]}, index=["a", "b"])
        result = uni.diversify(frame, 1)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
