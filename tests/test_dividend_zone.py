"""红利模块的区间判定。

这是整个筛选里唯一会直接给出「买/卖」结论的地方，
边界搞错会让买入区变成卖出区，所以单独测。
"""
import unittest

from src.modules.dividend import classify_zone


class TestClassifyZone(unittest.TestCase):
    def test_deep_below_is_buy(self):
        self.assertEqual(classify_zone(-0.25)[0], "买入区")

    def test_exact_buy_line(self):
        # 正好 -12% 算买入区（<=）
        self.assertEqual(classify_zone(-0.12)[0], "买入区")

    def test_just_above_buy_line_is_hold(self):
        self.assertEqual(classify_zone(-0.119)[0], "观望")

    def test_exact_sell_line(self):
        self.assertEqual(classify_zone(0.12)[0], "卖出区")

    def test_just_below_sell_line_is_hold(self):
        self.assertEqual(classify_zone(0.119)[0], "观望")

    def test_zero_is_hold(self):
        self.assertEqual(classify_zone(0.0)[0], "观望")

    def test_badge_matches_zone(self):
        self.assertEqual(classify_zone(-0.3)[1], "buy")
        self.assertEqual(classify_zone(0.0)[1], "hold")
        self.assertEqual(classify_zone(0.3)[1], "sell")

    def test_custom_thresholds(self):
        self.assertEqual(classify_zone(-0.10, buy_ratio=0.90)[0], "买入区")
        self.assertEqual(classify_zone(-0.10, buy_ratio=0.80)[0], "观望")


if __name__ == "__main__":
    unittest.main()
