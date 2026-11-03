"""成交额指标的渲染与分位着色测试。

这里锁的是一个真实修过的 bug：成交额曾取 index_daily 的沪市数据，
低于全市场一半以上。
"""
from __future__ import annotations

import unittest

from src.kit.util import percentile_of
from src.render.build import _pct_tone, _render_amount


class AmountRenderTest(unittest.TestCase):
    def base_ctx(self):
        return {
            "total_yi": 15691,
            "pct_1y": 2,
            "pct_3y": 49,
            "vs_ma20": -9.7,
            "similar_days": 124,
            "history_days": 728,
            "forward": {
                "5": {"samples": 31, "median": 0.36, "win": 58},
                "10": {"samples": 27, "median": 0.02, "win": 56},
            },
        }

    def test_renders_percentiles_and_forward(self):
        html = _render_amount(self.base_ctx())
        self.assertIn("近一年分位", html)
        self.assertIn("2%", html)
        self.assertIn("49%", html)
        self.assertIn("124", html)
        self.assertIn("+0.36%", html)
        self.assertIn("样本高度重叠", html)      # 必须提示样本重叠

    def test_short_history_hides_three_year_label(self):
        """只有 1 年数据时不能显示「近三年分位」，否则两套口径数字相同、标签不同。"""
        ctx = self.base_ctx()
        ctx["history_days"] = 243
        html = _render_amount(ctx)
        self.assertIn("近一年分位", html)
        self.assertNotIn("近三年分位", html)

    def test_tiny_history_label_is_honest(self):
        ctx = self.base_ctx()
        ctx["history_days"] = 60
        html = _render_amount(ctx)
        self.assertIn("近 60 日分位", html)
        self.assertNotIn("近一年分位", html)

    def test_empty_context_degrades(self):
        self.assertIn("数据不足", _render_amount({}))
        self.assertIn("数据不足", _render_amount({"total_yi": 0}))

    def test_missing_forward_still_renders(self):
        ctx = self.base_ctx()
        ctx["forward"] = {}
        html = _render_amount(ctx)
        self.assertIn("前瞻数据不足", html)


class ToneTest(unittest.TestCase):
    def test_hot_and_cold(self):
        self.assertEqual(_pct_tone(90), "up")
        self.assertEqual(_pct_tone(10), "down")
        self.assertEqual(_pct_tone(50), "")
        self.assertEqual(_pct_tone(None), "")


class PercentileTest(unittest.TestCase):
    def test_percentile_of_value(self):
        import pandas as pd
        series = pd.Series(range(1, 101))
        self.assertAlmostEqual(percentile_of(series, 20), 20.0, places=1)
        self.assertAlmostEqual(percentile_of(series, 100), 100.0, places=1)

    def test_empty_series(self):
        import pandas as pd
        self.assertIsNone(percentile_of(pd.Series(dtype=float), 10))


if __name__ == "__main__":
    unittest.main()
