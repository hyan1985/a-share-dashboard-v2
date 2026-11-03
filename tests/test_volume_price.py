"""量价原语的单元测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit import volume_price as vp


def make_df(closes, vols, opens=None, highs=None, lows=None):
    n = len(closes)
    dates = [f"202601{i + 1:02d}" for i in range(n)]
    return pd.DataFrame({
        "trade_date": dates,
        "open": opens or closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
        "vol": vols,
        "pct_chg": [0.0] + [(closes[i] / closes[i - 1] - 1) * 100 for i in range(1, n)],
    })


class VolumePriceTest(unittest.TestCase):
    def test_persistence_detects_shrinking_volume(self):
        # 价格新高但量能持续萎缩
        closes = list(range(100, 112))
        vols = [1000] * 6 + [900, 800, 700, 600, 500, 400]
        feat = vp.vol_persistence(make_df(closes, vols))
        self.assertLess(feat["median"], 1.0)
        self.assertLess(feat["today"], 1.0)

    def test_persistence_detects_expanding_volume(self):
        closes = list(range(100, 112))
        vols = [1000] * 6 + [1200, 1400, 1600, 1800, 2000, 2200]
        feat = vp.vol_persistence(make_df(closes, vols))
        self.assertGreater(feat["median"], 1.0)

    def test_divergence_active_on_new_high_with_weak_volume(self):
        closes = [100] * 20 + [101, 102, 103, 104]
        vols = [1000] * 20 + [400, 380, 360, 350]
        result = vp.divergence(make_df(closes, vols), lookback=20, vol_th=0.8)
        self.assertTrue(result["active"])
        self.assertGreaterEqual(result["days"], 2)

    def test_divergence_inactive_when_volume_confirms(self):
        closes = [100] * 20 + [101, 102, 103, 104]
        vols = [1000] * 20 + [2000, 2100, 2200, 2300]
        result = vp.divergence(make_df(closes, vols), lookback=20, vol_th=0.8)
        self.assertFalse(result["active"])

    def test_shrink_pullback_requires_holding_ma(self):
        # 先涨后小幅缩量回调，但不破 MA10
        closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 119, 118.5, 118]
        vols = [1000] * 10 + [600, 550, 500]
        self.assertTrue(vp.shrink_pullback(make_df(closes, vols), ma_window=10, check_days=3))

    def test_breakout_requires_volume(self):
        closes = [100] * 10 + [106]
        weak = make_df(closes, [1000] * 10 + [900])
        strong = make_df(closes, [1000] * 10 + [2000])
        self.assertFalse(vp.breakout_with_volume(weak, box_high=105, vol_th=1.5)["active"])
        self.assertTrue(vp.breakout_with_volume(strong, box_high=105, vol_th=1.5)["active"])

    def test_score_penalises_divergence(self):
        base = {"vol_median": 1.0, "vp_corr": 0.3, "divergence": False,
                "shrink_pullback": False}
        diverged = dict(base, divergence=True)
        self.assertLess(vp.score(diverged), vp.score(base))

    def test_empty_input_does_not_crash(self):
        self.assertEqual(vp.features(pd.DataFrame()), {
            "vol_ratio": 0.0, "vol_median": 0.0, "vol_slope": 0.0,
            "divergence": False, "divergence_days": 0,
            "shrink_pullback": False, "vp_corr": None,
            "turnover": {"declining": False, "values": []},
        })


if __name__ == "__main__":
    unittest.main()
