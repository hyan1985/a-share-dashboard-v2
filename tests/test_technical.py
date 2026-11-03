"""技术原语与存储层的单元测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.store import ParquetStore
from src.kit import technical as ta


class TechnicalTest(unittest.TestCase):
    def test_cluster_levels_merges_close_values(self):
        clusters = ta.cluster_levels([100.0, 100.2, 110.0, 100.1], tolerance=0.005)
        self.assertEqual(len(clusters), 2)
        merged = [c for c in clusters if c["touches"] == 3]
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["price"], 100.1, places=1)

    def test_cluster_levels_handles_empty(self):
        self.assertEqual(ta.cluster_levels([]), [])

    def test_box_bounds(self):
        df = pd.DataFrame({
            "trade_date": [f"202601{i:02d}" for i in range(1, 11)],
            "high": [10 + i for i in range(10)],
            "low": [5 + i for i in range(10)],
            "close": [7 + i for i in range(10)],
        })
        box = ta.box(df, window=5)
        self.assertAlmostEqual(box["high"], 19.0)
        self.assertAlmostEqual(box["low"], 10.0)

    def test_chip_peaks_returns_weighted_bins(self):
        df = pd.DataFrame({
            "trade_date": [f"202601{i:02d}" for i in range(1, 31)],
            "close": [10.0] * 25 + [20.0] * 5,
            "vol": [1000] * 25 + [100] * 5,
        })
        peaks = ta.chip_peaks(df, window=30, bin_pct=0.01, top_n=2)
        self.assertTrue(peaks)
        top = peaks[0]
        self.assertLess(top["mid"], 12.0)     # 主要在 10 元附近
        self.assertGreater(top["weight"], 0.5)

    def test_atr_positive(self):
        df = pd.DataFrame({
            "trade_date": [f"202601{i:02d}" for i in range(1, 21)],
            "high": [11.0] * 20, "low": [9.0] * 20, "close": [10.0] * 20,
        })
        self.assertAlmostEqual(ta.atr(df, window=14), 2.0, places=4)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ParquetStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_dated(self):
        df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
        self.store.write_dated("daily", "20261016", df)
        self.assertTrue(self.store.has_dated("daily", "20261016"))
        back = self.store.read_dated("daily", ["20261016"])
        self.assertEqual(len(back), 1)
        self.assertAlmostEqual(float(back["close"].iloc[0]), 10.0)

    def test_watermark_only_moves_forward(self):
        self.store.set_watermark("daily", "20261010")
        self.store.set_watermark("daily", "20261008")   # 旧日期不应覆盖
        self.assertEqual(self.store.get_watermark("daily"), "20261010")
        self.store.set_watermark("daily", "20261016")
        self.assertEqual(self.store.get_watermark("daily"), "20261016")

    def test_list_dates_sorted(self):
        for d in ("20261016", "20261014", "20261015"):
            self.store.write_dated("daily", d, pd.DataFrame({"close": [1.0]}))
        self.assertEqual(self.store.list_dates("daily"),
                         ["20261014", "20261015", "20261016"])

    def test_empty_read_returns_empty_frame(self):
        self.assertTrue(self.store.read_dated("nothing").empty)


if __name__ == "__main__":
    unittest.main()
