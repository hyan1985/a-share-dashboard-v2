"""财报季模块：读 SQLite 产物的去重、排序与降级。"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.modules.earnings import EarningsModule

SCHEMA = """
CREATE TABLE forecast_entry (
    ts_code TEXT, name TEXT, industry TEXT, ann_date TEXT, end_date TEXT,
    signal TEXT, report_status TEXT, close REAL, suggest_price REAL,
    dynamic_pe REAL, target_pe REAL, pe_pct REAL, upside_pct REAL,
    quality_ok INTEGER, is_one_off INTEGER, range_width_pct REAL,
    PRIMARY KEY (ts_code, end_date, ann_date)
);
"""


def row(ts_code, ann_date, signal, upside, name="X", status="未更新"):
    return (ts_code, name, "测试行业", ann_date, "20260930", signal, status,
            10.0, 9.5, 12.0, 20.0, 0.1, upside, 1, 0, 20.0)


class EarningsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "forecast.db"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, rows):
        con = sqlite3.connect(self.db)
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO forecast_entry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()

    def module(self):
        mod = EarningsModule({"params": {}},
                             {"earnings": {"db_path": str(self.db)}})
        return mod

    # ---------- 降级 ----------
    def test_missing_db_is_stale(self):
        result = self.module().compute(None)
        self.assertEqual(result["status"], "stale")
        self.assertIn("未找到", result["summary"])

    def test_empty_db_is_stale(self):
        con = sqlite3.connect(self.db)
        con.executescript(SCHEMA)
        con.commit()
        con.close()
        self.assertEqual(self.module().compute(None)["status"], "stale")

    # ---------- 正常 ----------
    def test_dedupes_by_latest_announcement(self):
        self.build([
            row("A", "20261001", "现价上车", 10),
            row("A", "20261015", "现价上车", 30),   # 更新的公告应覆盖
            row("B", "20261010", "现价上车", 20),
        ])
        result = self.module().compute(None)
        codes = [c.code for c in result["candidates"]]
        self.assertEqual(codes, ["A", "B"])
        a = next(c for c in result["candidates"] if c.code == "A")
        self.assertIn("+30%", a.metrics["距上车价"])

    def test_buy_ranks_before_wait(self):
        self.build([
            row("W", "20261010", "等待回落", 90),
            row("B", "20261010", "现价上车", 5),
        ])
        result = self.module().compute(None)
        self.assertEqual([c.code for c in result["candidates"]], ["B", "W"])

    def test_watch_and_blank_excluded(self):
        self.build([
            row("B", "20261010", "现价上车", 10),
            row("C", "20261010", "谨慎观察", 90),
            row("D", "20261010", None, 80),
        ])
        result = self.module().compute(None)
        self.assertEqual([c.code for c in result["candidates"]], ["B"])

    def test_negative_gap_flagged_when_signal_says_buy(self):
        """信号按 PE 判、距上车价按安全边际算，两者打架时必须提示。"""
        self.build([row("A", "20261010", "现价上车", -12)])
        cand = self.module().compute(None)["candidates"][0]
        self.assertEqual(cand.metrics["距上车价"], "-12%")
        self.assertTrue(any("高于上车价" in r for r in cand.risks))

    def test_badges_and_risks(self):
        self.build([row("B", "20261010", "现价上车", 10, status="已更新")])
        result = self.module().compute(None)
        cand = result["candidates"][0]
        self.assertIn("上车", cand.badges)
        self.assertIn("已更新", cand.badges)
        self.assertFalse(any("未披露" in r for r in cand.risks))

    def test_summary_counts(self):
        self.build([
            row("B", "20261010", "现价上车", 10),
            row("W", "20261010", "等待回落", 20),
            row("C", "20261010", "谨慎观察", 30),
        ])
        summary = self.module().compute(None)["summary"]
        self.assertIn("现价上车 1", summary)
        self.assertIn("等待回落 1", summary)
        self.assertIn("谨慎观察 1", summary)
        self.assertIn("2026 三季报", summary)


class PeriodLabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(EarningsModule._period_label("20260930"), "2026 三季报")
        self.assertEqual(EarningsModule._period_label("20261231"), "2026 年报")
        self.assertEqual(EarningsModule._period_label("20260630"), "2026 中报")
        self.assertEqual(EarningsModule._period_label("20260331"), "2026 一季报")


if __name__ == "__main__":
    unittest.main()
