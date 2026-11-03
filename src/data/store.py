"""本地 Parquet 存储：按日分区 + 原子写 + 水位线。

目录约定：
    data/parquet/{table}/{YYYYMMDD}.parquet   # 按日分区
    data/parquet/{table}.parquet              # 单文件（如 trade_cal）
    data/.watermark.json                      # 每张表已落库的最大日期
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


class ParquetStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.parquet_dir = self.root / "parquet"
        self.watermark_path = self.root / ".watermark.json"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 路径 ----------
    def table_dir(self, table: str) -> Path:
        return self.parquet_dir / table

    def dated_path(self, table: str, date: str) -> Path:
        return self.table_dir(table) / f"{date}.parquet"

    def single_path(self, table: str) -> Path:
        return self.parquet_dir / f"{table}.parquet"

    # ---------- 查询 ----------
    def has_dated(self, table: str, date: str) -> bool:
        return self.dated_path(table, date).exists()

    def list_dates(self, table: str) -> List[str]:
        d = self.table_dir(table)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.parquet"))

    # ---------- 写入（原子）----------
    def _atomic_write(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def write_dated(self, table: str, date: str, df: pd.DataFrame) -> None:
        self._atomic_write(self.dated_path(table, date), df)

    def write_single(self, table: str, df: pd.DataFrame) -> None:
        self._atomic_write(self.single_path(table), df)

    # ---------- 读取 ----------
    def read_dated(self, table: str, dates: Optional[Iterable[str]] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        if dates is None:
            dates = self.list_dates(table)
        frames = []
        for date in sorted(dates):
            path = self.dated_path(table, date)
            if path.exists():
                part = pd.read_parquet(path, columns=columns)
                if not part.empty:
                    frames.append(part)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def read_single(self, table: str) -> pd.DataFrame:
        path = self.single_path(table)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    # ---------- 水位线 ----------
    def _load_watermark(self) -> dict:
        if not self.watermark_path.exists():
            return {}
        try:
            return json.loads(self.watermark_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_watermark(self, table: str) -> Optional[str]:
        return self._load_watermark().get(table)

    def set_watermark(self, table: str, date: str) -> None:
        marks = self._load_watermark()
        current = marks.get(table)
        if current is None or date > current:
            marks[table] = date
        self.watermark_path.parent.mkdir(parents=True, exist_ok=True)
        self.watermark_path.write_text(
            json.dumps(marks, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def age_days(self, table: str, key: str) -> Optional[float]:
        """某个分区文件距今多少天；不存在返回 None。"""
        import time
        path = self.dated_path(table, key)
        if not path.exists():
            return None
        return (time.time() - path.stat().st_mtime) / 86400.0

    # ---------- 统计 ----------
    def stats(self) -> dict:
        result = {}
        if self.parquet_dir.exists():
            for entry in sorted(self.parquet_dir.iterdir()):
                if entry.is_dir():
                    files = sorted(entry.glob("*.parquet"))
                    result[entry.name] = {
                        "mode": "dated",
                        "files": len(files),
                        "first": files[0].stem if files else None,
                        "last": files[-1].stem if files else None,
                    }
                elif entry.suffix == ".parquet":
                    result[entry.stem] = {"mode": "single", "files": 1,
                                          "first": None, "last": None}
        return result
