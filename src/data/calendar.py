"""交易日历：从 trade_cal 派生开市日、最近交易日。"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd

from .store import ParquetStore

TRADE_CAL_TABLE = "trade_cal"


def _norm(date: str) -> str:
    return date.replace("-", "")[:8]


def _fmt(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


def ensure_trade_cal(client, store: ParquetStore, start_year: int = 2008,
                     force: bool = False) -> pd.DataFrame:
    """确保本地有交易日历；缺失或过旧则从 Tushare 刷新。"""
    existing = store.read_single(TRADE_CAL_TABLE)
    if not force and not existing.empty:
        latest = str(existing["cal_date"].max())
        if latest >= dt.date.today().strftime("%Y%m%d"):
            return existing

    start = f"{start_year}0101"
    end = (dt.date.today() + dt.timedelta(days=370)).strftime("%Y%m%d")
    df = client.query("trade_cal", exchange="SSE", start_date=start, end_date=end)
    if df.empty:
        return existing
    df = df.drop_duplicates(subset=["cal_date"]).sort_values("cal_date")
    store.write_single(TRADE_CAL_TABLE, df)
    return df


def calendar_frame(store: ParquetStore) -> pd.DataFrame:
    return store.read_single(TRADE_CAL_TABLE)


def open_dates(store: ParquetStore, start: Optional[str] = None,
               end: Optional[str] = None) -> List[str]:
    """返回区间内的开市日（compact YYYYMMDD，升序）。"""
    cal = calendar_frame(store)
    if cal.empty:
        return []
    df = cal[cal["is_open"].astype(int) == 1]
    if start:
        df = df[df["cal_date"] >= _norm(start)]
    if end:
        df = df[df["cal_date"] <= _norm(end)]
    return sorted(df["cal_date"].astype(str).tolist())


def latest_open_date(store: ParquetStore, on_or_before: Optional[str] = None) -> Optional[str]:
    """不晚于指定日期的最近开市日。"""
    on_or_before = _norm(on_or_before) if on_or_before else dt.date.today().strftime("%Y%m%d")
    dates = open_dates(store, end=on_or_before)
    return dates[-1] if dates else None


def is_open(store: ParquetStore, date: str) -> bool:
    return _norm(date) in set(open_dates(store))


def previous_open_date(store: ParquetStore, date: str, n: int = 1) -> Optional[str]:
    dates = open_dates(store, end=date)
    if len(dates) <= n:
        return None
    return dates[-1 - n]


def recent_open_dates(store: ParquetStore, count: int, end: Optional[str] = None) -> List[str]:
    dates = open_dates(store, end=end)
    return dates[-count:] if count > 0 else []


def to_display(compact: str) -> str:
    return _fmt(str(compact))


def today_compact() -> str:
    return dt.date.today().strftime("%Y%m%d")


def shift_days(date_compact: str, days: int) -> str:
    d = dt.datetime.strptime(_norm(date_compact), "%Y%m%d").date()
    return (d + dt.timedelta(days=days)).strftime("%Y%m%d")


def shift_years(date_compact: str, years: int) -> str:
    d = dt.datetime.strptime(_norm(date_compact), "%Y%m%d").date()
    try:
        shifted = d.replace(year=d.year + years)
    except ValueError:          # 2 月 29 日
        shifted = d.replace(year=d.year + years, day=28)
    return shifted.strftime("%Y%m%d")
