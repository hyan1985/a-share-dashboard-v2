"""财务数据加载：报告期覆盖度校验 + 横向对齐。

踩过的坑：`groupby.tail(3)` 是「每只股票最近 3 行」，不同股票的 3 行可能落在
不同报告期，pivot 后会出现整列 NaN，让 `(df > 0).all(axis=1)` 永远为 False。
正确做法是**先确定全局最近 N 个有效报告期**，再按报告期取值。
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..data.store import ParquetStore

FIN_TABLE = "fina_indicator"
MIN_COVERAGE = 1000          # 一个报告期至少要覆盖这么多个股才算有效


def usable_periods(store: ParquetStore, need: int = 3,
                   min_rows: int = MIN_COVERAGE,
                   as_of: str = None) -> List[str]:
    """返回最近 need 个覆盖率达标的报告期（升序）。

    as_of（YYYYMMDD）会按**公告日**截断：只有 ann_date ≤ as_of 的行才计入覆盖度。
    回测时这是必须的，否则 2024 年的截面会用到 2025 年才公告的年报。
    """
    usable: List[str] = []
    for period in store.list_dates(FIN_TABLE):
        df = store.read_dated(FIN_TABLE, [period])
        if as_of and "ann_date" in df.columns:
            df = df[df["ann_date"].astype(str).str.replace("-", "") <= str(as_of)]
        if len(df) >= min_rows:
            usable.append(period)
    return usable[-need:] if need > 0 else usable


def load(store: ParquetStore, periods: List[str],
         as_of: str = None) -> pd.DataFrame:
    if not periods:
        return pd.DataFrame()
    df = store.read_dated(FIN_TABLE, periods)
    if df.empty:
        return df
    df = df[df["end_date"].astype(str).isin([str(p) for p in periods])]
    if as_of and "ann_date" in df.columns:
        # 同一报告期可能有多条公告（更正），保留 as_of 之前最新的一条
        df = df[df["ann_date"].astype(str).str.replace("-", "") <= str(as_of)]
        df = df.sort_values(["ts_code", "end_date", "ann_date"])
        df = df.groupby(["ts_code", "end_date"], as_index=False).tail(1)
    return df


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """ts_code × 报告期 的宽表。"""
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="ts_code", columns="end_date", values=value)


def latest_row(df: pd.DataFrame) -> pd.DataFrame:
    """每个 ts_code 最新一期的行，索引为 ts_code。"""
    if df.empty:
        return pd.DataFrame()
    ordered = df.sort_values(["ts_code", "end_date"])
    return ordered.groupby("ts_code").tail(1).set_index("ts_code")


def ratio_quality(profit_dedt: pd.Series, extra_item: pd.Series) -> pd.Series:
    """利润质量 = 扣非净利润 / 净利润，净利润 = 扣非 + 非经常性损益。

    fina_indicator_vip 不返回 dtprofit_to_profit，所以用 extra_item 推导。
    """
    dedt = pd.to_numeric(profit_dedt, errors="coerce")
    extra = pd.to_numeric(extra_item, errors="coerce").fillna(0.0)
    net = dedt + extra
    return (dedt / net).where(net != 0)


def norm_ratio(value) -> Optional[float]:
    """把可能是百分数的比率归一到 0~1。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v / 100.0 if abs(v) > 1.5 else v
