"""分红历史：把 tushare dividend 的杂乱记录整理成「连续分红年数」。

tushare 每只股票每年会有多条分红记录（预案 / 预披露 / 股东大会通过 / 实施），
只有「实施」且现金分红 > 0 才算真正派了钱。
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

IMPLEMENTED = "实施"


def _cash(df: pd.DataFrame) -> pd.Series:
    """优先用税前每股派现，缺失时退回税后。"""
    cash = pd.to_numeric(df.get("cash_div"), errors="coerce")
    if "cash_div_tax" in df.columns:
        cash = cash.fillna(pd.to_numeric(df["cash_div_tax"], errors="coerce"))
    return cash.fillna(0.0)


def implemented(df: pd.DataFrame) -> pd.DataFrame:
    """只保留已实施的现金分红记录。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "div_proc" in out.columns:
        out = out[out["div_proc"].astype(str).str.strip() == IMPLEMENTED]
    out = out.assign(_cash=_cash(out))
    out = out[out["_cash"] > 0]
    if "end_date" not in out.columns:
        return pd.DataFrame()
    out["_year"] = pd.to_numeric(
        out["end_date"].astype(str).str[:4], errors="coerce")
    return out.dropna(subset=["_year"]).assign(_year=lambda d: d["_year"].astype(int))


def years_paid(df: pd.DataFrame) -> List[int]:
    """有过现金分红的年份（升序）。"""
    rows = implemented(df)
    return sorted(rows["_year"].unique().tolist()) if not rows.empty else []


def streak(df: pd.DataFrame) -> int:
    """从最近一个有分红的年份往前数，连续的年数。

    例：[2021, 2022, 2023, 2025] → 2（2025、2024 断在 2024 缺 2023→2025）。
    实际上按年份集合连续判断：2025 有、2024 无 → 连续 1 年。
    """
    years = years_paid(df)
    if not years:
        return 0
    count = 1
    for index in range(len(years) - 1, 0, -1):
        if years[index] - years[index - 1] == 1:
            count += 1
        else:
            break
    return count


def dps_for_year(df: pd.DataFrame, year: int) -> Optional[float]:
    """指定财年的每股派现（只算年报 1231 的实施记录）。

    为什么需要它：早先的实现是「取最近一个实施年度的每股派现 ÷ 最新一期
    EPS」，两个年份经常对不上——招商银行 EPS 取 2025 年，分红只到 2024 年，
    派息率算出 17.6%（真实 35%）。那个实现已经删掉，只保留按年取数。
    """
    rows = implemented(df)
    if rows.empty:
        return None
    # 同一财年可能派两次（中期 0630 + 年度 1231），必须都算上。
    # 只取 1231 会漏掉中期分红，派息率会算成实际的一半——
    # 招商银行就是这样算出 17.6%（真实 35%）的。
    part = rows[rows["_year"] == int(year)]
    # 但同一笔分红可能在 tushare 里存在多条实施记录（公告日不同、金额相同），
    # 直接 sum 会翻倍。成都银行 2022/2023 年就是这样。
    # 按 (end_date, 金额) 去重，保留最早公告的那条。
    if not part.empty:
        part = (part.sort_values("ann_date" if "ann_date" in part.columns else "end_date")
                    .drop_duplicates(subset=["end_date", "_cash"], keep="first"))
    if part.empty:
        return None
    return float(part["_cash"].sum())


def trailing_dps(df: pd.DataFrame) -> Optional[float]:
    """滚动 12 个月每股派现（含中期分红），用来和行情的 dv_ttm 对齐。

    只取年报会漏掉中期分红，导致算出来的股息率比 dv_ttm 低一截。
    """
    rows = implemented(df)
    if rows.empty:
        return None
    end = pd.to_datetime(rows["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    rows = rows.assign(_end=end).dropna(subset=["_end"])
    if rows.empty:
        return None
    latest = rows["_end"].max()
    window = rows[rows["_end"] > latest - pd.Timedelta(days=365)]
    return float(window["_cash"].sum())


def filter_as_of(df: pd.DataFrame, as_of: str = None) -> pd.DataFrame:
    """按公告日截断。回测时避免用到回测日之后才公告的分红。"""
    if df is None or df.empty or not as_of or "ann_date" not in df.columns:
        return df
    keep = df["ann_date"].astype(str).str.replace("-", "") <= str(as_of)
    return df[keep]


def payout_summary(df: pd.DataFrame, as_of: str = None) -> dict:
    df = filter_as_of(df, as_of)
    rows = implemented(df)
    if rows.empty:
        return {"years": [], "streak": 0,
                "trailing_dps": None, "first_year": None, "last_year": None}
    years = years_paid(df)
    return {
        "years": years,
        "streak": streak(df),
        "trailing_dps": trailing_dps(df),
        "first_year": min(years),
        "last_year": max(years),
    }
