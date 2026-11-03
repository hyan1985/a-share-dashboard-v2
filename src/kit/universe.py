"""股票池：元数据、基础过滤。"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..data.store import ParquetStore

# 金融企业的资产负债率没有可比性：存款、保费本身就是负债。
# 用通用杠杆门槛筛它们，会把整个行业误杀。
FINANCIAL_KEYWORDS = ("银行", "保险", "证券", "多元金融", "信托", "租赁")


def is_financial(industry: Optional[str]) -> bool:
    text = industry or ""
    return any(word in text for word in FINANCIAL_KEYWORDS)


def stock_meta(store: ParquetStore) -> pd.DataFrame:
    df = store.read_single("stock_basic")
    if df.empty:
        return pd.DataFrame(columns=["name", "industry"])
    keep = [c for c in ("name", "industry", "market", "list_date") if c in df.columns]
    return df.set_index("ts_code")[keep]


def attach_meta(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """给以 ts_code 为索引的帧补上 name / industry。"""
    if meta.empty:
        frame = frame.copy()
        frame["name"] = ""
        frame["industry"] = ""
        return frame
    out = frame.join(meta[["name", "industry"]], how="left")
    out["name"] = out.get("name").fillna("")
    out["industry"] = out.get("industry").fillna("")
    return out


def diversify(frame: pd.DataFrame, max_per_industry: int,
              industry_col: str = "industry") -> pd.DataFrame:
    """按行业限流，避免单一行业占满整个分区。

    例：破净股里银行占比极高，不限制的话「捡烟蒂」会全是城商行。
    """
    if frame.empty or max_per_industry <= 0:
        return frame
    counts = {}
    kept = []
    for code, row in frame.iterrows():
        industry = str(row.get(industry_col) or "其他")
        if counts.get(industry, 0) >= max_per_industry:
            continue
        counts[industry] = counts.get(industry, 0) + 1
        kept.append(code)
    return frame.loc[kept]


def apply_base_filters(frame: pd.DataFrame, meta: pd.DataFrame,
                       cfg: Optional[dict] = None) -> pd.DataFrame:
    """基础池过滤：ST、低价、流动性、上市时间。"""
    cfg = cfg or {}
    if frame.empty:
        return frame
    out = frame.copy()

    if not meta.empty:
        names = meta["name"].reindex(out.index).fillna("")
        if cfg.get("exclude_st", True):
            out = out[~names.str.contains("ST", case=False, na=False)]
        if cfg.get("exclude_delisting_risk", True):
            # 退市整理股（如「元道退」）PB 再低也是价值陷阱，必须排除
            names = meta["name"].reindex(out.index).fillna("")
            out = out[~names.str.contains("退", na=False)]

    if "close" in out.columns and cfg.get("min_price"):
        out = out[out["close"].astype(float) >= float(cfg["min_price"])]

    # daily.amount 单位为千元
    if "amount" in out.columns and cfg.get("min_avg_amount_wan"):
        threshold = float(cfg["min_avg_amount_wan"]) * 10.0  # 万元 → 千元
        out = out[out["amount"].astype(float) >= threshold]

    return out
