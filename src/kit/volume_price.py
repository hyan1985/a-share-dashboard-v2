"""量价关系原语。

核心观点：放量本身不是信号，量价配合的**方向**才是。
所以这里输出的都是「结构特征」，具体怎么给分由各策略决定。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名与排序，只保留需要的数值列。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.sort_values("trade_date").reset_index(drop=True)
    for col in ("close", "high", "low", "open", "vol", "amount", "pct_chg"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def vol_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """当日成交量 / 近 window 日均量（不含当日）。"""
    d = _clean(df)
    if d.empty or "vol" not in d.columns:
        return pd.Series(dtype=float)
    base = d["vol"].shift(1).rolling(window).mean()
    return (d["vol"] / base).replace([float("inf"), float("-inf")], pd.NA)


def vol_persistence(df: pd.DataFrame, n: int = 5) -> Dict[str, float]:
    """量能持续性：近 n 日量比的中位数 + 趋势斜率。

    单日爆量次日萎缩的「假突破」，中位数会很低。
    """
    ratios = vol_ratio(df).dropna()
    if len(ratios) < 3:
        return {"median": 0.0, "slope": 0.0, "today": 0.0}
    tail = ratios.tail(n)
    idx = pd.Series(range(len(tail)), dtype=float)
    try:
        slope = float(idx.corr(pd.Series(tail.values, dtype=float)))
    except Exception:
        slope = 0.0
    return {
        "median": round(float(tail.median()), 3),
        "slope": round(slope if slope == slope else 0.0, 3),
        "today": round(float(ratios.iloc[-1]), 3),
    }


def divergence(df: pd.DataFrame, lookback: int = 20, vol_th: float = 0.8,
               need_days: int = 2) -> Dict[str, object]:
    """量价背离：价格创 lookback 日新高，但量比不足。

    连续 need_days 日出现才算确认（沿用 MarketMonitoring 的持续性思路）。
    """
    d = _clean(df)
    if len(d) < lookback + 2:
        return {"active": False, "days": 0}
    ratios = vol_ratio(d)
    rolling_high = d["close"].shift(1).rolling(lookback).max()
    at_high = d["close"] >= rolling_high
    weak_vol = ratios < vol_th
    flag = (at_high & weak_vol).fillna(False)

    days = 0
    for value in reversed(flag.tolist()):
        if value:
            days += 1
        else:
            break
    return {"active": days >= need_days, "days": days}


def shrink_pullback(df: pd.DataFrame, ma_window: int = 10,
                    vol_th: float = 0.7, check_days: int = 3) -> bool:
    """缩量回调：近 check_days 日下跌且量比萎缩，但收盘仍在 MA 之上。"""
    d = _clean(df)
    if len(d) < ma_window + check_days:
        return False
    ratios = vol_ratio(d)
    ma = d["close"].rolling(ma_window).mean()
    tail = d.tail(check_days)
    tail_ratio = ratios.tail(check_days)
    tail_ma = ma.tail(check_days)
    if tail_ratio.isna().any() or tail_ma.isna().any():
        return False
    falling = bool((tail["close"].diff().dropna() < 0).all())
    shrinking = bool((tail_ratio <= vol_th).all())
    holding = bool((tail["close"] > tail_ma).all())
    return falling and shrinking and holding


def breakout_with_volume(df: pd.DataFrame, box_high: float,
                         vol_th: float = 1.5) -> Dict[str, object]:
    """放量突破：收盘站上箱体上沿，且当日量比达阈。"""
    d = _clean(df)
    if d.empty:
        return {"active": False, "vol_ratio": 0.0}
    ratios = vol_ratio(d)
    today_ratio = float(ratios.iloc[-1]) if len(ratios) and ratios.iloc[-1] == ratios.iloc[-1] else 0.0
    today_close = float(d["close"].iloc[-1])
    active = bool(box_high and today_close > box_high and today_ratio >= vol_th)
    return {"active": active, "vol_ratio": round(today_ratio, 2),
            "close": today_close, "box_high": box_high}


def vp_rank_corr(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """量价秩相关：涨跌幅与量比的 Spearman 系数。

    正 = 涨放量、跌缩量（健康）；负 = 涨缩量、跌放量（危险）。
    """
    d = _clean(df)
    if len(d) < window + 6 or "pct_chg" not in d.columns:
        return None
    ratios = vol_ratio(d)
    frame = pd.DataFrame({"pct": d["pct_chg"], "vr": ratios}).dropna().tail(window)
    if len(frame) < 8:
        return None
    try:
        value = frame["pct"].corr(frame["vr"], method="spearman")
    except Exception:
        return None
    if value != value:
        return None
    return round(float(value), 3)


def turnover_structure(df: pd.DataFrame, turnover: Optional[pd.Series] = None,
                       limit_up_mask: Optional[pd.Series] = None) -> Dict[str, object]:
    """换手结构（擒龙用）。

    连板过程中换手率递减 = 惜售（强）；递增 = 分歧（弱）。
    """
    d = _clean(df)
    if d.empty or turnover is None:
        return {"declining": False, "values": []}
    series = pd.Series(turnover).reset_index(drop=True)
    if limit_up_mask is not None:
        mask = pd.Series(limit_up_mask).fillna(False).astype(bool).reset_index(drop=True)
        streak = series[mask].tail(5).tolist()
    else:
        streak = series.tail(3).tolist()
    if len(streak) < 2:
        return {"declining": False, "values": [round(v, 2) for v in streak]}
    declining = all(streak[i] >= streak[i + 1] for i in range(len(streak) - 1))
    return {"declining": declining, "values": [round(float(v), 2) for v in streak]}


def features(df: pd.DataFrame, turnover: Optional[pd.Series] = None) -> Dict[str, object]:
    """汇总一份量价特征，供各策略按需取用。"""
    persist = vol_persistence(df)
    div = divergence(df)
    corr = vp_rank_corr(df)
    return {
        "vol_ratio": persist["today"],
        "vol_median": persist["median"],
        "vol_slope": persist["slope"],
        "divergence": div["active"],
        "divergence_days": div["days"],
        "shrink_pullback": shrink_pullback(df),
        "vp_corr": corr,
        "turnover": turnover_structure(df, turnover),
    }


def score(feat: Dict[str, object]) -> float:
    """把特征压成一个 0~100 的量价分，用于排序/过滤。

    只用特征，不做绝对阈值判断，方便各策略单独调权重。
    """
    value = 50.0
    median = feat.get("vol_median") or 0.0
    corr = feat.get("vp_corr")
    if median >= 1.2:
        value += 15
    elif median >= 1.0:
        value += 8
    elif median and median < 0.7:
        value -= 10

    if corr is not None:
        value += max(-15.0, min(15.0, corr * 25))

    if feat.get("divergence"):
        value -= 20
    if feat.get("shrink_pullback"):
        value += 10

    return max(0.0, min(100.0, value))
