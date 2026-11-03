"""技术原语：均线、箱体、布林、摆动点、ATR、筹码峰。"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_values("trade_date").reset_index(drop=True).copy()
    for col in ("close", "high", "low", "open", "vol", "amount", "pct_chg"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ma(df: pd.DataFrame, window: int) -> Optional[float]:
    d = clean(df)
    if len(d) < window:
        return None
    value = d["close"].tail(window).mean()
    return None if value != value else float(value)


def box(df: pd.DataFrame, window: int = 60) -> Dict[str, object]:
    """箱体：近 window 日的最高/最低/振幅。"""
    d = clean(df)
    if len(d) < window:
        return {}
    part = d.tail(window)
    high = float(part["high"].max())
    low = float(part["low"].min())
    if low <= 0:
        return {}
    return {
        "high": round(high, 2),
        "low": round(low, 2),
        "amplitude_pct": round((high - low) / low * 100, 2),
    }


def boll_width(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """布林带宽 (上轨-下轨)/中轨，用收盘价近似。"""
    d = clean(df)
    if len(d) < window:
        return None
    close = d["close"].tail(window)
    mid = close.mean()
    std = close.std(ddof=0)
    if not mid or mid != mid:
        return None
    return float((4 * std) / mid)


def swing_levels(df: pd.DataFrame, windows: Sequence[int] = (20, 60)) -> List[float]:
    d = clean(df)
    levels: List[float] = []
    for n in windows:
        if len(d) >= n:
            levels.append(float(d["high"].tail(n).max()))
            levels.append(float(d["low"].tail(n).min()))
    return levels


def atr(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    d = clean(df)
    if len(d) < window + 1:
        return None
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    value = tr.tail(window).mean()
    return None if value != value else float(value)


def chip_peaks(df: pd.DataFrame, window: int = 250, bin_pct: float = 0.005,
               top_n: int = 3) -> List[Dict[str, float]]:
    """自算筹码峰：按价格分箱累加成交量，取成交量最大的若干箱体。

    Tushare 的 cyq_chips 需 10000 积分（本账号 5000），所以自己算。
    这是近似：不区分买卖方向，也不做筹码转移，只用「成交密集区」当支撑压力。
    """
    d = clean(df)
    if d.empty or "vol" not in d.columns:
        return []
    part = d.tail(window).dropna(subset=["close", "vol"])
    if part.empty:
        return []
    price = part["close"].astype(float)
    vol = part["vol"].astype(float)
    low, high = float(price.min()), float(price.max())
    if high <= low:
        return []
    step = max(high * bin_pct, 1e-6)
    bins = np.arange(low, high + step, step)
    if len(bins) < 2:
        return []
    idx = np.digitize(price.values, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    sums = np.zeros(len(bins) - 1, dtype=float)
    for i, v in zip(idx, vol.values):
        sums[i] += v
    order = np.argsort(sums)[::-1][:top_n]
    peaks = []
    for i in order:
        if sums[i] <= 0:
            continue
        peaks.append({
            "low": round(float(bins[i]), 2),
            "high": round(float(bins[i + 1]), 2),
            "mid": round(float((bins[i] + bins[i + 1]) / 2), 2),
            "weight": round(float(sums[i] / sums.sum()), 4),
        })
    return peaks


def cluster_levels(values: Sequence[float], tolerance: float = 0.005) -> List[Dict[str, float]]:
    """把候选点位按相对距离聚类，返回价格与触碰次数。"""
    vals = sorted(float(v) for v in values if v and v == v)
    if not vals:
        return []
    clusters: List[List[float]] = []
    for v in vals:
        if clusters and abs(v - clusters[-1][-1]) / max(clusters[-1][-1], 1e-9) <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [{"price": sum(c) / len(c), "touches": len(c)} for c in clusters]
