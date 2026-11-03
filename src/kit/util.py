"""选股原语：分位数、安全除法、格式化。"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import pandas as pd


def safe_div(a: Optional[float], b: Optional[float], default: float = 0.0) -> float:
    try:
        if a is None or b is None:
            return default
        if b == 0 or math.isnan(b) or math.isnan(a):
            return default
        return float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def percentile_of(series: pd.Series, value: float) -> Optional[float]:
    """value 在 series 中的分位（0~100）。series 为空时返回 None。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float((s <= value).mean() * 100.0)


def parse_pct(spec, series: pd.Series, default: Optional[float] = None) -> Optional[float]:
    """把 "p20" 这类分位表达解析成具体数值；不是分位表达则原样返回。"""
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, str) or not spec.lower().startswith("p"):
        return default
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return default
    try:
        q = float(spec[1:]) / 100.0
    except ValueError:
        return default
    return float(s.quantile(q))


def pct_change(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """(a-b)/b。"""
    if not a or not b:
        return None
    return (a - b) / b


def fmt_yi(amount_yuan: Optional[float]) -> str:
    """元 → 亿元，保留 2 位。"""
    if amount_yuan is None:
        return "-"
    return f"{amount_yuan / 1e8:.2f} 亿"


def fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
