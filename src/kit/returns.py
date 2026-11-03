"""收益计算的守门人：不许拿价格指数去算红利类资产的收益。

为什么需要这个东西：
我在两个地方犯过同一个错误——擒龙回测、定投乘数对比，都用 `000922.CSI`
（中证红利价格指数）算收益。价格指数在成分股除息时会下跌，股息从来没进模型。
实测 2016-2026 的差距：

    上证红利  价格 +73.4%   含分红 +214.5%   差 141 个百分点
    红利低波  价格 +23.2%   含分红 +152.2%   差 129 个百分点

也就是说，用价格指数算红利，答案错了一大半。

用法：
    from src.kit import returns as ret
    frame = ret.total_return_frame(store, ["512890.SH"], dates)   # 复权价
    ret.cagr(series)                                              # 年化

    传价格指数会直接抛 PriceIndexError，除非显式 allow_price=True。
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd

# 已知的「价格指数」：成分股除息时指数会下跌，不含股息再投。
# 拿它们算红利、算长期收益都会系统性低估。
PRICE_ONLY_INDICES = {
    "000001.SH": "上证综指",
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000922.CSI": "中证红利",
    "000932.SH": "中证800",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
}


class PriceIndexError(ValueError):
    """把价格指数当成收益序列使用。"""


def is_price_only(code: str) -> bool:
    return code in PRICE_ONLY_INDICES


def guard(code: str, purpose: str = "收益计算", allow_price: bool = False) -> None:
    """价格指数用于收益计算时直接报错。

    确实需要看指数涨跌（比如算回撤、看市场状态）时传 allow_price=True，
    但要在调用处写清楚"这里只是看位置，不是算收益"。
    """
    if allow_price or not is_price_only(code):
        return
    name = PRICE_ONLY_INDICES[code]
    raise PriceIndexError(
        f"{code}（{name}）是价格指数，不含股息再投，不能用于{purpose}。\n"
        f"  红利类资产的股息常占总收益一半以上，用价格指数会严重低估。\n"
        f"  改用跟踪同一指数的 ETF 复权价（total_return_frame），"
        f"或显式传 allow_price=True 表示「这里只看位置，不是算收益」。"
    )


# ---------------- 复权序列 ----------------
def total_return_frame(store, codes: Sequence[str],
                       dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """返回复权后的收盘价矩阵（ETF 或个股），价格指数会被拒绝。

    ETF 用 etf_daily × etf_adj；个股用 daily × adj_factor。
    """
    for code in codes:
        guard(code, purpose="收益计算")

    dates = list(dates) if dates is not None else None
    frames: List[pd.DataFrame] = []

    etf_codes = set()
    try:
        etf = store.read_single("etf_daily")
        if not etf.empty:
            etf_codes = set(etf["ts_code"])
    except Exception:                        # noqa: BLE001
        etf = pd.DataFrame()

    if etf_codes & set(codes):
        use = [c for c in codes if c in etf_codes]
        sub = etf[etf["ts_code"].isin(use)]
        if dates is not None:
            sub = sub[sub["trade_date"].isin(dates)]
        adj = store.read_single("etf_adj")
        if not adj.empty:
            sub = sub.merge(adj[["ts_code", "trade_date", "adj_factor"]],
                            on=["ts_code", "trade_date"], how="left")
        else:
            sub["adj_factor"] = 1.0
        sub["px"] = sub["close"].astype(float) * sub["adj_factor"].fillna(1.0)
        frames.append(sub.pivot_table(index="trade_date", columns="ts_code",
                                      values="px"))

    stock_codes = [c for c in codes if c not in etf_codes]
    if stock_codes:
        d = store.read_dated("daily", dates,
                             columns=["ts_code", "trade_date", "close"])
        if not d.empty:
            d = d[d["ts_code"].isin(stock_codes)]
            a = store.read_dated("adj_factor", dates,
                                 columns=["ts_code", "trade_date", "adj_factor"])
            if not a.empty:
                d = d.merge(a, on=["ts_code", "trade_date"], how="left")
            else:
                d["adj_factor"] = 1.0
            d["px"] = d["close"].astype(float) * d["adj_factor"].fillna(1.0)
            frames.append(d.pivot_table(index="trade_date", columns="ts_code",
                                        values="px"))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ---------------- 收益指标 ----------------
def cagr(series: pd.Series, periods_per_year: int = 243) -> float:
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    years = len(s) / periods_per_year
    if years <= 0 or s.iloc[0] <= 0:
        return 0.0
    return (s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0


def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    return float((s / s.cummax() - 1.0).min())


def forward_return(series: pd.Series, steps: int) -> pd.Series:
    s = series.dropna()
    return (s.shift(-steps) / s - 1.0)
