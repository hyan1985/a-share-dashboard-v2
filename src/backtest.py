"""回测：把选股模块放到历史日期上重跑，看未来 N 日收益。

设计取舍：
- **基准用同期全市场等权**，不是 0。否则牛市里任何策略看起来都很好。
- **默认次日开盘买入**。收盘价买入对涨停股是不现实的价格，
  擒龙这类"当日涨停"的策略用收盘价会系统性高估收益。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .context import Ctx
from .data import calendar
from .data.store import ParquetStore


# ---------------- 价格矩阵 ----------------
@dataclass
class PricePanel:
    close: pd.DataFrame            # index=trade_date, columns=ts_code
    open: pd.DataFrame

    @classmethod
    def load(cls, store: ParquetStore, dates: Sequence[str],
             adjust: bool = True) -> "PricePanel":
        """adjust=True 时用后复权价（close × adj_factor）。

        高股息策略必须复权：不复权会把每年的除息当成下跌，
        5% 股息的票三年会被低估约 15%，这是系统性偏差。
        """
        daily = store.read_dated("daily", list(dates))
        if daily.empty:
            return cls(pd.DataFrame(), pd.DataFrame())
        daily["trade_date"] = daily["trade_date"].astype(str)
        close = daily.pivot_table(index="trade_date", columns="ts_code",
                                  values="close")
        open_ = daily.pivot_table(index="trade_date", columns="ts_code",
                                  values="open")

        if adjust:
            adj = store.read_dated("adj_factor", list(dates),
                                   columns=["ts_code", "trade_date", "adj_factor"])
            if not adj.empty:
                adj["trade_date"] = adj["trade_date"].astype(str)
                factors = adj.pivot_table(index="trade_date", columns="ts_code",
                                          values="adj_factor")
                factors = factors.reindex(index=close.index, columns=close.columns)
                close = close * factors
                open_ = open_ * factors
        return cls(close.sort_index(), open_.sort_index())

    def entry_price(self, date: str, code: str, mode: str) -> Optional[float]:
        """买入价。next_open 取下一交易日开盘价。"""
        try:
            pos = self.close.index.get_indexer([date])[0]
        except Exception:
            return None
        if pos < 0:
            return None
        if mode == "close":
            value = self.close.at[date, code] if code in self.close.columns else None
        else:
            if pos + 1 >= len(self.close.index):
                return None
            next_date = self.close.index[pos + 1]
            value = (self.open.at[next_date, code]
                     if code in self.open.columns else None)
        return None if value is None or value != value else float(value)

    def exit_price(self, date: str, code: str, horizon: int,
                   offset: int = 0) -> Optional[float]:
        """持有 horizon 个交易日后的收盘价。offset 与买入口径对齐。"""
        pos = self.close.index.get_indexer([date])[0]
        target = pos + offset + horizon
        if pos < 0 or target >= len(self.close.index):
            return None
        if code not in self.close.columns:
            return None
        value = self.close.iat[target, self.close.columns.get_loc(code)]
        return None if value != value else float(value)

    def benchmark(self, horizon: int, mode: str = "next_open") -> pd.Series:
        """每个交易日全市场等权的同期收益，口径与策略一致。

        next_open：次日开盘买、持有 horizon 日后收盘卖。
        close：当日收盘买、持有 horizon 日后收盘卖。
        """
        close = self.close
        if close.empty:
            return pd.Series(dtype=float)
        if mode == "next_open" and not self.open.empty:
            entry = self.open.shift(-1)
            exit_ = close.shift(-(1 + horizon))
        else:
            entry = close
            exit_ = close.shift(-horizon)
        fwd = exit_ / entry - 1.0
        return fwd.mean(axis=1, skipna=True)


# ---------------- 回测 ----------------
@dataclass
class Pick:
    date: str
    code: str
    score: float


@dataclass
class HorizonStat:
    horizon: int
    samples: int = 0
    mean: float = 0.0
    median: float = 0.0
    win_rate: float = 0.0
    benchmark: float = 0.0

    @property
    def excess(self) -> float:
        return self.mean - self.benchmark


@dataclass
class BacktestReport:
    module_id: str
    start: str
    end: str
    sampled_days: int
    pick_days: int
    stats: List[HorizonStat] = field(default_factory=list)

    def by_year(self) -> str:
        """按自然年拆分，看策略在不同市况下是否稳定。"""
        if not self.periods:
            return ""
        buckets: Dict[str, List[PeriodResult]] = {}
        for p in self.periods:
            buckets.setdefault(p.date[:4], []).append(p)
        lines = [f"  {'年份':<6}{'期数':>4}{'策略':>9}{'等权基准':>10}{'超额':>9}"]
        for year in sorted(buckets):
            items = buckets[year]
            cum_s = cum_b = 1.0
            for item in items:
                cum_s *= 1.0 + item.ret
                cum_b *= 1.0 + item.bench
            lines.append(
                f"  {year:<6}{len(items):>4}{(cum_s-1)*100:>8.1f}%"
                f"{(cum_b-1)*100:>9.1f}%{((cum_s-cum_b))*100:>8.1f}%")
        return "\n".join(lines)

    def table(self) -> str:
        header = f"{'周期':<8}{'样本':>6}{'平均':>10}{'中位':>10}{'胜率':>8}{'基准':>10}{'超额':>10}"
        lines = [f"模块 {self.module_id} · {self.start}~{self.end} · "
                 f"抽样 {self.sampled_days} 日 · 有票 {self.pick_days} 日",
                 "-" * len(header), header, "-" * len(header)]
        for s in self.stats:
            lines.append(
                f"{str(s.horizon) + '日':<8}{s.samples:>6}"
                f"{s.mean*100:>9.2f}%{s.median*100:>9.2f}%"
                f"{s.win_rate*100:>7.0f}%{s.benchmark*100:>9.2f}%"
                f"{s.excess*100:>9.2f}%"
            )
        return "\n".join(lines)


def sample_dates(store: ParquetStore, days: int, step: int,
                 warmup: int = 140, end: Optional[str] = None) -> List[str]:
    """取最近 days 个交易日，按 step 抽样，并预留 warmup 根历史。

    上界必须裁到「今天」，否则交易日历里预置的未来日期会被当成历史。
    """
    cutoff = end or calendar.today_compact()
    all_dates = [d for d in calendar.open_dates(store) if d <= cutoff]
    if len(all_dates) <= warmup + 5:
        return []
    usable = all_dates[warmup:-1]          # 末尾留一天给 next_open
    recent = usable[-days:] if days > 0 else usable
    return recent[::max(1, step)]


def collect_picks(store: ParquetStore, thresholds: dict, module_cls,
                  cfg: dict, dates: Sequence[str],
                  progress=None) -> List[Pick]:
    picks: List[Pick] = []
    for date in dates:
        ctx = Ctx(trade_date=date, store=store, thresholds=thresholds)
        module = module_cls(cfg, thresholds)
        result = module.run(ctx)
        if result.status != "ok":
            continue
        for cand in result.candidates:
            picks.append(Pick(date=date, code=cand.code, score=cand.score))
        if progress:
            progress(date, len(result.candidates))
    return picks


def evaluate(picks: Sequence[Pick], panel: PricePanel,
             horizons: Sequence[int], entry: str = "next_open") -> List[HorizonStat]:
    stats: List[HorizonStat] = []
    offset = 1 if entry == "next_open" else 0
    for horizon in horizons:
        bench_series = panel.benchmark(horizon, mode=entry)
        returns: List[float] = []
        benches: List[float] = []
        for pick in picks:
            entry_px = panel.entry_price(pick.date, pick.code, entry)
            exit_px = panel.exit_price(pick.date, pick.code, horizon, offset=offset)
            if not entry_px or not exit_px:
                continue
            returns.append(exit_px / entry_px - 1.0)
            bench = bench_series.get(pick.date)
            if bench is not None and bench == bench:
                benches.append(float(bench))
        if not returns:
            stats.append(HorizonStat(horizon=horizon))
            continue
        series = pd.Series(returns)
        stats.append(HorizonStat(
            horizon=horizon,
            samples=len(series),
            mean=float(series.mean()),
            median=float(series.median()),
            win_rate=float((series > 0).mean()),
            benchmark=float(pd.Series(benches).mean()) if benches else 0.0,
        ))
    return stats


def run(store: ParquetStore, thresholds: dict, module_id: str, module_cls,
        cfg: dict, days: int = 200, step: int = 5,
        horizons: Sequence[int] = (5, 10, 20),
        entry: str = "next_open", verbose: bool = True) -> Optional[BacktestReport]:
    dates = sample_dates(store, days=days, step=step)
    if not dates:
        return None

    # 价格面板必须覆盖到「最后一个样本日之后」，否则最近几天的前瞻收益算不出来
    cutoff = calendar.today_compact()
    panel_dates = [d for d in calendar.open_dates(store)
                   if dates[0] <= d <= cutoff]
    panel = PricePanel.load(store, panel_dates)

    if verbose:
        print(f"  抽样 {len(dates)} 个交易日：{dates[0]} ~ {dates[-1]}")
    picks = collect_picks(store, thresholds, module_cls, cfg, dates,
                          progress=(lambda d, n: print(f"    {d} → {n} 只"))
                          if verbose else None)

    stats = evaluate(picks, panel, horizons, entry=entry)
    pick_days = len({p.date for p in picks})
    return BacktestReport(
        module_id=module_id,
        start=dates[0], end=dates[-1],
        sampled_days=len(dates), pick_days=pick_days, stats=stats,
    )


# ---------------- 组合回测（月度调仓）----------------
@dataclass
class PeriodResult:
    date: str
    next_date: str
    picks: int
    ret: float
    bench: float
    index: float = 0.0


@dataclass
class PortfolioReport:
    module_id: str
    start: str
    end: str
    periods: List[PeriodResult] = field(default_factory=list)
    period_months: int = 1        # 每期跨度：月度=1，季度=3
    cost: float = 0.001           # 每期换仓成本

    @property
    def n(self) -> int:
        return len(self.periods)

    @property
    def cum(self) -> float:
        value = 1.0
        for p in self.periods:
            value *= 1.0 + p.ret
        return value - 1.0

    @property
    def cum_index(self) -> float:
        value = 1.0
        for p in self.periods:
            value *= 1.0 + p.index
        return value - 1.0

    @property
    def cum_bench(self) -> float:
        value = 1.0
        for p in self.periods:
            value *= 1.0 + p.bench
        return value - 1.0

    def _stats(self, key: str) -> tuple:
        """(年化收益, 年化波动, 最大回撤)"""
        rets = [getattr(p, key) for p in self.periods]
        if not rets:
            return 0.0, 0.0, 0.0
        n = len(rets)
        cum = 1.0
        for r in rets:
            cum *= 1.0 + r
        months = n * max(1, self.period_months)
        annual = cum ** (12 / months) - 1.0 if cum > 0 else -1.0
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / max(n - 1, 1)
        vol = (var ** 0.5) * ((12 / max(1, self.period_months)) ** 0.5)
        peak, worst, value = 1.0, 0.0, 1.0
        for r in rets:
            value *= 1.0 + r
            peak = max(peak, value)
            worst = min(worst, value / peak - 1.0)
        return annual, vol, worst

    @property
    def win_rate(self) -> float:
        if not self.periods:
            return 0.0
        return sum(1 for p in self.periods if p.ret > p.bench) / len(self.periods)

    def by_year(self) -> str:
        """按自然年拆分，看策略在不同市况下是否稳定。"""
        if not self.periods:
            return ""
        buckets: Dict[str, List[PeriodResult]] = {}
        for p in self.periods:
            buckets.setdefault(p.date[:4], []).append(p)
        lines = [f"  {'年份':<6}{'期数':>4}{'策略':>9}{'等权基准':>10}{'超额':>9}"]
        for year in sorted(buckets):
            items = buckets[year]
            cum_s = cum_b = 1.0
            for item in items:
                cum_s *= 1.0 + item.ret
                cum_b *= 1.0 + item.bench
            lines.append(
                f"  {year:<6}{len(items):>4}{(cum_s-1)*100:>8.1f}%"
                f"{(cum_b-1)*100:>9.1f}%{((cum_s-cum_b))*100:>8.1f}%")
        return "\n".join(lines)

    def table(self) -> str:
        if not self.periods:
            return f"模块 {self.module_id}：没有可用区间"
        months = max(self.n * max(1, self.period_months), 1)
        a1, v1, d1 = self._stats("ret")
        a2, v2, d2 = self._stats("bench")
        rf = 0.02
        s1 = (a1 - rf) / v1 if v1 else 0.0
        s2 = (a2 - rf) / v2 if v2 else 0.0
        lines = [
            f"模块 {self.module_id} · {self.start}~{self.end} · {self.n} 期"
            f"{'季度' if self.period_months == 3 else '月度'}调仓",
            "-" * 52,
            f"  策略累计   {self.cum*100:+8.2f}%（已扣每期 {self.cost*100:.2f}% 换仓成本）",
            f"  基准累计   {self.cum_bench*100:+8.2f}%",
            f"  累计超额   {(self.cum-self.cum_bench)*100:+8.2f}%",
            f"  年化策略   {((1+self.cum)**(12/months)-1)*100:+8.2f}%",
            f"  年化基准   {((1+self.cum_bench)**(12/months)-1)*100:+8.2f}%",
            f"  上证累计   {self.cum_index*100:+8.2f}%",
            f"  跑赢期数   {sum(1 for p in self.periods if p.ret > p.bench)}/{self.n}"
            f"（{self.win_rate*100:.0f}%）",
            f"  最差单期   {min(p.ret for p in self.periods)*100:+8.2f}%",
            "",
            f"  {'':10}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>10}",
            f"  {'策略':10}{a1*100:>8.1f}%{v1*100:>8.1f}%{s1:>8.2f}{d1*100:>9.1f}%",
            f"  {'等权基准':10}{a2*100:>8.1f}%{v2*100:>8.1f}%{s2:>8.2f}{d2*100:>9.1f}%",
        ]
        yearly = self.by_year()
        if yearly:
            lines += ["", "  分年度：", yearly]
        return "\n".join(lines)


def month_end_dates(store: ParquetStore, warmup: int = 140,
                    end: Optional[str] = None,
                    freq: str = "monthly") -> List[str]:
    """月末（或季末）换仓日，预热 warmup 根之后。"""
    cutoff = end or calendar.today_compact()
    dates = [d for d in calendar.open_dates(store) if d <= cutoff]
    if len(dates) <= warmup + 5:
        return []
    usable = dates[warmup:]
    buckets: Dict[str, str] = {}
    for d in usable:
        buckets[d[:6]] = d            # 每月最后一个覆盖前面的
    values = sorted(buckets.values())
    if freq == "quarterly":
        # 只在 3/6/9/12 月的月末换仓
        values = [d for d in values if d[4:6] in ("03", "06", "09", "12")]
    return values


def run_portfolio(store: ParquetStore, thresholds: dict, module_id: str,
                  module_cls, cfg: dict, warmup: int = 140,
                  top_n: Optional[int] = None,
                  end: Optional[str] = None,
                  cost: float = 0.001, freq: str = "monthly",
                  verbose: bool = True) -> Optional[PortfolioReport]:
    """月末换仓：月末信号 → 次日开盘买入 → 下月末次日开盘卖出，等权。"""
    rebal = month_end_dates(store, warmup=warmup, end=end, freq=freq)
    if len(rebal) < 3:
        return None

    cutoff = end or calendar.today_compact()
    panel_dates = [d for d in calendar.open_dates(store)
                   if rebal[0] <= d <= cutoff]
    panel = PricePanel.load(store, panel_dates)
    if panel.close.empty:
        return None

    index = {d: i for i, d in enumerate(panel.close.index)}

    # 大盘对照：上证综指（市值加权），避免只用等权全市场（偏小盘）
    idx_raw = store.read_dated("index_daily", panel_dates,
                               columns=["ts_code", "trade_date", "close"])
    idx_close = (idx_raw[idx_raw["ts_code"] == "000001.SH"]
                 .set_index("trade_date")["close"].astype(float)
                 .reindex(panel.close.index)) if not idx_raw.empty else None

    report = PortfolioReport(module_id=module_id, start=rebal[0], end=rebal[-1],
                             period_months=3 if freq == "quarterly" else 1,
                             cost=cost)

    for i in range(len(rebal) - 1):
        date, nxt = rebal[i], rebal[i + 1]
        if date not in index or nxt not in index:
            continue
        entry_i = index[date] + 1
        exit_i = index[nxt] + 1
        if exit_i >= len(panel.close.index):
            continue

        ctx = Ctx(trade_date=date, store=store, thresholds=thresholds)
        result = module_cls(cfg, thresholds).run(ctx)
        codes = [c.code for c in result.candidates]
        if top_n:
            codes = codes[:top_n]

        rets = []
        for code in codes:
            if code not in panel.close.columns:
                continue
            try:
                entry = panel.open.iat[entry_i, panel.close.columns.get_loc(code)]
                exit_ = panel.close.iat[exit_i, panel.close.columns.get_loc(code)]
            except (IndexError, KeyError):
                continue
            if entry != entry or exit_ != exit_ or entry <= 0:
                continue
            rets.append(exit_ / entry - 1.0)

        bench = (panel.close.iloc[exit_i] / panel.open.iloc[entry_i] - 1.0).mean()
        ret = sum(rets) / len(rets) if rets else 0.0
        ret -= cost          # 每期换仓成本（买卖双边合计）
        idx_ret = 0.0
        if idx_close is not None and entry_i < len(idx_close) and exit_i < len(idx_close):
            a, b = idx_close.iloc[entry_i], idx_close.iloc[exit_i]
            if a == a and b == b and a > 0:
                idx_ret = float(b / a - 1.0)
        report.periods.append(PeriodResult(
            date=date, next_date=nxt, picks=len(rets),
            ret=float(ret), bench=float(bench) if bench == bench else 0.0,
            index=idx_ret))
        if verbose:
            print(f"    {date} → {nxt}  {len(rets):2d} 只  "
                  f"策略 {ret*100:+6.2f}%  基准 {bench*100:+6.2f}%")
    if report.periods:
        report.start = report.periods[0].date
        report.end = report.periods[-1].next_date
    return report
