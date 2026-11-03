"""总览：大盘情绪 regime + 大跌风险 + 关键点位。

数据取用原则：只用「小表」算历史分位，大表（daily）只取当日，
避免为了一个分位数把 250 天的全市场日线都读进内存。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..context import Ctx
from ..data import calendar
from ..kit import risk as risk_kit
from ..kit import sentiment as senti
from ..kit import text_sentiment as text_kit
from ..kit import technical
from ..kit import universe as uni
from ..kit.util import fmt_yi, parse_pct, percentile_of, safe_div
from .base import BaseModule, STATUS_EMPTY, STATUS_OK, STATUS_STALE

MAIN_INDEX = "000001.SH"
HISTORY_DAYS = 250

REGIMES = {
    "ice":     {"label": "冰点",     "emoji": "🧊", "tone": "cold"},
    "repair":  {"label": "修复",     "emoji": "🌱", "tone": "warm"},
    "mild":    {"label": "温和上涨", "emoji": "☀️", "tone": "warm"},
    "accel":   {"label": "加速",     "emoji": "🚀", "tone": "hot"},
    "overheat": {"label": "过热",    "emoji": "🔥", "tone": "hot"},
    "ebb":     {"label": "退潮",     "emoji": "🌊", "tone": "cold"},
    "neutral": {"label": "震荡",     "emoji": "➖", "tone": "flat"},
}


class OverviewModule(BaseModule):
    id = "overview"
    title = "总览"
    kind = "environment"
    requires = ["daily", "limit_list_d", "index_daily"]

    # ---------------- 主流程 ----------------
    def compute(self, ctx: Ctx) -> dict:
        index_row = self._index_row(ctx)
        daily_today = ctx.on("daily")
        breadth = self._breadth(ctx, daily_today)
        limit_hist, limit_today = self._limit_stats(ctx)
        amount_hist, amount_today = self._amount_series(ctx, index_row)
        amount_ctx = self._amount_context(ctx, amount_hist, amount_today)

        regime = self._regime(breadth, limit_today, limit_hist,
                              amount_hist, amount_today, index_row)
        sentiment = self._sentiment(ctx, breadth, limit_today, limit_hist,
                                    amount_hist, amount_ctx)
        dividend_valuation = self._dividend_valuation(ctx)
        concentration, conc_pct = self._concentration(ctx)
        industry, ind_pct = self._industry_share(ctx, daily_today)
        risk = self._risk(ctx, breadth, limit_today, index_row,
                          concentration, conc_pct, industry, ind_pct)
        levels = self._levels(ctx, index_row)

        context = {
            "index": index_row,
            "amount": amount_ctx,
            "sentiment": sentiment,
            "dividend_valuation": dividend_valuation,
            "breadth": breadth,
            "limit": limit_today,
            "regime": regime,
            "risk": risk,
            "levels": levels,
        }
        return {
            "status": STATUS_OK,
            "context": context,
            "summary": self.summarize({"context": context}, ctx),
        }

    # ---------------- 指数 ----------------
    def _index_row(self, ctx: Ctx) -> dict:
        idx = ctx.on("index_daily")
        if idx.empty:
            return {}
        row = idx[idx["ts_code"] == MAIN_INDEX]
        if row.empty:
            row = idx.head(1)
        r = row.iloc[0]
        return {
            "code": str(r.get("ts_code", MAIN_INDEX)),
            "close": float(r.get("close", 0) or 0),
            "pct_chg": float(r.get("pct_chg", 0) or 0),
            "amount": float(r.get("amount", 0) or 0),
        }

    # ---------------- 市场宽度 ----------------
    def _breadth(self, ctx: Ctx, daily: pd.DataFrame = None) -> dict:
        daily = ctx.on("daily") if daily is None else daily
        if daily.empty:
            return {}
        pct = pd.to_numeric(daily.get("pct_chg"), errors="coerce").dropna()
        up = int((pct > 0).sum())
        down = int((pct < 0).sum())
        flat = int((pct == 0).sum())
        # 跌幅≥5% 占比：恐慌扩散
        panic = int((pct <= -5).sum())
        return {
            "total": int(len(pct)),
            "up": up,
            "down": down,
            "flat": flat,
            "up_down_ratio": round(safe_div(up, down, 0.0), 2),
            "panic_ratio": round(safe_div(panic, len(pct), 0.0), 4),
            "median_pct": round(float(pct.median()), 2) if len(pct) else 0.0,
        }

    # ---------------- 涨跌停 ----------------
    def _limit_frames(self, ctx: Ctx, days: int):
        dates = calendar.recent_open_dates(ctx.store, days, end=ctx.trade_date)
        df = ctx.store.read_dated("limit_list_d", dates)
        return df

    def _limit_stats(self, ctx: Ctx):
        hist = self._limit_frames(ctx, HISTORY_DAYS)
        today = ctx.on("limit_list_d")
        stats = self._limit_counts(today)
        history = {}
        if not hist.empty and "trade_date" in hist.columns:
            grouped = hist.groupby("trade_date")
            rows = []
            index = []
            for date, g in grouped:
                counts = self._limit_counts(g)
                index.append(date)
                rows.append({
                    "up": counts["up"],
                    "down": counts["down"],
                    "broken": counts["broken"],
                    "broken_ratio": counts["broken_ratio"],
                    "ladder": counts["ladder"],
                })
            if rows:
                hdf = pd.DataFrame(rows, index=index)
                for key in ("up", "down", "broken_ratio", "ladder"):
                    history[key] = hdf[key]
        return history, stats

    @staticmethod
    def _limit_counts(df: pd.DataFrame) -> dict:
        empty = {"up": 0, "down": 0, "broken": 0, "broken_ratio": 0.0, "ladder": 0}
        if df is None or df.empty or "limit" not in df.columns:
            return empty
        flags = df["limit"].astype(str).str.upper()
        up = int((flags == "U").sum())
        down = int((flags == "D").sum())
        broken = int((flags == "Z").sum())
        ladder = 0
        if "limit_times" in df.columns:
            series = pd.to_numeric(df["limit_times"], errors="coerce").dropna()
            ladder = int(series.max()) if len(series) else 0
        return {
            "up": up,
            "down": down,
            "broken": broken,
            "broken_ratio": round(safe_div(broken, up + broken, 0.0), 4),
            "ladder": ladder,
        }

    # ---------------- 成交额（全市场）----------------
    def _amount_series(self, ctx: Ctx, index_row: dict):
        """全市场成交额（沪 + 深 + 北），单位统一成元。

        踩过的坑：原先取 index_daily 的 000001.SH.amount，那只是沪市
        （约占全市场 47%），会把总成交额低估一半以上。
        正确做法是 daily 表按日对所有个股求和。
        """
        dates = calendar.recent_open_dates(ctx.store, 750, end=ctx.trade_date)
        daily = ctx.store.read_dated("daily", dates,
                                     columns=["trade_date", "amount"])
        if daily.empty:
            fallback = float(index_row.get("amount") or 0) * 1000.0
            return pd.Series(dtype=float), fallback
        agg = (daily.groupby("trade_date")["amount"].sum() * 1000.0).sort_index()
        today = float(agg.get(ctx.trade_date, agg.iloc[-1]))
        return agg, today

    # ---------------- 量能位置 ----------------
    def _amount_context(self, ctx: Ctx, agg: pd.Series, current: float) -> dict:
        """回答「现在的成交额算多还是少」。

        不给「历史同等成交额对应的指数点位」——成交额和点位各带趋势，
        匹配出来的点位是散乱的。改为三个可解释的量：
        历史分位、相对 20 日均值、以及历史上同量能之后指数的实际走势。
        """
        if agg is None or len(agg) < 20 or current <= 0:
            return {}
        pct_1y = percentile_of(agg.tail(250), current)
        pct_3y = percentile_of(agg.tail(750), current)
        ma20 = float(agg.tail(20).mean())

        band = agg[(agg >= current * 0.85) & (agg <= current * 1.15)]
        close = self._index_close(ctx, list(agg.index))
        forward = {}
        if not close.empty and len(band):
            for horizon in (5, 10, 20):
                rets = []
                for date in band.index:
                    if date not in close.index:
                        continue
                    pos = close.index.get_loc(date)
                    if pos + horizon >= len(close):
                        continue
                    base = float(close.iloc[pos])
                    if base <= 0:
                        continue
                    rets.append(float(close.iloc[pos + horizon]) / base - 1.0)
                if rets:
                    series = pd.Series(rets)
                    forward[str(horizon)] = {
                        "samples": int(len(series)),
                        "median": round(float(series.median()) * 100, 2),
                        "win": round(float((series > 0).mean()) * 100, 0),
                    }

        return {
            "total_yi": round(current / 1e8, 0),
            "pct_1y": round(pct_1y, 0) if pct_1y is not None else None,
            "pct_3y": round(pct_3y, 0) if pct_3y is not None else None,
            "vs_ma20": round((current / ma20 - 1) * 100, 1) if ma20 else None,
            "similar_days": int(len(band)),
            "forward": forward,
            "history_days": int(len(agg)),
        }

    def _index_close(self, ctx: Ctx, dates: List[str]) -> pd.Series:
        idx = ctx.store.read_dated("index_daily", dates,
                                   columns=["ts_code", "trade_date", "close"])
        if idx.empty:
            return pd.Series(dtype=float)
        main = idx[idx["ts_code"] == MAIN_INDEX]
        if main.empty:
            return pd.Series(dtype=float)
        return main.set_index("trade_date")["close"].astype(float).sort_index()

    # ---------------- regime ----------------
    def _regime(self, breadth, limit_today, limit_hist, amount_hist,
                amount_today, index_row=None) -> dict:
        """六档状态机。

        原则：单一指标不足以定档。炸板率高但宽度强，是结构性行情而非退潮；
        同理「连板高度大」必须配合指数涨幅才算加速。
        """
        cfg = self.thresholds.get("overview") or {}
        reasons: List[str] = []

        up = limit_today.get("up", 0)
        down = limit_today.get("down", 0)
        broken_ratio = limit_today.get("broken_ratio", 0.0) or 0.0
        ladder = limit_today.get("ladder", 0)
        ratio = breadth.get("up_down_ratio", 0.0) or 0.0
        pct_chg = float((index_row or {}).get("pct_chg", 0.0) or 0.0)

        hist_up = limit_hist.get("up", pd.Series())
        p_up_low = parse_pct(cfg.get("limit_up_cold", "p20"), hist_up)
        p_up_high = parse_pct(cfg.get("limit_up_hot", "p80"), hist_up)
        p_amount_hot = parse_pct(cfg.get("amount_pct_hot", "p90"), amount_hist)

        def hit(cond, text):
            if cond:
                reasons.append(text)
            return cond

        code = "neutral"
        if hit(up <= (p_up_low if p_up_low is not None else 20)
               and ratio <= 0.6 and pct_chg < 0,
               f"涨停仅 {up} 家、涨跌比 {ratio:.2f}、指数 {pct_chg:+.2f}%"):
            code = "ice"
        elif hit(broken_ratio >= 0.45 and ratio < 1.0,
                 f"炸板率 {broken_ratio*100:.0f}% 且涨跌比仅 {ratio:.2f}"):
            code = "ebb"
        elif hit(p_amount_hot is not None and amount_today >= p_amount_hot
                 and ratio >= 2.0 and p_up_high is not None and up >= p_up_high,
                 f"成交额与涨停家数同处近一年高位，涨跌比 {ratio:.2f}"):
            code = "overheat"
        elif hit(pct_chg >= 0.8 and up >= 60 and ladder >= 5,
                 f"指数 {pct_chg:+.2f}%、涨停 {up} 家、最高 {ladder} 连板"):
            code = "accel"
        elif hit(ratio >= 1.3 and up >= 30 and pct_chg > 0,
                 f"涨跌比 {ratio:.2f}、涨停 {up} 家、指数 {pct_chg:+.2f}%"):
            code = "mild"
        elif hit(ratio >= 0.85 and broken_ratio < 0.35,
                 f"涨跌比 {ratio:.2f}，封板质量尚可"):
            code = "repair"

        meta = REGIMES.get(code, REGIMES["neutral"])
        return {
            "code": code,
            "label": meta["label"],
            "emoji": meta["emoji"],
            "tone": meta["tone"],
            "reasons": reasons or ["各指标处于中间区域，无明确方向"],
        }

    # ---------------- 红利估值 ----------------
    def _dividend_valuation(self, ctx: Ctx) -> dict:
        """A 股高股息个股的「中位股息率」历史分位。

        股息率越高 = 红利资产越便宜。用月度采样控制读取量（10 年约 120 个点）。
        这是「红利贵不贵」的代理指标，比指数点位稳定，不会随成分股漂移。
        """
        dates = calendar.recent_open_dates(ctx.store, 2600, end=ctx.trade_date)
        if len(dates) < 60:
            return {}
        buckets: dict = {}
        for d in dates:
            buckets[d[:6]] = d          # 每月最后一个交易日
        sample = sorted(buckets.values())

        values = {}
        for day in sample:
            frame = ctx.store.read_dated("daily_basic", [day],
                                         columns=["ts_code", "dv_ttm"])
            if frame.empty:
                continue
            dv = pd.to_numeric(frame["dv_ttm"], errors="coerce")
            dv = dv[(dv > 0) & (dv < 20)]
            if len(dv) >= 100:
                # 只看高股息组（≥3%），全市场中位数会被不分红的成长股稀释
                high = dv[dv >= 3.0]
                values[day] = float((high if len(high) >= 50 else dv).median())

        # 今天单独算一次（sample 里可能没有今天）
        today = values.get(ctx.trade_date)
        if today is None:
            frame = ctx.on("daily_basic")
            if not frame.empty and "dv_ttm" in frame.columns:
                dv = pd.to_numeric(frame["dv_ttm"], errors="coerce")
                dv = dv[(dv > 0) & (dv < 20)]
                if len(dv) >= 100:
                    high = dv[dv >= 3.0]
                    today = float((high if len(high) >= 50 else dv).median())
                    values[ctx.trade_date] = today
        if today is None or not values:
            return {}

        series = pd.Series(values).sort_index()
        pct = percentile_of(series, today)
        if pct is None:
            return {}
        # 股息率高 → 便宜。分位直接对应「便宜程度」
        if pct >= 80:
            level, tone = "便宜", "cold"
        elif pct >= 60:
            level, tone = "偏便宜", "cold"
        elif pct >= 40:
            level, tone = "中性", "flat"
        elif pct >= 20:
            level, tone = "偏贵", "warm"
        else:
            level, tone = "贵", "hot"
        return {
            "median_dv": round(today, 2),
            "percentile": round(float(pct), 0),
            "level": level,
            "tone": tone,
            "samples": int(len(series)),
            "min_dv": round(float(series.min()), 2),
            "max_dv": round(float(series.max()), 2),
        }

    # ---------------- 情绪强度 ----------------
    @staticmethod
    def _at(series, date):
        if series is None or len(series) == 0:
            return None
        value = series.get(date)
        if value is None or value != value:
            return None
        return float(value)

    def _sentiment(self, ctx: Ctx, breadth: dict, limit_today: dict,
                   limit_hist: dict, amount_hist: pd.Series,
                   amount_ctx: dict) -> dict:
        up = int(breadth.get("up") or 0)
        down = int(breadth.get("down") or 0)
        total = up + down
        up_ratio = (up / total) if total else None

        parts = {
            "breadth": senti.breadth_score(up_ratio),
            "profit": senti.profit_score(limit_today.get("up"), limit_today.get("down")),
            "ladder": senti.ladder_score(limit_today.get("ladder")),
            "seal": senti.seal_score(limit_today.get("broken_ratio")),
            "volume": senti.volume_score(amount_ctx.get("pct_1y")),
        }
        details = {
            "breadth": f"上涨家数占比 {up_ratio*100:.1f}%" if up_ratio is not None else "",
            "profit": f"涨停 {limit_today.get('up', 0)} / 跌停 {limit_today.get('down', 0)}",
            "ladder": f"最高 {limit_today.get('ladder', 0)} 连板",
            "seal": f"炸板率 {(limit_today.get('broken_ratio') or 0)*100:.0f}%",
            "volume": f"成交额近一年分位 {amount_ctx.get('pct_1y', 0):.0f}%",
        }

        guba = self._guba_sentiment(ctx)
        parts["guba"] = senti.guba_score(guba.get("sentiment"))
        if guba:
            details["guba"] = (f"股吧 {guba['posts']} 条，"
                               f"看多 {guba['bull_posts']} / 看空 {guba['bear_posts']}")

        current = senti.composite(parts)

        # 历史回算里没有股吧数据，分位必须用同一口径（去掉股吧分项）比较，
        # 否则是拿 6 项分数去比 5 项分数。
        core_now = senti.composite({k: v for k, v in parts.items() if k != "guba"})
        history = self._sentiment_history(ctx, limit_hist, amount_hist)
        percentile = (percentile_of(history, core_now)
                      if core_now is not None and len(history) else None)

        result = senti.summarize(parts, details, percentile)
        result["guba"] = guba or None
        result["percentile_basis"] = "按不含股吧分项的同一口径回算"
        if guba:
            result["note"] = (f"含股吧文本情绪（{guba['posts']} 条帖子，"
                              f"多空净分 {guba['sentiment']:+.2f}）")
        else:
            result["note"] = "股吧情绪未获取（先执行 make fetch），已按可用分项归一化"
        return result

    def _guba_sentiment(self, ctx: Ctx) -> dict:
        """读本地缓存的股吧帖子并打分；模块本身不发网络请求。"""
        posts = ctx.store.read_dated("guba", [ctx.trade_date])
        if posts.empty:
            return {}
        if getattr(self, "_lexicon", None) is None:
            self._lexicon = text_kit.load_lexicon()
        return text_kit.analyze(posts, self._lexicon)

    def _sentiment_history(self, ctx: Ctx, limit_hist: dict,
                           amount_hist: pd.Series) -> pd.Series:
        """用同样口径回算近 250 日的情绪分，用于分位比较。"""
        dates = calendar.recent_open_dates(ctx.store, 250, end=ctx.trade_date)
        daily = ctx.store.read_dated("daily", dates,
                                     columns=["trade_date", "pct_chg"])
        if daily.empty:
            return pd.Series(dtype=float)
        pct = pd.to_numeric(daily["pct_chg"], errors="coerce")
        breadth = daily.assign(_up=(pct > 0)).groupby("trade_date")["_up"].mean()

        ups = limit_hist.get("up")
        downs = limit_hist.get("down")
        ladders = limit_hist.get("ladder")
        broken = limit_hist.get("broken_ratio")

        scores = []
        for date in breadth.index:
            volume_pct = None
            if amount_hist is not None and len(amount_hist):
                value = amount_hist.get(date)
                window = amount_hist.loc[:date].tail(250)
                if value is not None and len(window) >= 20:
                    volume_pct = percentile_of(window, float(value))
            value = senti.composite({
                "breadth": senti.breadth_score(float(breadth.get(date))),
                "profit": senti.profit_score(self._at(ups, date), self._at(downs, date)),
                "ladder": senti.ladder_score(self._at(ladders, date)),
                "seal": senti.seal_score(self._at(broken, date)),
                "volume": senti.volume_score(volume_pct),
            })
            if value is not None:
                scores.append(value)
        return pd.Series(scores)

    # ---------------- 风险：双轨模型 ----------------
    def _concentration(self, ctx: Ctx):
        """资金集中度：成交额前 5% 个股占全市场比重，附 60 日历史分位。"""
        dates = calendar.recent_open_dates(ctx.store, 60, end=ctx.trade_date)
        daily = ctx.store.read_dated("daily", dates,
                                     columns=["trade_date", "amount"])
        if daily.empty:
            return None, None

        def _share(values):
            series = pd.to_numeric(values, errors="coerce").dropna()
            total = series.sum()
            if series.empty or total <= 0:
                return None
            k = max(1, int(len(series) * 0.05))
            return float(series.nlargest(k).sum() / total)

        shares = daily.groupby("trade_date")["amount"].apply(_share).dropna()
        if shares.empty:
            return None, None
        today = shares.get(ctx.trade_date, shares.iloc[-1])
        return float(today), percentile_of(shares, float(today))

    def _industry_share(self, ctx: Ctx, daily: pd.DataFrame):
        """行业集中度：前三大行业成交额占比。"""
        if daily is None or daily.empty:
            return None, None
        meta = uni.stock_meta(ctx.store)
        if meta.empty or "industry" not in meta.columns:
            return None, None
        frame = daily[["ts_code", "amount"]].copy()
        frame["industry"] = meta["industry"].reindex(
            frame["ts_code"].astype(str)).values
        frame = frame.dropna(subset=["industry"])
        if frame.empty:
            return None, None
        amounts = pd.to_numeric(frame["amount"], errors="coerce").dropna()
        total = amounts.sum()
        if total <= 0:
            return None, None
        grouped = frame.groupby("industry")["amount"].sum()
        top3 = grouped.nlargest(3).sum()
        share = float(top3 / total)
        return share, percentile_of(grouped, grouped.nlargest(3).min())

    def _risk(self, ctx: Ctx, breadth: dict, limit_today: dict, index_row: dict,
              concentration, conc_pct, industry, ind_pct) -> dict:
        up = int(breadth.get("up") or 0)
        down = int(breadth.get("down") or 0)
        total = up + down
        up_ratio = (up / total) if total else None

        result = risk_kit.evaluate(
            concentration=concentration,
            industry=industry,
            severe_ratio=breadth.get("panic_ratio"),
            up_ratio=up_ratio,
            broken_ratio=limit_today.get("broken_ratio"),
            limit_down=limit_today.get("down"),
            bull_trend=self._bull_trend(ctx),
            concentration_percentile=conc_pct,
            industry_percentile=ind_pct,
        )
        payload = result.as_dict()
        payload["score"] = payload["total"]     # 兼容旧渲染字段
        return payload

    def _bull_trend(self, ctx: Ctx) -> bool:
        """主升趋势：收盘 > MA20 且 MA20 > MA60。"""
        dates = calendar.recent_open_dates(ctx.store, 70, end=ctx.trade_date)
        close = self._index_close(ctx, dates)
        if len(close) < 60:
            return False
        last = float(close.iloc[-1])
        ma20 = float(close.tail(20).mean())
        ma60 = float(close.tail(60).mean())
        return last > ma20 > ma60

    # ---------------- 关键点位 ----------------
    def _levels(self, ctx: Ctx, index_row: dict) -> dict:
        dates = calendar.recent_open_dates(ctx.store, HISTORY_DAYS, end=ctx.trade_date)
        idx = ctx.store.read_dated("index_daily", dates)
        if idx.empty:
            return {}
        main = idx[idx["ts_code"] == MAIN_INDEX].sort_values("trade_date").reset_index(drop=True)
        if len(main) < 30:
            return {}
        close = pd.to_numeric(main["close"], errors="coerce")
        high = pd.to_numeric(main["high"], errors="coerce")
        low = pd.to_numeric(main["low"], errors="coerce")
        current = float(close.iloc[-1])

        candidates: List[tuple] = []      # (价格, 权重, 来源)

        # 1) 均线：周期越长权重越高
        for n, weight in ((5, 1), (10, 1), (20, 1), (60, 2), (120, 2), (250, 3)):
            if len(close) >= n:
                candidates.append((float(close.tail(n).mean()), weight, f"MA{n}"))

        # 2) 摆动高低点
        for n, weight in ((20, 1), (60, 2)):
            if len(high) >= n:
                candidates.append((float(high.tail(n).max()), weight, f"{n}日高点"))
                candidates.append((float(low.tail(n).min()), weight, f"{n}日低点"))

        # 3) 自算筹码峰（cyq_chips 需 10000 积分，这里按成交量分箱自算）
        for peak in technical.chip_peaks(main, window=250, bin_pct=0.005, top_n=3):
            candidates.append((float(peak["mid"]), 3, "筹码峰"))

        # 4) 未回补缺口
        for price in self._unfilled_gaps(main):
            candidates.append((price, 2, "缺口"))

        # 5) 整数关口
        step = 100 if current > 3000 else 50
        candidates.append((float(round(current / step) * step), 1, "整数关口"))

        clusters = self._cluster(candidates, tolerance=0.005)
        above = sorted([c for c in clusters if c["price"] > current],
                       key=lambda c: c["price"])
        below = sorted([c for c in clusters if c["price"] <= current],
                       key=lambda c: current - c["price"])

        def pack(c):
            if not c:
                return None
            return {
                "price": round(c["price"], 2),
                "distance_pct": round((c["price"] - current) / current * 100, 2),
                "touches": c["touches"],
                "weight": round(c["weight"], 1),
                "sources": c["sources"][:3],
            }

        # 承接位 = 最近的下方簇（第一道防线，看的是反应速度）
        support = below[0] if below else None

        # 破位位 = 承接位下方权重最高的簇（最后一道防线，跌破即趋势破坏）。
        # 必须严格低于承接位，否则"破位"会排到"承接"上方，语义是错的。
        rest = [c for c in below if support and c["price"] < support["price"]]
        breakdown = max(rest, key=lambda c: c["weight"]) if rest else None

        return {
            "current": round(current, 2),
            "breakout": pack(above[0] if above else None),
            "support": pack(support),
            "breakdown": pack(breakdown),
            "note": "均线 / 摆动高低点 / 自算筹码峰 / 未回补缺口 / 整数关口，"
                    "按类型加权聚类；承接位取最近、破位位取下方最强；历史验证样本量待补",
        }

    @staticmethod
    def _unfilled_gaps(df: pd.DataFrame, lookback: int = 120) -> List[float]:
        """未回补缺口，返回缺口边缘价格（向上取前一日高点，向下取前一日低点）。"""
        part = df.tail(lookback + 1).reset_index(drop=True)
        if len(part) < 3:
            return []
        high = pd.to_numeric(part["high"], errors="coerce")
        low = pd.to_numeric(part["low"], errors="coerce")
        gaps: List[float] = []
        for i in range(1, len(part)):
            prev_high, prev_low = high.iloc[i - 1], low.iloc[i - 1]
            cur_high, cur_low = high.iloc[i], low.iloc[i]
            if pd.isna(prev_high) or pd.isna(cur_low):
                continue
            if cur_low > prev_high and low.iloc[i:].min() > prev_high:
                gaps.append(float(prev_high))
            elif cur_high < prev_low and high.iloc[i:].max() < prev_low:
                gaps.append(float(prev_low))
        return gaps[-3:]

    @staticmethod
    def _cluster(points: List[tuple], tolerance: float = 0.005) -> List[dict]:
        """按价格聚类，簇内按权重加权平均，并记录来源与命中数。"""
        if not points:
            return []
        ordered = sorted(points, key=lambda p: p[0])
        groups: List[List[tuple]] = []
        for item in ordered:
            price = item[0]
            if groups and abs(price - groups[-1][-1][0]) / max(groups[-1][-1][0], 1e-9) <= tolerance:
                groups[-1].append(item)
            else:
                groups.append([item])
        result = []
        for group in groups:
            weight = sum(g[1] for g in group) or 1.0
            price = sum(g[0] * g[1] for g in group) / weight
            result.append({
                "price": price,
                "touches": len(group),
                "weight": sum(g[1] for g in group),
                "sources": [g[2] for g in group],
            })
        return result

    # ---------------- 文案 ----------------
    def summarize(self, payload: dict, ctx: Ctx) -> str:
        c = payload.get("context") or {}
        regime = c.get("regime") or {}
        risk = c.get("risk") or {}
        breadth = c.get("breadth") or {}
        index_row = c.get("index") or {}
        parts = []
        if index_row:
            parts.append(f"上证 {index_row.get('close', 0):.2f}（{index_row.get('pct_chg', 0):+.2f}%）")
        if regime:
            parts.append(f"情绪 {regime.get('emoji','')}{regime.get('label','')}")
        if breadth:
            parts.append(f"涨跌 {breadth.get('up',0)}/{breadth.get('down',0)}")
        if risk:
            parts.append(f"风险 {risk.get('level','')}({risk.get('score',0)})")
        return " · ".join(parts) if parts else "数据不足"


MODULE = OverviewModule
