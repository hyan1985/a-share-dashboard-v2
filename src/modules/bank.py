"""银行专区

为什么单独一档：银行是高负债经营，拿普通企业的尺子量会全军覆没。

以华夏银行为例（2026-10-27 实际数据）：
    PE 4.05、股息率 6.58%   都很好
    ROE 7.18%               卡在「红利股」模块的 ROE>15% 门槛外
    资产负债率 91%          被豁免了，但 ROE 那道坎过不去

而 A 股银行 ROE 普遍在 7~12%，15% 是优秀制造业的标准，
不是银行的标准。所以这一档单独用银行自己的尺子：

    股息率 > 4%        （银行普遍高，门槛要抬）
    PE < 8             （银行 PE 常年 4~7）
    ROE > 8%           （不是越高越好，高 ROE 往往意味着高杠杆扩张）
    派息率 20%~70%     （银行要留利润补充资本，30% 的下限对它们偏高）
    连续分红 ≥ 5 年

不看负债率（银行的负债主要是存款，不是有息负债）。
不看收现比（银行没有「销售商品收到的现金」这个概念）。

仍然用 MA120 定位买卖区，和红利模块一致。
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..context import Ctx
from ..kit import dividend as div_kit
from ..kit import financials as fin_kit
from ..kit import returns as ret
from ..kit import universe as uni
from ..data import calendar
from .base import STATUS_OK, STATUS_STALE, BaseModule, Candidate
from .dividend import classify_zone

DEFAULTS = {
    "yield_min": 0.04,
    "pe_max": 8.0,
    "roe_min": 0.08,
    "payout_min": 0.0,
    "payout_max": 1.00,
    "dividend_years_min": 5,
    "ma_long": 120,
    "buy_ratio": 0.88,
    "sell_ratio": 1.12,
}

MA_WINDOW = 130
BANK_KEYWORDS = ("银行",)


def load_valuation_config():
    """每家银行的目标 PE 和目标股息率（来自 bank-scan 的配置）。

    不同银行类型给的估值锚不一样：国有大行按 5.5% 股息率、6.67 倍 PE，
    优秀城商行放宽到 7.67 倍。这不是我定的，是「大佬刘双估值战法」的原文参数。
    """
    import csv
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "config", "bank_valuation.csv")
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = str(row.get("ts_code", "")).strip()
                if not code:
                    continue
                # 配置里是 601398，数据里是 601398.SH
                out[code] = {
                    "type": row.get("type", ""),
                    "target_pe": float(row.get("target_pe") or 0),
                    "target_dv": float(row.get("target_div_yield") or 0),
                }
    except FileNotFoundError:
        return {}
    return out



# ---------------- 综合评分 ----------------
# 两套逻辑本身是互斥的：
#   双估值说「按目标 PE/股息率算，现价便宜」——不看股价走到哪了
#   MA120/位置说「相对自己过去一年，现在在高位」——不看绝对估值
# 护盘行情下两者会同时成立，直接并排展示等于把矛盾丢给用户。
#
# 这里把它们融成一个分：
#   估值分 = 现价低于双估值买点的幅度（越低于买点越高）
#   位置分 = 一年区间的低位程度（越低位越高）
#   趋势分 = 近 3 月涨幅的反向（涨得越多越低）
# 权重是我的选择，没有文献依据，写在这里是为了将来可以调：
# 估值占 45%（双估值是方法主体），位置占 40%（用户实测高位误报最致命），
# 趋势占 15%（只做辅助，避免把追涨的标的打得太低）。
WEIGHT_VALUE = 0.45
WEIGHT_POSITION = 0.40
WEIGHT_TREND = 0.15

# 现价低于买点 25% 就算估值满分
VALUE_FULL_AT = 0.25
# 近 3 月涨 30% 就算趋势零分
TREND_ZERO_AT = 30.0


def score_components(buy_price, price, yr_position, chg_3m):
    """返回 (综合分, 估值分, 位置分, 趋势分)，缺数据的分项按中性 50 计。"""
    value = 50.0
    if buy_price and price and buy_price > 0:
        discount = buy_price / price - 1.0          # 现价比买点低多少
        value = max(0.0, min(1.0, discount / VALUE_FULL_AT)) * 100.0
    position = 50.0
    if yr_position is not None:
        position = max(0.0, min(1.0, 1.0 - yr_position / 100.0)) * 100.0
    trend = 50.0
    if chg_3m is not None:
        trend = max(0.0, min(1.0, 1.0 - chg_3m / TREND_ZERO_AT)) * 100.0
    total = (value * WEIGHT_VALUE
             + position * WEIGHT_POSITION
             + trend * WEIGHT_TREND)
    return round(total, 1), round(value, 1), round(position, 1), round(trend, 1)


class BankModule(BaseModule):
    id = "bank"
    title = "银行专区"
    kind = "screener"
    requires = ["daily", "daily_basic", fin_kit.FIN_TABLE]

    def compute(self, ctx: Ctx) -> dict:
        cfg = dict(DEFAULTS)
        cfg.update(self.thresholds.get("bank") or {})
        cfg.update(self.params or {})

        snap = ctx.on("daily_basic")
        if snap.empty:
            return {"status": STATUS_STALE, "summary": "缺少日行情快照"}

        meta = uni.stock_meta(ctx.store)
        if meta.empty:
            return {"status": STATUS_STALE, "summary": "缺少股票列表"}

        # 只留银行
        # 先判断 c 在不在 meta 里——顺序反了会 KeyError，
        # 因为 daily_basic 里有而 stock_basic 里没有的代码（新股/退市）确实存在。
        bank_codes = [c for c in snap.ts_code
                      if c in meta.index
                      and any(k in str(meta.loc[c].get("industry", ""))
                              for k in BANK_KEYWORDS)]
        if not bank_codes:
            return {"status": STATUS_OK, "summary": "股票列表里没有银行分类"}

        # 多拉一年：要算「近一年价格位置」和「近 3 月涨幅」，
        # 只看 130 天算不出位置——这正是双估值漏掉的东西。
        dates = calendar.recent_open_dates(ctx.store, 260, end=ctx.trade_date)
        px = ret.total_return_frame(ctx.store, bank_codes, dates) if dates else pd.DataFrame()

        raw_close = {}
        try:
            d1 = ctx.on("daily")
            if not d1.empty and "close" in d1.columns:
                raw_close = dict(zip(d1["ts_code"],
                                     pd.to_numeric(d1["close"], errors="coerce")))
        except Exception:
            pass

        periods = fin_kit.usable_periods(ctx.store, need=3, as_of=ctx.trade_date)
        fin = fin_kit.load(ctx.store, periods)
        latest = fin_kit.latest_row(fin) if not fin.empty else pd.DataFrame()
        roe_wide = fin_kit.pivot(fin, "roe") if not fin.empty else pd.DataFrame()

        try:
            div_tbl = ctx.store.read_dated("dividend")
        except Exception:
            div_tbl = None

        val_cfg = load_valuation_config()

        # 每只银行「本年度」的每股分红（中期 + 年度合计），双估值要用
        dividends_by_code = {}
        if div_tbl is not None and not div_tbl.empty:
            for c in bank_codes:
                if c not in latest.index:
                    continue
                fy = int(str(latest.loc[c].get("end_date") or "")[:4] or 0)
                if fy:
                    d = div_kit.dps_for_year(div_tbl[div_tbl.ts_code == c], fy)
                    if d:
                        dividends_by_code[c] = float(d)

        rows: List[Candidate] = []
        stats = {"banks": len(bank_codes), "pe": 0, "yield": 0, "roe": 0, "passed": 0}

        for code in bank_codes:
            s = snap[snap.ts_code == code].iloc[0]
            pe = pd.to_numeric(s.get("pe_ttm"), errors="coerce")
            dv = pd.to_numeric(s.get("dv_ttm"), errors="coerce")
            if pd.isna(pe) or pd.isna(dv) or pe <= 0 or pe > cfg["pe_max"]:
                continue
            stats["pe"] += 1
            if dv / 100.0 < cfg["yield_min"]:
                continue
            stats["yield"] += 1

            if code not in latest.index:
                continue
            row = latest.loc[code]
            roe = fin_kit.norm_ratio(row.get("roe"))
            if roe is None or roe < cfg["roe_min"]:
                continue
            stats["roe"] += 1

            streak, payout = 0, None
            if div_tbl is not None and not div_tbl.empty:
                sub = div_tbl[div_tbl.ts_code == code]
                if not sub.empty:
                    streak = int(div_kit.streak(sub) or 0)
                    # 派息率必须同年对齐（见 kit/dividend.py 的说明）
                    eps = pd.to_numeric(row.get("eps"), errors="coerce")
                    fin_year = int(str(row.get("end_date") or "")[:4] or 0)
                    dps = (div_kit.dps_for_year(sub, fin_year)
                           if fin_year else None)
                    if dps and eps and eps > 0:
                        payout = float(dps) / float(eps)

            if streak and streak < cfg["dividend_years_min"]:
                continue
            # 派息率只挡 >100%（文章原文的「警惕线」：在吃老本不可持续）。
            # 70~100% 不加硬筛——高派息率是「现金奶牛」的特征，正是要找的。
            # 30~70% 之外给提示，不改判。
            if payout is not None and payout > cfg["payout_max"]:
                continue

            # ---- 双估值法（来自 bank-scan 仓库）----
            #   股息率估值价 = 每股分红 ÷ 目标股息率
            #   PE 估值价   = 当年 EPS × 目标 PE
            #   基准价      = 两者取低
            #   买入价      = 基准价 + 即将派发的分红（除权前买能拿到这笔）
            vcfg = val_cfg.get(code.split(".")[0]) or {}
            dps_year = dividends_by_code.get(code)
            base_price = buy_price = pe_price = div_price = None
            gap_pct = None
            if vcfg and dps_year:
                if vcfg["target_dv"] > 0:
                    div_price = dps_year / (vcfg["target_dv"] / 100.0)
                eps_now = pd.to_numeric(row.get("eps"), errors="coerce")
                if vcfg["target_pe"] > 0 and eps_now and eps_now > 0:
                    pe_price = float(eps_now) * vcfg["target_pe"]
                cand = [x for x in (pe_price, div_price) if x]
                if cand:
                    base_price = min(cand)
                    buy_price = base_price + dps_year

            zone, badge = "无数据", "hold"
            price = price_adj = ma_display = None
            deviation = yr_position = chg_3m = None
            if code in px.columns:
                series = px[code].dropna()
                if len(series) >= cfg["ma_long"]:
                    price_adj = float(series.iloc[-1])
                    ma = float(series.tail(cfg["ma_long"]).mean())
                    if ma > 0:
                        ratio = price_adj / ma
                        rp = raw_close.get(code)
                        price = float(rp) if rp and rp == rp else price_adj
                        ma_display = price / ratio
                        deviation = (ratio - 1) * 100
                        zone, badge = classify_zone(ratio - 1, cfg["buy_ratio"],
                                                    cfg["sell_ratio"])
                        if buy_price and price:
                            gap_pct = (price / buy_price - 1) * 100
                        # 位置指标：双估值只回答「相对目标价贵不贵」，
                        # 不回答「相对自己这一年的走势在哪」。护盘行情里
                        # 银行能从低位涨 20%，双估值仍然是「买点」——
                        # 这就是它最危险的地方，所以位置要单独算、单独展示。
                        tail = series.tail(250)
                        lo, hi = float(tail.min()), float(tail.max())
                        if hi > lo:
                            yr_position = (price_adj - lo) / (hi - lo) * 100
                        if len(series) > 63:
                            chg_3m = (price_adj / float(series.iloc[-63]) - 1) * 100

            total_score, v_s, p_s, t_s = score_components(
                buy_price, price, yr_position, chg_3m)

            stats["passed"] += 1
            rows.append(Candidate(
                code=code,
                name=str(meta.loc[code].get("name", "")),
                score=total_score,
                industry=str(meta.loc[code].get("industry", "")),
                metrics={
                    "pe": round(float(pe), 2),
                    "dv": round(float(dv), 2),
                    "roe": round(roe * 100, 1),
                    "dividend_years": streak,
                    "payout": round(payout * 100, 1) if payout else None,
                    "price": round(price, 2) if price else None,
                    "ma120": round(ma_display, 2) if ma_display else None,
                    "deviation": round(deviation, 1) if deviation is not None else None,
                    "zone": zone,
                    "buy_price": round(buy_price, 2) if buy_price else None,
                    "gap_pct": round(gap_pct, 1) if gap_pct is not None else None,
                    "yr_position": round(yr_position, 0) if yr_position is not None else None,
                    "chg_3m": round(chg_3m, 1) if chg_3m is not None else None,
                    # 只看「近一年位置」。原来还要求 MA120 偏离 ≥8%，
                    # 但护盘行情里股价是台阶式抬升的，偏离均线并不大，
                    # 位置却已经到顶——那个附加条件会让高位标的漏判。
                    "overheated": bool(yr_position is not None and yr_position >= 85),
                    "score_value": v_s,
                    "score_position": p_s,
                    "score_trend": t_s,
                    "bank_type": vcfg.get("type", ""),
                },
                reasons=[f"PE {pe:.1f}", f"股息率 {dv:.2f}%", f"ROE {roe*100:.1f}%"],
                risks=[f"派息率 {payout*100:.0f}% 偏低" if payout and payout < 0.30 else ""]
                      if payout and payout < 0.30 else [],
                badges=([badge] + (
                    [f"低于买点 {abs(gap_pct):.1f}%"] if gap_pct is not None and gap_pct <= 0
                    else [f"还需跌 {gap_pct:.1f}%"] if gap_pct is not None
                    else [])
                    + ([f"⚠️一年高位 {yr_position:.0f}%"] if yr_position is not None
                       and yr_position >= 80 else [])),
            ))

        rows.sort(key=lambda r: r.score, reverse=True)
        cap = int(cfg.get("max_candidates", 20))
        rows = rows[:cap]
        buy = sum(1 for r in rows if r.metrics["zone"] == "买入区")
        # 双估值口径：现价已低于「双估值买入价」的只数
        in_range = sum(1 for r in rows
                       if r.metrics.get("gap_pct") is not None
                       and r.metrics["gap_pct"] <= 0
                       and not r.metrics.get("overheated"))
        hot = sum(1 for r in rows if r.metrics.get("overheated"))

        return {
            "status": STATUS_OK,
            "summary": (f"银行 {stats['banks']} 只 · 通过 {stats['passed']} 只 · "
                        f"双估值到位且不在高位 {in_range} 只 · "
                        f"⚠️一年高位 {hot} 只"),
            "candidates": rows,
            "context": {"thresholds": cfg, "funnel": stats},
        }


MODULE = BankModule
