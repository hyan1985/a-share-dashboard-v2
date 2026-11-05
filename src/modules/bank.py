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
# ---- 分工来自回测，不是凑权重 ----
# tools/backtest_bank_windows.py（106 个月 × 42 只银行）的结果是：
#   双估值便宜度  IC 0.088 (t 4.3)  —— 弱
#   一年位置      IC 0.235 (t 8.6)  —— 强
# 弱的那个不该参与排序，但它衡量的是「市场价有没有基本面支撑」，
# 适合当门槛；强的那个衡量「现在买进去的盈亏比」，适合当排序。
#
# 所以本模块分成两步：
#   第一步 基本面门槛：现价不能高于双估值买点太多，否则基本面撑不住这个价
#   第二步 盈亏比排序：过关的里面按位置分排
#
# 位置和趋势权重：位置 0.85、趋势 0.15（趋势只是位置的辅助确认）。
WEIGHT_POSITION = 0.85
WEIGHT_TREND = 0.15

# 现价高于双估值买点的容忍上限。超过就算「基本面撑不住」。
VALUATION_TOLERANCE = 0.30

# 现价低于买点 25% 就算估值满分
VALUE_FULL_AT = 0.25
# 近 3 月涨 30% 就算趋势零分
TREND_ZERO_AT = 30.0


def risk_reward(price, support, target):
    """盈亏比 = 上行空间 ÷ 下行空间。

    support（支撑价）：双估值买点——基本面撑得住的价格，跌破它说明
                       要么基本面变坏，要么市场给了更好的价。
    target（目标价）：近一年高点——过去的阻力位。

    返回 (盈亏比, 上行%, 下行%)。
      上行% = (target - price) / price
      下行% = (price - support) / price
    现价已低于支撑价时下行记为 0（买在支撑下方，理论下行有限），
    盈亏比给 None 表示「无下行风险」，避免除零后显示成 999。
    """
    if not price or price <= 0 or not support or not target:
        return None, None, None
    up = (target - price) / price * 100.0
    down = (price - support) / price * 100.0
    if down <= 0:
        return None, round(up, 1), 0.0
    if up <= 0:
        return 0.0, round(up, 1), round(down, 1)
    return round(up / down, 2), round(up, 1), round(down, 1)


def _rr_label(rr):
    """盈亏比 → 一句人话标签。"""
    if rr is None:
        return "下行有限"
    if rr >= 1.5:
        return "性价比高"
    if rr >= 1.0:
        return "尚可"
    return "空间不足"


def fundamentals_ok(buy_price, price):
    """基本面门槛：现价有没有超出双估值买点太多。

    返回 (是否通过, 溢价百分比)。拿不到买点时按通过处理，
    并在候选里标注「估值未校验」，不默认否决。
    """
    if not buy_price or not price or buy_price <= 0:
        return True, None
    premium = price / buy_price - 1.0
    return premium <= VALUATION_TOLERANCE, premium * 100.0


def score_components(buy_price, price, yr_position, chg_3m):
    """返回 (综合分, 估值分, 位置分, 趋势分)。

    估值分只作为展示，不参与综合分——它衡量基本面，不是盈亏比。
    """
    value = 50.0
    if buy_price and price and buy_price > 0:
        discount = buy_price / price - 1.0
        value = max(0.0, min(1.0, discount / VALUE_FULL_AT)) * 100.0
    # 位置和趋势都是「越高越好」——回测里这两个因子符号为正，
    # 原本按「低位好」设计是反的，会系统性压低强势标的。
    position = 50.0
    if yr_position is not None:
        position = max(0.0, min(1.0, yr_position / 100.0)) * 100.0
    trend = 50.0
    if chg_3m is not None:
        trend = max(0.0, min(1.0, chg_3m / TREND_ZERO_AT)) * 100.0
    total = (position * WEIGHT_POSITION
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

            fund_ok, premium = fundamentals_ok(buy_price, price)
            if not fund_ok:
                stats["rejected_fundamentals"] = stats.get("rejected_fundamentals", 0) + 1
                continue
            # 盈亏比：支撑=双估值买点，目标=近一年高点
            # 支撑价用 MA120 而不是双估值买点。双估值买点几乎总在市价上方
            # （这是它「便宜」的定义），拿来当下行支撑会让所有标的的下行都算成 0，
            # 盈亏比全部退化。MA120 是市场真正守过的位置，跌破才算趋势走坏。
            support = price / (1.0 + (deviation or 0) / 100.0) if deviation is not None else None
            # 目标价必须和现价同尺度：series 是复权价，price 是真实价。
            # 直接取复权高点会和真实价比出 567% 这种荒谬结果。
            # 用比例换算：target_raw = price × (adj_high / adj_price)
            target = None
            if len(series) and price_adj and price_adj > 0:
                target = price * (float(series.tail(250).max()) / price_adj)
            rr, up_pct, down_pct = risk_reward(price, support, target)
            total_score, v_s, p_s, t_s = score_components(
                buy_price, price, yr_position, chg_3m)

            stats["passed"] += 1
            rows.append(Candidate(
                code=code,
                name=str(meta.loc[code].get("name", "")),
                # score 字段用于表格展示，这里放排序键（股息率），
                # 不放综合分——综合分既不参与排序，又和标签口径不一致，
                # 展示出来只会让人困惑（出现过「0 分却标 buy」）。
                score=float(dv),
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
                    "risk_reward": rr,
                    "upside": up_pct,
                    "downside": down_pct,
                    "score_value": v_s,
                    "valuation_premium": round(premium, 1) if premium is not None else None,
                    "score_position": p_s,
                    "score_trend": t_s,
                    "bank_type": vcfg.get("type", ""),
                },
                reasons=[f"PE {pe:.1f}", f"股息率 {dv:.2f}%", f"ROE {roe*100:.1f}%"],
                risks=[f"派息率 {payout*100:.0f}% 偏低" if payout and payout < 0.30 else ""]
                      if payout and payout < 0.30 else [],
                # 标签用盈亏比口径。原来用的是 MA120 的 buy/hold——
                # 那是「价格低于半年线」的意思，和位置分（越高越好）方向相反，
                # 会出现「评分 0 分却标 buy」这种自相矛盾的展示。
                badges=([_rr_label(rr)]
                        + ([f"低于买点 {abs(gap_pct):.1f}%"] if gap_pct is not None and gap_pct <= 0
                           else [f"还需跌 {gap_pct:.1f}%"] if gap_pct is not None
                           else [])),
            ))

        # 排序按「确定的事实」，不按预测。
        # 原因：tools/backtest_bank_rr.py 测过，盈亏比 >1 的 120 日胜率
        # 只比全样本高 0.5 个百分点（55.1% vs 54.6%），分年度在 35%~90%
        # 之间乱跳，被市场环境主导。IC 0.104 看着显著，实际不可交易。
        # 所以排序改成股息率（确定的现金流）优先，ROE 次之。
        # 盈亏比仍然展示，但只作参考。
        rows.sort(key=lambda r: (
            -(r.metrics.get("dv") or 0),
            -(r.metrics.get("roe") or 0),
        ))
        cap = int(cfg.get("max_candidates", 20))
        rows = rows[:cap]
        buy = sum(1 for r in rows if r.metrics["zone"] == "买入区")
        # 双估值口径：现价已低于「双估值买入价」的只数
        rejected = stats.get("rejected_fundamentals", 0)

        return {
            "status": STATUS_OK,
            "summary": (f"银行 {stats['banks']} 只 · 通过 {stats['passed']} 只 · "
                        f"基本面过关 {stats['passed']} 只"
                        + (f"（{rejected} 只因估值撑不住基本面被剔除）" if rejected else "")
                        + "，按股息率排序（不按盈亏比——见 README 的回测结论）"),
            "candidates": rows,
            "context": {"thresholds": cfg, "funnel": stats},
        }


MODULE = BankModule
