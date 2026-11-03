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
    "payout_min": 0.20,
    "payout_max": 0.70,
    "dividend_years_min": 5,
    "ma_long": 120,
    "buy_ratio": 0.88,
    "sell_ratio": 1.12,
}

MA_WINDOW = 130
BANK_KEYWORDS = ("银行",)


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

        dates = calendar.recent_open_dates(ctx.store, MA_WINDOW + 10,
                                           end=ctx.trade_date)
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
            # 派息率：>100% 说明在吃老本，<20% 说明分红意愿弱
            if payout is not None and not (cfg["payout_min"] <= payout <= cfg["payout_max"]):
                continue

            zone, badge = "无数据", "hold"
            price = price_adj = ma_display = None
            deviation = None
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

            stats["passed"] += 1
            rows.append(Candidate(
                code=code,
                name=str(meta.loc[code].get("name", "")),
                score=round(-(deviation or 0), 2),
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
                },
                reasons=[f"PE {pe:.1f}", f"股息率 {dv:.2f}%", f"ROE {roe*100:.1f}%"],
                risks=[f"派息率 {payout*100:.0f}% 偏低" if payout and payout < 0.30 else ""]
                      if payout and payout < 0.30 else [],
                badges=[badge],
            ))

        rows.sort(key=lambda r: r.metrics.get("deviation") or 999)
        cap = int(cfg.get("max_candidates", 20))
        rows = rows[:cap]
        buy = sum(1 for r in rows if r.metrics["zone"] == "买入区")

        return {
            "status": STATUS_OK,
            "summary": (f"银行 {stats['banks']} 只 · 通过 {stats['passed']} 只，"
                        f"买入区 {buy} 只（ROE 门槛按银行口径取 8%）"),
            "candidates": rows,
            "context": {"thresholds": cfg, "funnel": stats},
        }


MODULE = BankModule
