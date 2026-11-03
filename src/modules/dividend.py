"""红利股 · 渣男手册法

来源：知乎《红利股投资—渣男手册法》（不吃韭菜的鸭鸭，2026-02）。
作者方法的核心不是"高股息就买"，而是三道关卡：

  一、用财报排雷，筛掉"因为基本面差所以股息率高"的陷阱
  二、用 MA120 当锚，只在这条线下方 12% 买、上方 12% 卖
  三、卖出后不回头，去找下一个

本模块只做「筛 + 定位」，不做买卖建议。

筛选门槛（全部满足才算候选）：
  PE(TTM) < 20         >25 不考虑
  股息率   > 3%        >5% 优秀
  毛利率   > 30%       证明有定价权
  ROE      > 15%       连续多年
  经营现金流 > 净利润   利润是真金白银
  收现比   > 1.0       销售商品收到的现金 ÷ 营业收入
  资产负债率 < 65%     非金融企业
  派息率   30%~100%    >100% 在吃老本

定位（相对 120 日均线）：
  ≤ MA120 × 0.88  → 买入区
  ≥ MA120 × 1.12  → 卖出区
  之间            → 观望
  分红复投位置：价格 < MA60
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..context import Ctx
from ..kit import dividend as div_kit
from ..kit import financials as fin_kit
from ..kit import universe as uni
from .base import STATUS_OK, STATUS_STALE, BaseModule, Candidate

DEFAULTS = {
    "pe_max": 20.0,
    "yield_min": 0.03,
    "gross_margin_min": 0.30,
    "roe_min": 0.15,
    "debt_max": 0.65,
    "payout_min": 0.30,
    "payout_max": 1.00,
    "cash_ratio_min": 1.0,
    "ma_long": 120,
    "ma_short": 60,
    "buy_ratio": 0.88,
    "sell_ratio": 1.12,
}

MA_WINDOW = 130          # 用 130 天算 MA120，留点余量



def classify_zone(deviation: float, buy_ratio: float = 0.88,
                  sell_ratio: float = 1.12):
    """按价格相对 MA120 的偏离度判定区间。

    deviation = price / MA120 - 1
    ≤ buy_ratio - 1（默认 -12%）→ 买入区
    ≥ sell_ratio - 1（默认 +12%）→ 卖出区
    """
    # 容差是必须的：1.12 - 1 在浮点里等于 0.1200000000000001，
    # 正好 -12% 或 +12% 的股票会卡在边界外，判定和显示对不上。
    eps = 1e-9
    if deviation <= buy_ratio - 1 + eps:
        return "买入区", "buy"
    if deviation >= sell_ratio - 1 - eps:
        return "卖出区", "sell"
    return "观望", "hold"


class DividendModule(BaseModule):
    id = "dividend"
    title = "红利股 · 渣男手册"
    kind = "screener"
    requires = ["daily", "daily_basic", fin_kit.FIN_TABLE, "dividend"]

    # ---------------- 数据准备 ----------------
    def _prices(self, ctx: Ctx, codes) -> pd.DataFrame:
        """近 MA_WINDOW 个交易日的复权收盘价，返回 index=日期 的宽表。"""
        from ..data import calendar
        from ..kit import returns as ret

        dates = calendar.recent_open_dates(ctx.store, MA_WINDOW + 10,
                                           end=ctx.trade_date)
        if len(dates) < 80:
            return pd.DataFrame()
        px = ret.total_return_frame(ctx.store, codes, dates)
        return px

    def _snapshot(self, ctx: Ctx) -> pd.DataFrame:
        f = ctx.on("daily_basic")
        if f.empty:
            return pd.DataFrame()
        cols = [c for c in ("ts_code", "pe_ttm", "dv_ttm", "total_mv") if c in f.columns]
        out = f[cols].copy()
        for c in ("pe_ttm", "dv_ttm", "total_mv"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _cash_ratio(self, ctx: Ctx) -> Dict[str, float]:
        """收现比 = 销售商品收到的现金 ÷ 营业收入。

        需要 cashflow 与 income 两张表。缺表时返回空 dict，
        候选里会标注「收现比未校验」，而不是默认通过。
        """
        # 这两张表按报告期存（20251231 这样的目录），不是单文件，
        # 也不是按交易日。取最近一期即可。
        def _latest(table: str):
            dates = ctx.store.list_dates(table)
            if not dates:
                return None
            return ctx.store.read_dated(table, [dates[-1]])

        try:
            cf = _latest("cashflow")
            inc = _latest("income")
        except Exception:
            return {}
        if cf is None or inc is None or cf.empty or inc.empty:
            return {}
        cf = cf.sort_values("end_date").drop_duplicates("ts_code", keep="last")
        inc = inc.sort_values("end_date").drop_duplicates("ts_code", keep="last")
        m = cf[["ts_code", "c_fr_sale_sg"]].merge(
            inc[["ts_code", "revenue"]], on="ts_code", how="inner")
        m["c_fr_sale_sg"] = pd.to_numeric(m.c_fr_sale_sg, errors="coerce")
        m["revenue"] = pd.to_numeric(m.revenue, errors="coerce")
        m = m[(m.revenue > 0) & m.c_fr_sale_sg.notna()]
        return dict(zip(m.ts_code, m.c_fr_sale_sg / m.revenue))

    # ---------------- 主流程 ----------------
    def compute(self, ctx: Ctx) -> dict:
        # 阈值统一放 config/thresholds.yaml，代码里只留兜底默认值
        cfg = dict(DEFAULTS)
        cfg.update(self.thresholds.get("dividend") or {})
        cfg.update(self.params or {})

        snap = self._snapshot(ctx)
        px = self._prices(ctx, snap.ts_code.tolist())
        if snap.empty or px.empty:
            return {"status": STATUS_STALE,
                    "summary": "缺少行情或快照数据，请先 make fetch"}

        # 当日不复权收盘价，只用于显示（用户拿行情软件对照）
        raw_close = {}
        try:
            d1 = ctx.on("daily")
            if not d1.empty and "close" in d1.columns:
                raw_close = dict(zip(d1["ts_code"],
                                     pd.to_numeric(d1["close"], errors="coerce")))
        except Exception:
            raw_close = {}

        meta = uni.stock_meta(ctx.store)
        periods = fin_kit.usable_periods(ctx.store, need=3, as_of=ctx.trade_date)
        fin = fin_kit.load(ctx.store, periods)
        if fin.empty:
            return {"status": STATUS_STALE, "summary": "缺少财务数据"}

        latest = fin_kit.latest_row(fin)
        roe_wide = fin_kit.pivot(fin, "roe")
        # 文章要求「连续 5 年 ROE > 15%」，不是只看最新一期。
        # 手头只加载了 3 期年报，所以按近 3 年连续达标算，并在候选里给出实际年数。
        def _roe_streak(code: str) -> int:
            if code not in roe_wide.index:
                return 0
            vals = [fin_kit.norm_ratio(v) for v in roe_wide.loc[code].dropna().values]
            vals = [v for v in vals if v is not None]
            cnt = 0
            for v in reversed(vals):
                if v >= cfg["roe_min"]:
                    cnt += 1
                else:
                    break
            return cnt
        cash_ratio = self._cash_ratio(ctx)

        # 分红表以 ts_code 为分区键（一只股票一个文件），不是单文件表。
        # 用 read_dated 不传日期 = 读全部已缓存的分红记录。
        try:
            div_tbl = ctx.store.read_dated("dividend")
        except Exception:
            div_tbl = None

        rows: List[Candidate] = []
        stats = {"total": 0, "pe": 0, "yield": 0, "gross": 0, "roe": 0,
                 "debt": 0, "cash": 0, "dividend": 0, "passed": 0}

        ma_long = int(cfg["ma_long"])
        ma_short = int(cfg["ma_short"])
        for code in snap.ts_code:
            stats["total"] += 1
            s = snap[snap.ts_code == code]
            pe = float(s.pe_ttm.iloc[0]) if pd.notna(s.pe_ttm.iloc[0]) else None
            # dv_ttm 是百分数（如 5.71 表示 5.71%），统一换成小数再比
            dv = (float(s.dv_ttm.iloc[0]) / 100.0
                  if pd.notna(s.dv_ttm.iloc[0]) else None)
            if pe is None or dv is None or pe <= 0 or pe > cfg["pe_max"]:
                continue
            stats["pe"] += 1
            if dv < cfg["yield_min"]:
                continue
            stats["yield"] += 1

            if code not in latest.index:
                continue
            row = latest.loc[code].copy()
            # 财务表里没有名称和行业，从 stock_basic 补——不补的话
            # 金融股豁免（银行不看负债率和收现比）就永远不生效。
            if code in meta.index:
                row["name"] = meta.loc[code].get("name", "")
                row["industry"] = meta.loc[code].get("industry", "")
            gm = fin_kit.norm_ratio(row.get("grossprofit_margin"))
            roe = fin_kit.norm_ratio(row.get("roe"))
            debt = fin_kit.norm_ratio(row.get("debt_to_assets"))
            ocfps = fin_kit.norm_ratio(row.get("ocfps"))
            # EPS 是绝对数值（元/股），不能过 norm_ratio——
            # 那个函数会把 >1 的值当百分数除以 100，招商银行 5.6 元/股
            # 会变成 0.056，派息率算出来 1760%。这个坑很隐蔽。
            eps = pd.to_numeric(row.get("eps"), errors="coerce")

            if gm is not None and gm < cfg["gross_margin_min"]:
                continue
            stats["gross"] += 1
            if roe is None or roe < cfg["roe_min"]:
                continue
            roe_streak = _roe_streak(code)
            if roe_streak < 2:
                continue
            stats["roe"] += 1
            is_fin = uni.is_financial(row.get("industry"))
            if not is_fin and debt is not None and debt > cfg["debt_max"]:
                continue
            stats["debt"] += 1
            if ocfps is not None and eps and eps > 0 and ocfps < eps:
                continue
            stats["cash"] += 1

            # 收现比对银行/保险不适用（作者本人也在评论区强调了这点）
            cr = None if is_fin else cash_ratio.get(code)
            if cr is not None and cr < cfg["cash_ratio_min"]:
                continue

            # 分红：连续年数与派息率
            streak = 0
            payout = None
            if div_tbl is not None and not div_tbl.empty:
                sub = div_tbl[div_tbl.ts_code == code]
                if not sub.empty:
                    streak = int(div_kit.streak(sub) or 0)
                    info = div_kit.payout_summary(sub, ctx.trade_date) or {}
                    # payout_summary 不返回派息率，自己算：每股分红 ÷ EPS
                    # 派息率必须同年对齐：EPS 用哪一年的年报，
                    # 每股分红就取哪一年，否则两边错一年，数会差一倍。
                    fin_year = int(str(row.get("end_date") or "")[:4] or 0)
                    dps_latest = (div_kit.dps_for_year(sub, fin_year)
                                  if fin_year else None)
                    if dps_latest and eps and eps > 0:
                        payout = float(dps_latest) / float(eps)
            # 派息率：>100% 说明在吃老本，<20% 说明分红意愿弱
            if payout is not None and not (cfg["payout_min"] <= payout <= cfg["payout_max"]):
                continue
            stats["dividend"] += 1

            # 均线定位
            if code not in px.columns:
                continue
            series = px[code].dropna()
            if len(series) < ma_long:
                continue
            price_adj = float(series.iloc[-1])
            raw_px = raw_close.get(code)
            # 复权价只用来算均线和偏离；面板上要显示真实收盘价，
            # 否则用户拿自己的行情软件对不上（格力复权价能到 9000 多）。
            price = float(raw_px) if raw_px and raw_px == raw_px else price_adj
            ma = float(series.tail(ma_long).mean())
            ma_s = float(series.tail(ma_short).mean()) if len(series) >= ma_short else None
            if ma <= 0:
                continue
            ratio = price_adj / ma

            # 价格不到半年线一半，多半是复权或上市初期的数据问题，不是"便宜"
            if ratio < 0.5:
                continue
            zone, badge = classify_zone(ratio - 1, cfg["buy_ratio"], cfg["sell_ratio"])

            stats["passed"] += 1
            reasons = [
                f"PE {pe:.1f}",
                f"股息率 {dv*100:.2f}%",
                f"ROE {roe*100:.1f}%（连续 {roe_streak} 年）" if roe else "ROE -",
                f"毛利率 {gm*100:.0f}%" if gm else "毛利率 -",
                f"负债率 {debt*100:.0f}%" if debt else "负债率 -",
            ]
            risks = []
            if cr is None:
                risks.append("收现比未校验（缺 cashflow/income 表）")
            if payout is None:
                risks.append("派息率未校验")
            elif payout > 0.70:
                risks.append(f"派息率 {payout*100:.0f}% 偏高（文章建议 30~70%）")
            if streak and streak < 5:
                risks.append(f"连续分红仅 {streak} 年")

            rows.append(Candidate(
                code=code,
                name=str(row.get("name") or ""),
                score=round((cfg["sell_ratio"] - ratio) * 100, 2),
                industry=str(row.get("industry") or ""),
                metrics={
                    "pe": round(pe, 2), "dv": round(dv * 100, 2),
                    "roe": round(roe * 100, 1) if roe else None,
                    "gross_margin": round(gm * 100, 1) if gm else None,
                    "debt": round(debt * 100, 1) if debt else None,
                    "cash_ratio": round(cr, 2) if cr else None,
                    "payout": round(payout * 100, 1) if payout else None,
                    "dividend_years": streak,
                    "roe_streak": roe_streak,
                    "is_financial": bool(is_fin),
                    "price": round(price, 2),
                    "price_adj": round(price_adj, 2),
                    # 折算到现价尺度的半年线：用户拿行情软件对照时看得懂
                    "ma120": round(price / ratio, 2) if ratio else None,
                    "ma120_adj": round(ma, 2),
                    "deviation": round((ratio - 1) * 100, 1),
                    "ma60": (round(price / (price_adj / ma_s), 2)
                             if ma_s and price_adj else None),
                    "reinvest_ok": bool(ma_s and price < ma_s),
                    "zone": zone,
                },
                reasons=reasons,
                risks=risks,
                badges=[badge],
            ))

        rows.sort(key=lambda r: r.metrics.get("deviation", 999))
        cap = int(cfg.get("max_candidates", 20))
        rows = rows[:cap]

        buy = sum(1 for r in rows if r.metrics["zone"] == "买入区")
        return {
            "status": STATUS_OK,
            "summary": (f"{ctx.trade_date} · 通过筛选 {stats['passed']} 只，"
                        f"其中买入区 {buy} 只"),
            "candidates": rows,
            "context": {
                "thresholds": cfg,
                "funnel": stats,
                "cash_ratio_available": bool(cash_ratio),
            },
        }


MODULE = DividendModule
