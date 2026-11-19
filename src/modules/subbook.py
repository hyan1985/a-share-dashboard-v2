"""破净专区：PB < 1 的公司，单独看。

和红利是两套逻辑：
  红利  看的是「能不能持续派现」——现金流、分红年数、派息率
  破净  看的是「股价有没有低于净资产」——PB、ROE、资产质量

破净不等于便宜。A 股长期的破净股里，很大一部分是：
  银行（已有专区）、地产（资产可能减值）、钢铁煤炭（周期底）、
  以及一批 ROE 长期为负的「价值陷阱」——净资产在，但每年在亏。

所以这里只做三件事：筛掉亏损的、标出白马、按 PB 排序。
不预测会不会回归，也不给买卖建议。
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..context import Ctx
from ..kit import dividend as div_kit
from ..kit import financials as fin_kit
from ..kit import universe as uni
from .base import STATUS_OK, STATUS_STALE, BaseModule, Candidate

DEFAULTS = {
    "pb_max": 1.0,        # 破净的定义
    "roe_min": 0.0,       # 亏损的直接排除，破净+亏损是价值陷阱
    "white_roe": 0.12,    # 白马的 ROE 门槛
    "white_years": 5,     # 白马的连续分红年数
    "white_yield": 0.02,  # 白马的股息率下限
}

# ST / 退市风险的不参与
BAD_NAME = ("ST", "退", "*")

# 周期行业：破净高度集中在这里是正常的，不是筛出错了。
CYCLICAL_INDUSTRIES = ("普钢", "特钢", "水泥", "玻璃", "煤炭", "焦炭", "有色",
                       "化工", "建材", "建筑工程", "房地产开发", "航运", "港口")


def _risks(roe, dv, industry):
    """把容易误读的地方直接标在候选上。

    这个专区最容易踩的三个误读：
      1. 破净 = 便宜          → 净资产可能虚高或会减值
      2. ROE 越高越好         → 超高 ROE 往往只是净资产被侵蚀得很小
      3. 怎么全是周期行业     → 破净本来就集中在周期底，不是筛选出错
    """
    out = []
    if roe is not None and roe >= 0.5:
        out.append(f"ROE {roe*100:.0f}% 异常高，多半是净资产很小，不代表质地更好")
    if not dv:
        out.append("股息率 0，不派现")
    if roe is not None and 0 < roe < 0.08:
        out.append("ROE 低于 8%，净资产回报偏弱")
    if any(k in str(industry or "") for k in CYCLICAL_INDUSTRIES):
        out.append("周期行业：破净集中在周期底是常态，不代表被错杀")
    return out


class SubBookModule(BaseModule):
    id = "subbook"
    title = "破净专区"
    kind = "screener"
    requires = ["daily_basic", fin_kit.FIN_TABLE]

    def compute(self, ctx: Ctx) -> dict:
        cfg = dict(DEFAULTS)
        cfg.update(self.thresholds.get("subbook") or {})
        cfg.update(self.params or {})

        snap = ctx.on("daily_basic")
        if snap.empty:
            return {"status": STATUS_STALE, "summary": "缺少日行情快照"}

        meta = uni.stock_meta(ctx.store)
        if meta.empty:
            return {"status": STATUS_STALE, "summary": "缺少股票列表"}

        fin = fin_kit.load(ctx.store, fin_kit.usable_periods(
            ctx.store, need=2, as_of=ctx.trade_date))
        latest = fin_kit.latest_row(fin) if not fin.empty else pd.DataFrame()

        try:
            div_tbl = ctx.store.read_dated("dividend")
        except Exception:
            div_tbl = None

        rows: List[Candidate] = []
        stats = {"total": 0, "broken": 0, "profitable": 0, "white": 0,
                 "bank_excluded": 0}

        for _, s in snap.iterrows():
            code = s["ts_code"]
            stats["total"] += 1
            pb = pd.to_numeric(s.get("pb"), errors="coerce")
            if pd.isna(pb) or pb <= 0 or pb > cfg["pb_max"]:
                continue
            if code not in meta.index:
                continue
            info = meta.loc[code]
            name = str(info.get("name", ""))
            if any(t in name for t in BAD_NAME):
                continue
            # 银行单独有专区，这里排除，避免同一批公司在两个分区重复出现
            if "银行" in str(info.get("industry", "")):
                stats["bank_excluded"] += 1
                continue
            stats["broken"] += 1

            if code not in latest.index:
                continue
            row = latest.loc[code]
            roe = fin_kit.norm_ratio(row.get("roe"))
            if roe is None or roe <= cfg["roe_min"]:
                continue
            stats["profitable"] += 1

            dv = pd.to_numeric(s.get("dv_ttm"), errors="coerce")
            dv = float(dv) if pd.notna(dv) else None

            streak = 0
            if div_tbl is not None and not div_tbl.empty:
                sub = div_tbl[div_tbl.ts_code == code]
                if not sub.empty:
                    streak = int(div_kit.streak(sub) or 0)

            # 白马破净：ROE 够高 + 连续分红够久 + 股息率不为零。
            # 满足的排在前面——它们的净资产更可信（长期在赚钱），
            # 破净更可能是市场情绪而不是资产质量问题。
            white = bool(roe >= cfg["white_roe"]
                         and streak >= cfg["white_years"]
                         and (dv or 0) >= cfg["white_yield"] * 100.0)
            if white:
                stats["white"] += 1

            rows.append(Candidate(
                code=code,
                name=name,
                score=round(float(pb), 3),
                industry=str(info.get("industry", "")),
                metrics={
                    "pb": round(float(pb), 2),
                    "roe": round(roe * 100, 1),
                    "dv": round(dv, 2) if dv is not None else None,
                    "dividend_years": streak,
                    "pe": round(float(pd.to_numeric(s.get("pe_ttm"), errors="coerce")), 1)
                          if pd.notna(s.get("pe_ttm")) else None,
                },
                reasons=[f"PB {pb:.2f}", f"ROE {roe*100:.1f}%",
                         f"连续分红 {streak} 年" if streak else "分红记录缺失"],
                risks=_risks(roe, dv, info.get("industry")),
                badges=(["白马破净"] if white else []),
            ))

        # 白马在前，其次 PB 从低到高
        rows.sort(key=lambda r: (0 if r.badges else 1, r.score))
        # 全部列出，不截断。原来默认 40 只，用户看不到剩下的。
        cap = int(cfg.get("max_candidates", 800))
        rows = rows[:cap]

        return {
            "status": STATUS_OK,
            "summary": (f"{ctx.trade_date} · 破净 {stats['broken']} 只"
                        f"（已排除银行 {stats['bank_excluded']} 只）· "
                        f"其中盈利 {stats['profitable']} 只，"
                        f"白马破净 {stats['white']} 只"),
            "candidates": rows,
            "context": {"thresholds": cfg, "funnel": stats},
        }


MODULE = SubBookModule
