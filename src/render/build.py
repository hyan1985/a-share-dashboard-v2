"""渲染：把模块结果拼成单文件 HTML（数据内联，双击即开）。"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import OUTPUT_DIR, WEB_DIR
from ..modules.base import (ModuleResult, STATUS_EMPTY, STATUS_ERROR,
                            STATUS_STALE)

TITLE = "A股看板"


def _e(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _tone_class(tone: str) -> str:
    return tone if tone in ("hot", "warm", "cold", "flat") else "flat"


# ---------------- 总览 ----------------

def _verdict(regime, sentiment, risk, levels, close):
    """一句话结论：今天什么状态、要不要动手。

    原来页面把 6 个指标、2 个分数、3 个点位平铺出来，
    看完不知道该干什么。这里先给结论，细节放后面。
    """
    risk_score = float((risk or {}).get("score") or 0)
    sent_score = float((sentiment or {}).get("score") or 50)
    label = (regime or {}).get("label", "")
    tone = (regime or {}).get("tone", "flat")

    if risk_score >= 60:
        return "风险偏高，不追高、不加仓", "warn"
    if sent_score <= 25:
        return "情绪冰点，可以留意机会（不是叫你重仓）", "cold"
    if sent_score >= 78:
        return "情绪过热，别追", "warn"
    if tone == "cold":
        return f"{label}，位置偏低，按计划执行", "cold"
    if tone == "warm":
        return f"{label}，情绪回暖，按计划执行", "warm"
    return f"{label}，方向不明，按计划执行", "flat"


def _distance(level, close):
    """点位距现价多少。正数=在上方（还要涨多少才触及）。"""
    try:
        level = float(level)
        close = float(close)
        if close <= 0 or level <= 0:
            return None
        return (level / close - 1) * 100
    except (TypeError, ValueError):
        return None


def _render_overview(r: ModuleResult) -> str:
    c = r.context or {}
    index_row = c.get("index") or {}
    breadth = c.get("breadth") or {}
    limit = c.get("limit") or {}
    regime = c.get("regime") or {}
    risk = c.get("risk") or {}
    levels = c.get("levels") or {}
    amount_ctx = c.get("amount") or {}

    if r.status != "ok" or not c:
        return _state_block(r)

    pct = index_row.get("pct_chg", 0) or 0
    direction = "up" if pct > 0 else ("down" if pct < 0 else "")

    kpis = [
        ("上证指数", f"{index_row.get('close', 0):.2f}", direction),
        ("涨 / 跌", f"{breadth.get('up', 0)}/{breadth.get('down', 0)}", ""),
        ("涨停 / 跌停", f"{limit.get('up', 0)}/{limit.get('down', 0)}", ""),
        ("炸板率", f"{(limit.get('broken_ratio') or 0) * 100:.0f}%", ""),
        ("最高连板", f"{limit.get('ladder', 0)}", ""),
        ("成交额", _fmt_yi_amount(amount_ctx), ""),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="k">{_e(k)}</div>'
        f'<div class="v {cls}">{_e(v)}</div></div>'
        for (k, v, cls) in kpis
    )

    reasons = "".join(f"<li>{_e(x)}</li>" for x in (regime.get("reasons") or []))
    score = float(risk.get("score") or 0)
    signals = risk.get("signals") or []
    sig_html = "".join(f"<li>{_e(x)}</li>" for x in signals)

    levels_html = _render_levels(levels, index_row.get("close"))

    verdict, vtone = _verdict(regime, c.get("sentiment"), risk, levels,
                              index_row.get("close"))
    sent_score = float((c.get("sentiment") or {}).get("score") or 50)
    risk_score = float(risk.get("score") or 0)

    return f"""
<div class="card wide">
  <div class="card-head">
    <div>
      <div class="card-title">{_e(r.title)}</div>
    </div>
    <div class="card-sub elapsed">耗时 {r.elapsed_ms} ms</div>
  </div>

  <div class="verdict {vtone}">
    <div class="verdict-main">{_e(verdict)}</div>
    <div class="verdict-sub">上证 {_e(f"{index_row.get('close', 0):.2f}")}
      <span class="{direction}">{_e(f"{pct:+.2f}%")}</span>
      · 涨跌 {_e(f"{breadth.get('up', 0)}/{breadth.get('down', 0)}")}
      · 成交 {_e(_fmt_yi_amount(amount_ctx))}</div>
  </div>

  <div class="temps">
    <div class="temp">
      <div class="temp-head"><span>市场热度</span><b>{sent_score:.0f}</b><i>/100</i></div>
      <div class="bar"><i style="width:{min(sent_score,100):.0f}%"></i></div>
      <div class="temp-foot">{_e(_score_hint(sent_score, reverse=False))}</div>
    </div>
    <div class="temp">
      <div class="temp-head"><span>大跌风险</span><b>{risk_score:.0f}</b><i>/100</i></div>
      <div class="bar risk"><i style="width:{min(risk_score,100):.0f}%"></i></div>
      <div class="temp-foot">{_e(_score_hint(risk_score, reverse=True))}</div>
    </div>
  </div>

  <div class="kpis compact">{kpi_html}</div>

  <div class="ov-grid">
    <div class="ov-col">
      {_render_sentiment(c.get('sentiment') or {})}
      {_render_risk(risk)}
    </div>
    <div class="ov-col">
      {_render_valuation(c.get('dividend_valuation') or {})}

      <div class="section">
        <h4>量能位置 · 全市场成交额 {_e(f"{int(amount_ctx.get('total_yi') or 0):,}")} 亿</h4>
        {_render_amount(amount_ctx)}
      </div>

      <div class="section">
        <h4>关键点位</h4>
        {levels_html}
      </div>
    </div>
  </div>
</div>
"""


def _score_hint(score, reverse=False):
    """把一个 0~100 的分数翻成一句人话。"""
    if reverse:
        if score >= 60: return "结构拥挤，注意回撤"
        if score >= 35: return "有一定风险"
        return "结构健康"
    if score >= 78: return "过热，别追"
    if score >= 55: return "回暖中"
    if score >= 30: return "偏冷"
    return "冰点"


def _fmt_yi(amount_qian) -> str:
    """Tushare 的 amount 单位是千元，换算成「亿元」展示。"""
    try:
        return f"{float(amount_qian) * 1000 / 1e8:.0f} 亿"
    except (TypeError, ValueError):
        return "-"


def _render_levels(levels: dict, close=None) -> str:
    if not levels:
        return '<div class="state">点位数据不足</div>'
    items = [
        ("突破位", levels.get("breakout"), "resist"),
        ("承接位", levels.get("support"), "support"),
        ("破位", levels.get("breakdown"), "break"),
    ]
    cells = []
    for name, item, cls in items:
        if not item:
            cells.append(f'<div class="lv {cls}"><div class="name">{name}</div>'
                         f'<div class="price">-</div><div class="dist">暂无</div></div>')
            continue
        dist = item.get("distance_pct", 0)
        sources = " · ".join(item.get("sources") or [])
        cells.append(
            f'<div class="lv {cls}"><div class="name">{name}</div>'
            f'<div class="price">{item.get("price", 0):.2f}</div>'
            f'<div class="dist">{dist:+.2f}% · 吻合 {item.get("touches", 0)} 项'
            f' · 权重 {item.get("weight", 0):.0f}</div>'
            f'<div class="dist">{_e(sources)}</div></div>'
        )
    return '<div class="levels">' + "".join(cells) + "</div>"



def _tab_label(module: ModuleResult) -> str:
    """tab 标签：标题 + 候选数（异常状态给一个标记）。"""
    if module.status == "ok":
        return f'{_e(module.title)}<span class="n">{len(module.candidates)}</span>'
    if module.status == "stale":
        return f'{_e(module.title)}<span class="n muted">·</span>'
    if module.status == "error":
        return f'{_e(module.title)}<span class="n bad">!</span>'
    return f'{_e(module.title)}<span class="n muted">0</span>'


def _render_tabs(modules: List[ModuleResult]) -> str:
    """分区切换：总览常驻，其余用 tab，切到整屏宽后表格更易读。"""
    first = modules[0].id
    buttons = []
    panels = []
    for index, module in enumerate(modules):
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="tab{active}" data-tab="{_e(module.id)}" '
            f'role="tab" aria-selected="{"true" if index == 0 else "false"}">'
            f"{_tab_label(module)}</button>")
        panels.append(
            f'<div class="panel{active}" data-panel="{_e(module.id)}" role="tabpanel">'
            f"{_render_screener(module)}</div>")

    return (
        '<section class="tabs" data-tabs data-default="' + _e(first) + '">'
        '<div class="tabbar" role="tablist">' + "".join(buttons) + "</div>"
        + "".join(panels) + "</section>"
    )


def _valuation_color(pct: float) -> str:
    """股息率分位：高=便宜(绿)，低=贵(红)。"""
    if pct >= 70:
        return "var(--good)"
    if pct >= 40:
        return "var(--warm)"
    return "var(--hot)"


def _render_valuation(dv: dict) -> str:
    """红利估值：高股息股票中位股息率的历史分位。"""
    if not dv or dv.get("median_dv") is None:
        return ""
    pct = float(dv.get("percentile") or 0)
    level = dv.get("level", "")
    tone = {"便宜": "股息率处于历史高位，红利资产便宜",
            "偏便宜": "股息率偏高，可以正常投",
            "中性": "股息率居中，正常定投",
            "偏贵": "股息率偏低，红利资产偏贵，不必加码",
            "贵": "股息率处于历史低位，红利资产较贵"}.get(level, "")
    return f"""
      <div class="section">
        <h4>红利估值 · 高股息中位股息率 {dv.get('median_dv'):.2f}%
          （近 10 年 {pct:.0f}% 分位 · {_e(level)}）</h4>
        <div class="bar"><span style="width:{max(2, min(100, pct)):.0f}%;
          background:{_valuation_color(pct)}"></span></div>
        <div class="card-sub" style="margin-top:8px">
          股息率越高越便宜。当前 {tone}。历史区间
          {dv.get('min_dv'):.2f}% ~ {dv.get('max_dv'):.2f}%，样本 {dv.get('samples')} 个月。
        </div>
      </div>"""


def _render_sentiment(ctx: dict) -> str:
    """情绪强度：连续分 + 各分项（分项本身是 0–100，括号里是权重）。"""
    if not ctx or ctx.get("score") is None:
        return ""

    score = float(ctx["score"])
    percentile = ctx.get("percentile")
    components = ctx.get("components") or []

    rows = []
    for comp in components:
        if not comp.get("available"):
            rows.append(
                f'<div class="dim off"><span>{_e(comp.get("label", ""))}</span>'
                f'<i>数据缺失</i></div>')
            continue
        value = float(comp.get("score") or 0)
        weight = float(comp.get("weight") or 0)
        # 显示「拿到多少分 / 满分多少」而不是「0~100 的分数 · 权重%」。
        # 原来写「65 · 20%」，用户读成「65 分，满分 20」，完全对不上。
        got = value * weight / 100.0
        rows.append(
            f'<div class="dim"><span>{_e(comp.get("label", ""))}</span>'
            f'<div class="bar"><span style="width:{min(100, max(0, value)):.0f}%;'
            f'background:{_temp_color(value)}"></span></div>'
            f'<i>{got:.1f}<small>/{weight:.0f}</small></i></div>')

    head = f"{score:.0f}/100"
    if percentile is not None:
        head += f"（近一年 {percentile:.0f}% 分位）"

    total_style = f"width:{min(100, max(0, score)):.0f}%;background:{_temp_color(score)}"
    return f"""
  <div class="section">
    <h4>市场热度 · {head}</h4>
    <p class="hint">越高 = 市场越热。<b>不是越高越好</b>——过热时追高容易亏；
      冰点附近才有便宜筹码。这里只描述温度，不预测方向。</p>
    <div class="bar"><span style="{total_style}"></span></div>
    <div class="dims">{''.join(rows)}</div>
    <div class="card-sub" style="margin-top:8px">{_e(ctx.get("note", ""))}</div>
  </div>"""


def _bar_color(pct: float) -> str:
    """风险语义：低=安全(绿)，高=危险(红)。"""
    if pct < 34:
        return "var(--good)"
    if pct < 67:
        return "var(--warm)"
    return "var(--hot)"


def _temp_color(pct: float) -> str:
    """情绪语义：冷=蓝，温=灰，热=橙，过热=红。与风险配色区分开。"""
    if pct < 25:
        return "var(--cold)"
    if pct < 50:
        return "var(--flat)"
    if pct < 75:
        return "var(--warm)"
    return "var(--hot)"


def _render_risk(risk: dict) -> str:
    """双轨风险：总分 + 结构/破位两轨 + 五个维度明细。"""
    if not risk:
        return '<div class="state">风险数据不足</div>'

    total = float(risk.get("total") or risk.get("score") or 0)
    structure = float(risk.get("structure") or 0)
    breakdown = float(risk.get("breakdown") or 0)
    dims = risk.get("dimensions") or []
    signals = risk.get("signals") or []
    note = risk.get("note", "")

    def track(name, value, maximum):
        pct = 0 if not maximum else max(0, min(100, value / maximum * 100))
        return (f'<div class="track"><div class="t-head"><span>{name}</span>'
                f'<b>{value:.1f}</b><small>/{maximum:.0f}</small></div>'
                f'<div class="bar"><span style="width:{pct:.0f}%;'
                f'background:{_bar_color(pct)}"></span></div></div>')

    tracks = track("结构拥挤", structure, 50) + track("破位风险", breakdown, 50)

    dim_html = []
    for dim in dims:
        if not dim.get("available", True):
            dim_html.append(
                f'<div class="dim off"><span>{_e(dim.get("label", ""))}</span>'
                f'<i>数据缺失</i></div>')
            continue
        maximum = float(dim.get("max_score") or 0)
        score = float(dim.get("score") or 0)
        pct = 0 if not maximum else max(0, min(100, score / maximum * 100))
        dim_html.append(
            f'<div class="dim"><span>{_e(dim.get("label", ""))}</span>'
            f'<div class="bar"><span style="width:{pct:.0f}%;'
            f'background:{_bar_color(pct)}"></span></div>'
            # 和上面的情绪分项统一：「拿到多少 / 满分多少」
            f'<i>{score:.1f}<small>/{maximum:.0f}</small></i></div>')

    sig_html = "".join(f"<li>{_e(x)}</li>" for x in signals)
    total_style = f"width:{min(100, max(0, total)):.0f}%;background:{_bar_color(total)}"
    return f"""
  <div class="section">
    <h4>大跌风险 · {total:.0f}/100</h4>
    <p class="hint">越高 = 越危险。<b>这个才是越高越差</b>——
      结构拥挤（资金挤在少数票）和破位风险（跌破关键位）同时升高时，
      短期回撤概率明显变大。</p>
    <div class="bar"><span style="{total_style}"></span></div>
    <div class="tracks">{tracks}</div>
    <div class="dims">{''.join(dim_html)}</div>
    <ul class="reasons">{sig_html}</ul>
    <div class="card-sub" style="margin-top:8px">{_e(note)}</div>
  </div>"""


def _fmt_yi_amount(amount_ctx: dict) -> str:
    total = amount_ctx.get("total_yi")
    return f"{int(total):,} 亿" if total else "-"


def _pct_tone(value) -> str:
    """分位数着色：越高越热。"""
    if value is None:
        return ""
    if value >= 80:
        return "up"
    if value <= 20:
        return "down"
    return ""


def _render_amount(ctx: dict) -> str:
    """量能位置：分位 + 相对均值 + 历史同量能之后的表现。"""
    if not ctx or not ctx.get("total_yi"):
        return '<div class="state">成交额数据不足</div>'

    pct_1y = ctx.get("pct_1y")
    pct_3y = ctx.get("pct_3y")
    vs_ma20 = ctx.get("vs_ma20")

    # 标签必须跟真实历史长度一致：只有 1 年数据时不能硬写「近三年分位」
    history_days = int(ctx.get("history_days") or 0)
    windows = []
    if pct_1y is not None:
        label = "近一年分位" if history_days >= 200 else f"近 {history_days} 日分位"
        windows.append((label, pct_1y))
    if pct_3y is not None and history_days >= 600:
        windows.append(("近三年分位", pct_3y))

    cards = []
    for name, value in windows:
        cards.append(
            f'<div class="lv"><div class="name">{name}</div>'
            f'<div class="price {_pct_tone(value)}">{value:.0f}%</div>'
            f'<div class="bar" style="margin-top:6px"><span style="width:{max(2, min(100, value)):.0f}%;'
            f'background:{_bar_color(value)}"></span></div>'
            f'</div>'
        )
    if vs_ma20 is not None:
        tone = "up" if vs_ma20 > 0 else "down"
        cards.append(
            f'<div class="lv"><div class="name">相对 20 日均值</div>'
            f'<div class="price {tone}">{vs_ma20:+.1f}%</div>'
            f'<div class="dist">高于/低于近期均量</div></div>'
        )

    forward = ctx.get("forward") or {}
    if forward:
        parts = []
        for horizon in ("5", "10", "20"):
            item = forward.get(horizon)
            if not item:
                continue
            parts.append(
                f'{horizon} 日中位 <b>{item["median"]:+.2f}%</b>'
                f'（胜率 {item["win"]:.0f}%，n={item["samples"]}）'
            )
        note = (f'历史上成交额落在今日 ±15% 的 {ctx.get("similar_days", 0)} 个交易日，'
                f'其后上证表现：' + " · ".join(parts)
                + '。注意样本高度重叠（量能相近的日子往往连续），'
                  'n 是名义样本数，不能当概率用。')
    else:
        note = f'历史上成交额落在今日 ±15% 的 {ctx.get("similar_days", 0)} 个交易日，前瞻数据不足'

    return ('<div class="levels">' + "".join(cards) + "</div>"
            + f'<div class="card-sub" style="margin-top:10px">{note}</div>')


# ---------------- 选股分区 ----------------
def _render_screener(r: ModuleResult) -> str:
    caution = (r.context or {}).get("caution") or ""
    caution_html = f'<div class="caution">⚠ {_e(caution)}</div>' if caution else ""
    head = f"""
  <div class="card-head">
    <div>
      <div class="card-title">{_e(r.title)} <span class="card-sub">{len(r.candidates)} 只</span></div>
      <div class="card-sub">{_e(r.summary)}</div>
      {caution_html}
    </div>
    <div class="card-sub">{_e(r.as_of)}</div>
  </div>"""

    if r.status != "ok":
        return f'<div class="card">{head}{_state_block(r)}</div>'
    if not r.candidates:
        return f'<div class="card">{head}<div class="state">今日无符合条件的标的</div></div>'

    rows = []
    for cand in r.candidates:
        badges = "".join(
            f'<span class="badge {"horse" if b in ("白马",) else ""}">{_e(b)}</span>'
            for b in cand.badges
        )
        metrics = " · ".join(f"{_e(k)} {_e(v)}" for k, v in list(cand.metrics.items())[:3])
        reasons = "<br>".join(_e(x) for x in cand.reasons[:2])
        risks = "；".join(_e(x) for x in cand.risks[:2])
        rows.append(f"""
      <tr>
        <td><div>{_e(cand.name) or "-"}</div><div class="code">{_e(cand.code)}</div></td>
        <td class="score">{cand.score:.1f}</td>
        <td>{badges}<div class="reason">{metrics}</div></td>
        <td><div class="reason">{reasons}</div>{f'<div class="risk">{risks}</div>' if risks else ''}</td>
      </tr>""")

    return f"""
<div class="card">
  {head}
  <p class="legend"><b>↑</b> 涨到近一年高点能赚多少 · <b>↓</b> 跌到半年线会亏多少
    · <b>低于买点</b> 是现价相对双估值买点。</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>标的</th><th>股息率</th><th>关键指标</th><th>理由 / 风险</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
"""


def _state_block(r: ModuleResult) -> str:
    if r.status == STATUS_STALE:
        return f'<div class="state">{_e(r.summary or "数据未就绪")}</div>'
    if r.status == STATUS_ERROR:
        return f'<div class="state error">模块异常：{_e(r.error)}</div>'
    if r.status == STATUS_EMPTY:
        return '<div class="state">今日无符合条件的标的</div>'
    return '<div class="state">暂无数据</div>'


# ---------------- 组装 ----------------

def _data_status(results, trade_date):
    """右上角的数据状态：模块是否都拿到了数据、数据滞后几天。

    这几项原来完全没显示——页面看起来正常，但底下可能是 stale，
    或者数据停在几天前。用户没法从页面上看出来。
    """
    from datetime import date, datetime

    total = len(results)
    ok = sum(1 for r in results if r.status == "ok")
    stale = [r.title for r in results if r.status == "stale"]
    err = [r.title for r in results if r.status == "error"]

    parts = []
    if err:
        parts.append(f"{len(err)} 个出错")
        cls = "bad"
    elif stale:
        parts.append(f"{len(stale)} 个数据不足")
        cls = "warn"
    else:
        parts.append(f"{ok}/{total} 模块正常")
        cls = "good"

    # 数据滞后：交易日距今天几个自然日
    try:
        td = datetime.strptime(str(trade_date), "%Y%m%d").date()
        gap = (date.today() - td).days
        if gap >= 1:
            weekend = gap <= 3
            parts.append(f"数据 {td:%m-%d}" + ("" if weekend else f"，滞后 {gap} 天"))
            if gap > 3:
                cls = "warn"
    except Exception:
        pass

    return " · ".join(parts), cls


def render(results: List[ModuleResult], trade_date: str,
           generated_at: Optional[str] = None) -> str:
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    template = (WEB_DIR / "template.html").read_text(encoding="utf-8")
    style = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    overview = [r for r in results if r.kind == "environment"]
    screeners = [r for r in results if r.kind != "environment"]

    content = "".join(_render_overview(r)
                      for r in overview)
    if screeners:
        content += _render_tabs(screeners)

    status_text, status_cls = _data_status(results, trade_date)

    data = {
        "trade_date": trade_date,
        "generated_at": generated_at,
        "modules": [r.as_dict() for r in results],
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")

    return (
        template
        .replace("{{TITLE}}", TITLE)
        .replace("{{STYLE}}", style)
        .replace("{{CONTENT}}", content)
        .replace("{{TRADE_DATE}}", trade_date)
        .replace("{{STATUS_TEXT}}", status_text)
        .replace("{{STATUS_CLASS}}", status_cls)
        .replace("{{GENERATED_AT}}", generated_at)
        .replace("{{DATA}}", payload)
    )


def build(results: List[ModuleResult], trade_date: str,
          output_dir: Path = None) -> Dict[str, Path]:
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = render(results, trade_date, generated_at)

    index_path = output_dir / "index.html"
    latest_path = output_dir / "latest.html"
    data_path = output_dir / "data.json"

    index_path.write_text(page, encoding="utf-8")
    latest_path.write_text(page, encoding="utf-8")
    data_path.write_text(
        json.dumps(
            {
                "trade_date": trade_date,
                "generated_at": generated_at,
                "modules": [r.as_dict() for r in results],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return {"index": index_path, "latest": latest_path, "data": data_path}
