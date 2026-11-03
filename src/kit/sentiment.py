"""市场情绪强度：用客观指标合成 0–100 的连续分。

为什么不用股吧文本：
- 反爬和页面改版会让数据某天突然消失，且雪球需要 cookie，CI 里跑不通
- 文本情绪是"大家怎么说"，客观指标是"钱怎么动"，后者更难造假

与 regime（六档状态）的分工：
- regime 回答"现在是什么状态"（离散）
- 情绪强度回答"这个状态有多强"（连续 0–100 + 近一年分位）

同样是"温和上涨"，强度 45 和 85 是完全不同的市场。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# (id, 名称, 权重)
COMPONENTS = (
    ("breadth", "市场宽度", 20),
    ("profit", "赚钱效应", 20),
    ("ladder", "连板高度", 15),
    ("seal", "封板质量", 15),
    ("volume", "量能水平", 15),
    ("guba", "股吧情绪", 15),
)
WEIGHTS = {key: weight for key, _, weight in COMPONENTS}
LABELS = {key: label for key, label, _ in COMPONENTS}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def breadth_score(up_ratio: Optional[float]) -> Optional[float]:
    """上涨家数占比：20% → 0 分，80% → 100 分。"""
    if up_ratio is None:
        return None
    return _clamp((up_ratio - 0.20) / 0.60 * 100.0)


def profit_score(limit_up: Optional[int], limit_down: Optional[int]) -> Optional[float]:
    """赚钱效应：(涨停 - 跌停) / (涨停 + 跌停)，映射到 0–100。"""
    if limit_up is None:
        return None
    down = limit_down or 0
    total = limit_up + down
    if total <= 0:
        return 50.0
    return _clamp(((limit_up - down) / total + 1.0) / 2.0 * 100.0)


def ladder_score(ladder: Optional[int]) -> Optional[float]:
    """最高连板：2 板 → 0 分，10 板 → 100 分。"""
    if ladder is None:
        return None
    return _clamp((ladder - 2) / 8.0 * 100.0)


def seal_score(broken_ratio: Optional[float]) -> Optional[float]:
    """封板质量：炸板率越低越高分。"""
    if broken_ratio is None:
        return None
    return _clamp((1.0 - broken_ratio) * 100.0)


def volume_score(percentile: Optional[float]) -> Optional[float]:
    if percentile is None:
        return None
    return _clamp(percentile)


def guba_score(sentiment: Optional[float]) -> Optional[float]:
    """股吧情绪 [-1, 1] → 0–100。"""
    if sentiment is None:
        return None
    return _clamp((float(sentiment) + 1.0) / 2.0 * 100.0)


def composite(scores: Dict[str, Optional[float]]) -> Optional[float]:
    """按权重合成；缺失项不参与分母。"""
    available = {k: v for k, v in scores.items() if v is not None and k in WEIGHTS}
    if not available:
        return None
    total_weight = sum(WEIGHTS[k] for k in available)
    if total_weight <= 0:
        return None
    value = sum(available[k] * WEIGHTS[k] for k in available) / total_weight
    return round(value, 1)


def summarize(scores: Dict[str, Optional[float]],
              details: Dict[str, str] = None,
              percentile: Optional[float] = None) -> dict:
    details = details or {}
    total = composite(scores)
    components: List[dict] = []
    for key, label, weight in COMPONENTS:
        components.append({
            "id": key,
            "label": label,
            "weight": weight,
            "score": None if scores.get(key) is None else round(float(scores[key]), 1),
            "detail": details.get(key, ""),
            "available": scores.get(key) is not None,
        })

    tone = "flat"
    if total is not None:
        if total >= 75:
            tone = "hot"
        elif total >= 55:
            tone = "warm"
        elif total <= 25:
            tone = "cold"

    missing = [LABELS[k] for k in LABELS if scores.get(k) is None]
    note = "由涨停、宽度、连板、封板、量能合成，非股吧文本情绪"
    if missing:
        note += f"（缺：{'、'.join(missing)}）"

    return {
        "score": total,
        "tone": tone,
        "percentile": None if percentile is None else round(float(percentile), 1),
        "components": components,
        "note": note,
    }
