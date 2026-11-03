"""大跌风险：双轨模型（结构拥挤 + 破位风险）。

来源：MarketMonitoring 的 `risk_scorer.py`，但做了两处改造：
1. 只保留当前数据层拿得到的维度，缺的维度 **重归一化**，而不是给中位分凑数
   （凑数会让分母虚高、总分被稀释，看起来永远"中性"）。
2. 拆成纯函数，可以单独测试，不依赖 tushare 连接。

轨道定义：
  结构拥挤 0–50：资金越集中、行业越抱团，越容易在高位派发
  破位风险 0–50：恐慌扩散、宽度恶化、炸板跌停潮，是真正该减仓的信号
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

STRUCTURE_MAX = 50.0
BREAKDOWN_MAX = 50.0


@dataclass
class Dimension:
    id: str
    label: str
    score: float
    max_score: float
    value: Optional[float] = None
    detail: str = ""
    available: bool = True

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "value": None if self.value is None else round(self.value, 4),
            "detail": self.detail,
            "available": self.available,
        }


@dataclass
class RiskResult:
    total: float
    structure: float
    breakdown: float
    level: str
    tone: str
    dimensions: List[Dimension] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    capped: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 1),
            "structure": round(self.structure, 1),
            "breakdown": round(self.breakdown, 1),
            "level": self.level,
            "tone": self.tone,
            "dimensions": [d.as_dict() for d in self.dimensions],
            "signals": self.signals,
            "capped": self.capped,
            "note": self.note,
        }


# ---------------- 单维度打分（纯函数）----------------
def score_concentration(ratio: Optional[float]) -> Optional[float]:
    """资金集中度：成交额前 5% 个股占全市场比重。45% → 0 分，52% → 20 分。"""
    if ratio is None:
        return None
    if ratio <= 0.45:
        return 0.0
    if ratio >= 0.52:
        return 20.0
    return (ratio - 0.45) / 0.07 * 20.0


def score_industry(share: Optional[float]) -> Optional[float]:
    """行业集中度：前三大行业成交额占比。20% → 0 分，40% → 20 分。"""
    if share is None:
        return None
    if share <= 0.20:
        return 0.0
    if share >= 0.40:
        return 20.0
    return (share - 0.20) / 0.20 * 20.0


def score_stress(severe_ratio: Optional[float]) -> Optional[float]:
    """恐慌扩散：跌幅 ≥5% 个股占比。2% → 0 分，12% → 25 分。"""
    if severe_ratio is None:
        return None
    if severe_ratio <= 0.02:
        return 0.0
    if severe_ratio >= 0.12:
        return 25.0
    return (severe_ratio - 0.02) / 0.10 * 25.0


def score_breadth(up_ratio: Optional[float]) -> Optional[float]:
    """宽度：上涨家数占比。45% 以上不扣分，15% 以下满额。"""
    if up_ratio is None:
        return None
    if up_ratio >= 0.45:
        return 0.0
    if up_ratio <= 0.15:
        return 15.0
    return (0.45 - up_ratio) / 0.30 * 15.0


def score_limit(broken_ratio: Optional[float],
                limit_down: Optional[int]) -> Optional[float]:
    """炸板率 + 跌停家数，取两者中更严重的。"""
    if broken_ratio is None and limit_down is None:
        return None
    broken = 0.0
    if broken_ratio is not None:
        broken = min(1.0, max(0.0, (broken_ratio - 0.30) / 0.40))
    downs = 0.0
    if limit_down is not None:
        downs = min(1.0, max(0.0, float(limit_down) / 60.0))
    return max(broken, downs) * 10.0


def _level(total: float) -> tuple:
    if total < 20:
        return "安全", "cold"
    if total < 40:
        return "中性", "flat"
    if total < 60:
        return "偏高", "warm"
    return "危险", "hot"


def _scale(dimensions: List[Dimension], track_max: float) -> float:
    """把可用维度按满分比例缩放到轨道满分，缺失维度不参与分母。"""
    available = [d for d in dimensions if d.available and d.score is not None]
    if not available:
        return 0.0
    raw = sum(d.score for d in available)
    possible = sum(d.max_score for d in available)
    if possible <= 0:
        return 0.0
    return raw / possible * track_max


def evaluate(concentration: Optional[float] = None,
             industry: Optional[float] = None,
             severe_ratio: Optional[float] = None,
             up_ratio: Optional[float] = None,
             broken_ratio: Optional[float] = None,
             limit_down: Optional[int] = None,
             bull_trend: bool = False,
             concentration_percentile: Optional[float] = None,
             industry_percentile: Optional[float] = None) -> RiskResult:
    """双轨评分。任一维度为 None 时会被排除在分母之外。"""
    dims = [
        Dimension("concentration", "资金集中度", score_concentration(concentration) or 0.0,
                  20.0, concentration,
                  f"前 5% 个股成交额占比 {concentration*100:.1f}%"
                  if concentration is not None else "数据缺失",
                  concentration is not None),
        Dimension("industry", "行业集中度", score_industry(industry) or 0.0,
                  20.0, industry,
                  f"前三大行业成交额占比 {industry*100:.1f}%"
                  if industry is not None else "数据缺失",
                  industry is not None),
    ]
    breakdown_dims = [
        Dimension("stress", "恐慌扩散", score_stress(severe_ratio) or 0.0,
                  25.0, severe_ratio,
                  f"跌幅≥5% 个股占比 {severe_ratio*100:.1f}%"
                  if severe_ratio is not None else "数据缺失",
                  severe_ratio is not None),
        Dimension("breadth", "市场宽度", score_breadth(up_ratio) or 0.0,
                  15.0, up_ratio,
                  f"上涨家数占比 {up_ratio*100:.1f}%"
                  if up_ratio is not None else "数据缺失",
                  up_ratio is not None),
        Dimension("limit", "炸板跌停", score_limit(broken_ratio, limit_down) or 0.0,
                  10.0, broken_ratio,
                  f"炸板率 {broken_ratio*100:.0f}%，跌停 {limit_down} 家"
                  if broken_ratio is not None else "数据缺失",
                  broken_ratio is not None or limit_down is not None),
    ]

    structure = _scale(dims, STRUCTURE_MAX)
    breakdown = _scale(breakdown_dims, BREAKDOWN_MAX)

    signals: List[str] = []
    for dim in sorted(dims + breakdown_dims, key=lambda d: d.score, reverse=True):
        if dim.available and dim.score >= dim.max_score * 0.5:
            signals.append(f"{dim.label}：{dim.detail}")

    total = structure + breakdown
    capped = False
    if bull_trend and breakdown < 20 and total > 55:
        total = 55.0
        capped = True

    level, tone = _level(total)
    missing = [d.label for d in dims if not d.available]
    note = "结构轨已按可用维度归一化"
    if missing:
        note += f"（缺：{'、'.join(missing)}）"
    if capped:
        note += "；主升趋势下风险分封顶 55"

    return RiskResult(
        total=total, structure=structure, breakdown=breakdown,
        level=level, tone=tone, dimensions=dims + breakdown_dims,
        signals=signals[:4], capped=capped, note=note,
    )
