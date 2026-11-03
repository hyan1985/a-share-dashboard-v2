"""模块协议与通用基类。

新增一个分区只需要：
1. 在 src/modules/ 下建一个文件，定义继承 BaseModule 的类
2. 在 config/modules.yaml 里加一条（id 与文件名/类无关，靠 registry 约定）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..context import Ctx

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_STALE = "stale"
STATUS_ERROR = "error"


@dataclass
class Candidate:
    code: str
    name: str = ""
    score: float = 0.0
    industry: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    badges: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "score": round(float(self.score), 2),
            "industry": self.industry,
            "metrics": self.metrics,
            "reasons": self.reasons,
            "risks": self.risks,
            "badges": self.badges,
        }


@dataclass
class ModuleResult:
    id: str
    title: str
    kind: str
    status: str = STATUS_OK
    as_of: str = ""
    summary: str = ""
    candidates: List[Candidate] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_ms: int = 0
    expanded: bool = True

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "as_of": self.as_of,
            "summary": self.summary,
            "candidates": [c.as_dict() for c in self.candidates],
            "context": self.context,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "expanded": self.expanded,
        }


class BaseModule:
    """子类实现 compute()；其余生命周期由基类统一处理。"""

    id: str = ""
    title: str = ""
    kind: str = "screener"          # screener | environment
    requires: List[str] = []        # 依赖的数据表

    def __init__(self, cfg: Optional[dict] = None, thresholds: Optional[dict] = None):
        self.cfg = cfg or {}
        self.thresholds = thresholds or {}
        self.params = self.cfg.get("params") or {}

    # ---- 子类覆写 ----
    def compute(self, ctx: Ctx) -> dict:
        raise NotImplementedError

    def summarize(self, payload: dict, ctx: Ctx) -> str:
        return ""

    # ---- 生命周期 ----
    def run(self, ctx: Ctx) -> ModuleResult:
        started = time.time()
        result = ModuleResult(
            id=self.id, title=self.title, kind=self.kind,
            as_of=ctx.trade_date_display,
            expanded=bool(self.cfg.get("expanded", True)),
        )
        try:
            missing = ctx.missing_tables(self.requires)
            if missing:
                result.status = STATUS_STALE
                result.summary = f"缺少数据表：{', '.join(missing)}；请先执行 make fetch"
                result.elapsed_ms = int((time.time() - started) * 1000)
                return result

            payload = self.compute(ctx) or {}
            result.status = payload.get("status", STATUS_OK)
            result.candidates = payload.get("candidates", [])
            result.context = payload.get("context", {})
            result.summary = payload.get("summary") or self.summarize(payload, ctx)

            if not result.candidates and result.kind == "screener" and result.status == STATUS_OK:
                result.status = STATUS_EMPTY
        except Exception as exc:  # noqa: BLE001 - 单模块失败不能拖垮整张看板
            result.status = STATUS_ERROR
            result.error = f"{type(exc).__name__}: {exc}"

        result.elapsed_ms = int((time.time() - started) * 1000)
        return result
