"""模块注册表：按 config/modules.yaml 的 id 动态导入 src.modules.{id}.MODULE。"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional

from ..config import load_modules
from .base import BaseModule


class Registry:
    def __init__(self, thresholds: dict, only: Optional[List[str]] = None):
        self.thresholds = thresholds
        self.only = only
        self._modules: Dict[str, BaseModule] = {}
        self._load()

    def _load(self) -> None:
        for cfg in load_modules():
            mid = str(cfg.get("id") or "").strip()
            if not mid:
                continue
            if self.only and mid not in self.only:
                continue
            if not cfg.get("enabled", True):
                continue
            try:
                mod = importlib.import_module(f"src.modules.{mid}")
            except ModuleNotFoundError:
                continue
            cls = getattr(mod, "MODULE", None)
            if cls is None:
                continue
            self._modules[mid] = cls(cfg, self.thresholds)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def ids(self) -> List[str]:
        return list(self._modules.keys())

    def get(self, mid: str) -> Optional[BaseModule]:
        return self._modules.get(mid)
