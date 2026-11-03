"""运行上下文：一次 run 内共享的配置、数据与中间结果。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .data import calendar
from .data.store import ParquetStore

# 这些表按报告期存储（如 20251231），不按交易日，可用性判断方式不同
PERIOD_TABLES = {"fina_indicator"}
# 这些表以 ts_code 为分区键（按需缓存），不是日期
KEYED_TABLES = {"dividend"}


@dataclass
class Ctx:
    trade_date: str                      # compact YYYYMMDD
    store: ParquetStore
    thresholds: dict
    client: Optional[object] = None
    offline: bool = False
    shared: Dict[str, object] = field(default_factory=dict)   # 模块间共享

    # ---------- 数据访问 ----------
    def table(self, name: str, dates: Optional[List[str]] = None) -> pd.DataFrame:
        return self.store.read_dated(name, dates)

    def on(self, name: str, date: Optional[str] = None) -> pd.DataFrame:
        """读取某张表在指定交易日的数据（默认本次运行日）。"""
        return self.store.read_dated(name, [date or self.trade_date])

    def window(self, name: str, days: int, end: Optional[str] = None) -> pd.DataFrame:
        """读取截至 end 的最近 days 个开市日数据。"""
        dates = calendar.recent_open_dates(self.store, days, end=end or self.trade_date)
        return self.store.read_dated(name, dates)

    def available(self, name: str) -> bool:
        if name in PERIOD_TABLES or name in KEYED_TABLES:
            return bool(self.store.list_dates(name))
        return self.store.has_dated(name, self.trade_date)

    def missing_tables(self, names: List[str]) -> List[str]:
        return [n for n in names if not self.available(n)]

    @property
    def trade_date_display(self) -> str:
        return calendar.to_display(self.trade_date)
