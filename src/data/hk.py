"""港股数据：东方财富公开接口，无需 token、无需第三方库。

覆盖范围足够做基本面体检：
- K 线（前复权）       push2his.eastmoney.com
- 财务主要指标         datacenter.eastmoney.com  RPT_HKF10_FN_MAININDICATOR
  含 PE_TTM / PB_TTM / ROE_AVG / 毛利率 / 负债率 / EPS_TTM / DPS_HKD

注意：这是公开网页接口，不是正式 API。字段和路径可能变，
所以所有请求都做了失败兜底，并且缓存到本地，避免重复打。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://emweb.securities.eastmoney.com/"}
HK_MARKET = "116"          # 东方财富港股市场编号
TIMEOUT = 20


def _get(url: str, retries: int = 3) -> Optional[dict]:
    """公开接口偶发失败/限流，重试几次再放弃。"""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", "ignore"))
        except Exception:                    # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(1.2 * (attempt + 1))
    return None


def secid(code: str) -> str:
    """01378.HK → 116.01378"""
    digits = code.split(".")[0].zfill(5)
    return f"{HK_MARKET}.{digits}"


def secucode(code: str) -> str:
    return f"{code.split('.')[0].zfill(5)}.HK"


# ---------------- K 线 ----------------
def fetch_kline(code: str, start: str = "20100101", adjust: int = 0) -> pd.DataFrame:
    """adjust: 0=不复权（估值用） 1=前复权 2=后复权。

    默认不复权——前复权在长期高分红股票上会把历史价格减成负数
    （中国宏桥 2011 年前复权价约 -13 港元），拿它算 PE 会得到负值。
    """
    fields = "f51,f52,f53,f54,f55,f56,f57"
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid(code)}&fields1=f1,f2,f3,f4,f5&fields2={fields}"
           f"&klt=101&fqt={adjust}&beg={start}&end=20500101")
    data = _get(url)
    if not data or not (data.get("data") or {}).get("klines"):
        return pd.DataFrame()
    rows = []
    for line in data["data"]["klines"]:
        p = line.split(",")
        if len(p) < 7:
            continue
        rows.append({
            "trade_date": p[0].replace("-", ""),
            "open": float(p[1]), "close": float(p[2]),
            "high": float(p[3]), "low": float(p[4]),
            "volume": float(p[5]), "amount": float(p[6]),
        })
    df = pd.DataFrame(rows)
    df["ts_code"] = secucode(code)
    return df.sort_values("trade_date").reset_index(drop=True)


# ---------------- 财务主要指标 ----------------
FIN_REPORT = "RPT_HKF10_FN_MAININDICATOR"


def fetch_financials(code: str, page_size: int = 40) -> pd.DataFrame:
    """返回按报告期倒序的主要指标（含 PE_TTM / PB_TTM 等）。"""
    q = urllib.parse.urlencode({
        "reportName": FIN_REPORT, "columns": "ALL",
        "filter": f'(SECUCODE="{secucode(code)}")',
        "pageSize": page_size, "sortColumns": "REPORT_DATE", "sortTypes": "-1",
    })
    url = f"https://datacenter.eastmoney.com/securities/api/data/v1/get?{q}"
    data = _get(url)
    rows = ((data or {}).get("result") or {}).get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["report_date"] = df["REPORT_DATE"].astype(str).str[:10].str.replace("-", "")
    return df.sort_values("report_date").reset_index(drop=True)


# ---------------- 分红 ----------------
DIV_REPORT = "RPT_HKF10_MAIN_DIVBASIC"


def fetch_dividends(code: str, page_size: int = 50) -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "reportName": DIV_REPORT, "columns": "ALL",
        "filter": f'(SECUCODE="{secucode(code)}")',
        "pageSize": page_size, "sortColumns": "NOTICE_DATE", "sortTypes": "-1",
    })
    url = f"https://datacenter.eastmoney.com/securities/api/data/v1/get?{q}"
    data = _get(url)
    rows = ((data or {}).get("result") or {}).get("data") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------- 本地缓存 ----------------
def cache_dir(store) -> Path:
    d = Path(store.root) / "parquet" / "hk"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(store, kind: str, code: str) -> Path:
    return cache_dir(store) / f"{kind}_{code.split('.')[0].zfill(5)}.parquet"


def load_or_fetch(store, kind: str, code: str, max_age_hours: float = 20.0):
    path = _cache_path(store, kind, code)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    if fresh:
        try:
            return pd.read_parquet(path)
        except Exception:                    # noqa: BLE001
            pass
    func = {"kline": fetch_kline, "fin": fetch_financials, "div": fetch_dividends}[kind]
    df = func(code)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df
