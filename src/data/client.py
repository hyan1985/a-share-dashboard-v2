"""Tushare 客户端：token 发现、限速、重试。

全项目只有这里能创建 Tushare 连接（数据层单入口原则）。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from typing import Optional

import pandas as pd

KEYCHAIN_SERVICE = "cursor-quant-tushare"
KEYCHAIN_ACCOUNT = "default"

# 踩过的坑：tushare 各接口的频次上限并不相同。
# 积分表写的 500 次/分钟是「常规数据」的上限，但 limit_list_d 这类
# 打板专题接口单独限制 200 次/分钟，用全局 450 去跑会撞墙。
API_RATE_LIMITS = {
    # 分红是按 ts_code 逐个查的接口，全市场要 5000+ 次。
    # 官方频率上限较低，给 60/分钟留足余量，宁可慢也不被封。
    "dividend": 60,
    "limit_list_d": 180,
    "limit_step": 180,
    "limit_cpt_list": 180,
    "top_list": 180,
    "moneyflow": 180,
}


class RateLimiter:
    """滑动窗口限速。默认 450/分钟，给 500 上限留余量。"""

    def __init__(self, per_minute: int = 450):
        self.per_minute = max(1, int(per_minute))
        self._times: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._times and now - self._times[0] > 60.0:
                self._times.popleft()
            if len(self._times) >= self.per_minute:
                wait = 60.0 - (now - self._times[0]) + 0.05
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                while self._times and now - self._times[0] > 60.0:
                    self._times.popleft()
            self._times.append(time.monotonic())


def find_token() -> Optional[str]:
    """按优先级找 token：环境变量 → 钥匙串 → 常见文件。"""
    env = os.environ.get("TUSHARE_TOKEN", "").strip()
    if env:
        return env

    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE,
             "-a", KEYCHAIN_ACCOUNT, "-w"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass

    # 只约定一个与项目无关的固定位置，避免把某台机器的目录结构写进代码
    for candidate in (os.path.expanduser("~/.secrets/tushare_token"),):
        try:
            if os.path.isfile(candidate):
                with open(candidate, encoding="utf-8") as fh:
                    value = fh.read().strip()
                if value:
                    return value
        except Exception:
            continue
    return None


class TushareError(RuntimeError):
    pass


class TushareClient:
    """薄封装：限速 + 重试 + 统一的错误语义。"""

    def __init__(self, token: Optional[str] = None, per_minute: Optional[int] = None,
                 retries: int = None):
        token = token or find_token()
        if not token:
            raise TushareError(
                "未找到 TUSHARE_TOKEN。请设置环境变量、写入 .env，"
                "或存到钥匙串（服务名 cursor-quant-tushare）。"
            )
        import tushare as ts

        self._pro = ts.pro_api(token)
        self.default_rpm = per_minute or int(os.environ.get("TUSHARE_RPM", "450"))
        self._limiters = {}
        # 全量分红缓存在 Actions 里跑，限频重试次数可以通过环境变量放宽
        self.retries = int(os.environ.get("TUSHARE_RETRIES", "3")) if retries is None else retries
        self.calls = 0

    def _limiter(self, api: str) -> RateLimiter:
        if api not in self._limiters:
            rpm = API_RATE_LIMITS.get(api, self.default_rpm)
            self._limiters[api] = RateLimiter(rpm)
        return self._limiters[api]

    @property
    def limiter(self) -> RateLimiter:
        """兼容旧用法：默认限速器。"""
        return self._limiter("__default__")

    def query(self, api: str, **kwargs) -> pd.DataFrame:
        """调用一个接口，失败重试。返回 DataFrame（可能为空）。"""
        last_err: Optional[Exception] = None
        limiter = self._limiter(api)
        for attempt in range(self.retries):
            limiter.acquire()
            try:
                self.calls += 1
                df = self._pro.query(api, **kwargs)
                if isinstance(df, dict):
                    raise TushareError(f"{api} 返回错误: {df}")
                if df is None:
                    return pd.DataFrame()
                return df
            except Exception as exc:  # noqa: BLE001 - 需要吞掉网络/接口异常
                last_err = exc
                msg = str(exc)
                # 「每分钟最多访问该接口 N 次」这类是限频，不是网络问题。
                # 指数退避对限频没用——时间没到再试还是被拒，必须等够。
                if any(k in msg for k in ("每分钟", "频率", "超过", "limit", "too many")):
                    wait = 60.0 + 15.0 * attempt
                    if self.verbose:
                        print(f"    ⏳ {api} 触发限频，等 {wait:.0f} 秒后重试"
                              f"（第 {attempt + 1}/{self.retries} 次）")
                    time.sleep(wait)
                else:
                    time.sleep(min(2 ** attempt, 8))
        raise TushareError(f"{api} 调用失败（重试 {self.retries} 次）: {last_err}")
