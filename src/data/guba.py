"""东方财富股吧：只抓公开列表页的标题与发布时间。

设计约束：
- 只读列表页，不需要登录、不碰用户数据
- 单页 80 条，页间 sleep，避免给站点压力
- 页面里内嵌 `var article_list={...}` JSON，比解析 HTML 稳定
- 抓不到就返回已有部分，绝不抛异常拖垮整条流水线
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Optional

import pandas as pd

BASE_URL = "https://guba.eastmoney.com/list,{code},f_{page}.html"
DEFAULT_CODE = "zssh000001"          # 上证指数吧

USER_AGENTS = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
)

ARTICLE_RE = re.compile(r"var\s+article_list\s*=\s*(\{.*?\});", re.S)

COLUMNS = ["post_id", "title", "publish_time", "user", "clicks", "comments"]


def _parse_page(text: str) -> list:
    match = ARTICLE_RE.search(text)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return data.get("re") or []


def _to_frame(posts: list) -> pd.DataFrame:
    rows = []
    for post in posts:
        title = (post.get("post_title") or "").strip()
        publish = (post.get("post_publish_time") or "").strip()
        if not title or not publish:
            continue
        rows.append({
            "post_id": post.get("post_id"),
            "title": title,
            "publish_time": publish,
            "user": post.get("user_nickname") or "",
            "clicks": post.get("post_click_count") or 0,
            "comments": post.get("post_comment_count") or 0,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_posts(code: str = DEFAULT_CODE,
                target_date: Optional[str] = None,
                max_pages: int = 40,
                delay: float = 0.4,
                timeout: int = 15,
                verbose: bool = False) -> pd.DataFrame:
    """抓取列表页。target_date 形如 2026-10-19，用于决定何时停止翻页。"""
    import requests

    frames = []
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://guba.eastmoney.com/",
    }

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(code=code, page=page)
        request_headers = dict(headers, **{"User-Agent": random.choice(USER_AGENTS)})
        try:
            response = requests.get(url, headers=request_headers, timeout=timeout)
            if response.status_code != 200:
                if verbose:
                    print(f"    ✗ 股吧第 {page} 页 HTTP {response.status_code}")
                break
            response.encoding = "utf-8"
            frame = _to_frame(_parse_page(response.text))
        except Exception as exc:                      # noqa: BLE001
            if verbose:
                print(f"    ✗ 股吧第 {page} 页失败: {type(exc).__name__}")
            break

        if frame.empty:
            break
        frames.append(frame)

        if verbose:
            print(f"    第 {page} 页 {len(frame)} 条")

        # 整页都早于目标日期 → 已经翻过头了
        if target_date:
            dates = frame["publish_time"].str.slice(0, 10)
            if (dates < target_date).all():
                break

        time.sleep(delay)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    result["publish_date"] = result["publish_time"].str.slice(0, 10)
    return result


def filter_by_date(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    if frame.empty or "publish_date" not in frame.columns:
        return frame
    return frame[frame["publish_date"] == date].reset_index(drop=True)
