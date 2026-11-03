"""股吧文本情绪：基于词典的标题打分为 [-1, 1]。

做法沿用 MarketMonitoring 已验证的思路，但简化：
1. 命中垃圾关键词的帖子直接跳过（配资、广告等）
2. **短语优先**：「关灯吃面」这类短语先匹配并整段移除，避免被拆成单词重复计分
3. 单词匹配时检查前缀否定词，「不涨了」要翻成看空而不是看多
4. 汇总 (多头 - 空头) / (多头 + 空头)
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..config import CONFIG_DIR, load_yaml

NEGATION_WINDOW = 4          # 否定词最多往前看几个字

_DEFAULT_LEXICON = {
    "bullish_words": ["涨停", "大涨", "反弹", "抄底", "看多"],
    "bearish_words": ["跌停", "大跌", "暴跌", "割肉", "看空"],
    "bullish_phrases": [],
    "bearish_phrases": [],
    "negation_words": ["不", "没", "别", "未", "无法"],
    "spam_keywords": ["配资", "广告"],
}


def load_lexicon(path=None) -> dict:
    if path is None:
        path = CONFIG_DIR / "lexicon.yaml"
    try:
        data = load_yaml(path)
    except FileNotFoundError:
        return dict(_DEFAULT_LEXICON)
    merged = dict(_DEFAULT_LEXICON)
    for key, value in (data or {}).items():
        if isinstance(value, list):
            merged[key] = [str(v) for v in value]
    return merged


def _has_negation(text: str, index: int, negations) -> bool:
    start = max(0, index - NEGATION_WINDOW)
    window = text[start:index]
    return any(word in window for word in negations)


def score_title(title: str, lexicon: dict) -> int:
    """单条标题净得分：正数偏多，负数偏空。

    短语先匹配并整段挖掉，再对剩余文本做单词匹配。
    """
    if not title:
        return 0

    text = title
    bull = 0
    bear = 0

    for phrase in lexicon.get("bullish_phrases", []):
        if phrase and phrase in text:
            bull += text.count(phrase)
            text = text.replace(phrase, " ")

    for phrase in lexicon.get("bearish_phrases", []):
        if phrase and phrase in text:
            bear += text.count(phrase)
            text = text.replace(phrase, " ")

    negations = lexicon.get("negation_words", [])

    for word in lexicon.get("bullish_words", []):
        if not word:
            continue
        start = 0
        while True:
            index = text.find(word, start)
            if index < 0:
                break
            if _has_negation(text, index, negations):
                bear += 1          # 「不涨」按看空处理
            else:
                bull += 1
            start = index + len(word)

    for word in lexicon.get("bearish_words", []):
        if not word:
            continue
        start = 0
        while True:
            index = text.find(word, start)
            if index < 0:
                break
            if _has_negation(text, index, negations):
                bull += 1          # 「不会跌」按看多处理
            else:
                bear += 1
            start = index + len(word)

    return bull - bear


def is_spam(title: str, lexicon: dict) -> bool:
    return any(word and word in title for word in lexicon.get("spam_keywords", []))


def analyze(posts: pd.DataFrame, lexicon: Optional[dict] = None,
            title_col: str = "title") -> dict:
    """汇总一批帖子的情绪。

    返回 sentiment ∈ [-1, 1]，以及多空帖数与覆盖率。
    """
    lexicon = lexicon or load_lexicon()
    empty = {
        "sentiment": None, "score": None, "posts": 0, "scored": 0,
        "bull_posts": 0, "bear_posts": 0, "neutral_posts": 0,
        "spam_posts": 0, "bull_hits": 0, "bear_hits": 0,
    }
    if posts is None or posts.empty or title_col not in posts.columns:
        return empty

    bull_hits = bear_hits = 0
    bull_posts = bear_posts = neutral = spam = 0

    for title in posts[title_col].astype(str):
        if is_spam(title, lexicon):
            spam += 1
            continue
        net = score_title(title, lexicon)
        if net > 0:
            bull_posts += 1
            bull_hits += net
        elif net < 0:
            bear_posts += 1
            bear_hits += -net
        else:
            neutral += 1

    total_hits = bull_hits + bear_hits
    sentiment = (bull_hits - bear_hits) / total_hits if total_hits else 0.0
    scored = bull_posts + bear_posts + neutral

    return {
        "sentiment": round(float(sentiment), 4),
        "score": round((float(sentiment) + 1.0) / 2.0 * 100.0, 1) if total_hits else 50.0,
        "posts": int(len(posts)),
        "scored": int(scored),
        "bull_posts": int(bull_posts),
        "bear_posts": int(bear_posts),
        "neutral_posts": int(neutral),
        "spam_posts": int(spam),
        "bull_hits": int(bull_hits),
        "bear_hits": int(bear_hits),
    }
