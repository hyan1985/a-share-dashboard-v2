#!/usr/bin/env python3
"""词库体检：找出「被判中性但其实有情绪」的帖子，提取漏掉的短语。

论坛里的情绪大多不在单个词上，而在短语和句子里。单词层面的高频词
永远是「明天/今天/指数」这类中性名词，光看词频找不到盲区。

做法反过来：先看哪些帖子被判成中性、但字面明显带情绪（惨/麻/崩/哈哈…），
再从这些帖子里提 3~6 字片段，去掉已在词库里的，按「不同帖子数」排序。

输出的候选要人来判断归属，不自动写入配置——自动加词会把反讽加反，
这是这个项目踩过的坑（「抄底」曾被当成多头词）。

用法：
    python tools/lexicon_gap.py                # 用本地缓存的最新股吧数据
    python tools/lexicon_gap.py --top 60 --json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

EMOTION_CHARS = "惨麻哭骂坑套亏割跌崩死完蛋狗屎烂垃圾骗贪婪怕慌急天台上吊跳楼凉灭"
EMOTION_RE = re.compile("[" + EMOTION_CHARS + "]" + "|！")

# 只有含这些字/词的片段才可能是情绪表达。光是高频没用——
# 「创业板」「上市公司」出现得再多也不带情绪。
HINT = re.compile(
    r"跌|破|割|套|亏|崩|死|完|凉|灭|守|顶|底|逃|接盘|空仓|跑|惨|麻|哭|骂|坑|垃圾|"
    r"骗|怕|慌|急|天台上|跳楼|新低|净流|拉|砸|抄|护|反攻|红|绿|亏钱|割肉"
)

NOISE = re.compile(
    r"明天|今天|昨天|上证|指数|大盘|科技|市场|个股|收盘|美股|茅台|银行|资金|"
    r"大家|什么|时候|一个|一天|现在|就是|还是|这个|那个|知道|可能|可以|如果"
)


def load_posts(data_dir: str):
    """读最近一天的股吧帖子。"""
    import glob
    import pandas as pd

    fs = sorted(glob.glob(os.path.join(data_dir, "parquet", "guba", "*.parquet")))
    if not fs:
        return None
    return pd.read_parquet(fs[-1])


def known_phrases(lexicon: dict) -> set:
    known = set()
    for k in ("bullish_words", "bearish_words", "bullish_phrases", "bearish_phrases"):
        known |= set(lexicon.get(k) or [])
    return known


def ngrams(texts, n_min=3, n_max=6):
    """按「不同帖子数」计数。

    论坛里同一条内容会被反复转发，按出现次数统计会让一条水帖刷满整个榜单
    （实测「分红未超过融资」就是这么刷出来的）。先去重再计数。
    """
    cnt = collections.Counter()
    for t in dict.fromkeys(texts):
        clean = re.sub(r"[^\u4e00-\u9fa5]", "", t)
        for n in range(n_min, n_max + 1):
            for i in range(len(clean) - n + 1):
                cnt[clean[i:i + n]] += 1
    return cnt


def report(posts, lexicon: dict, top: int = 40, min_count: int = 2) -> dict:
    from src.kit import text_sentiment as ts

    titles = [str(t) for t in posts["title"].tolist()] if "title" in posts.columns else []
    neutral, suspicious = [], []
    for t in titles:
        if ts.is_spam(t, lexicon):
            continue
        if ts.score_title(t, lexicon) == 0:
            neutral.append(t)
            if EMOTION_RE.search(t):
                suspicious.append(t)

    known = known_phrases(lexicon)
    cnt = ngrams(suspicious)
    cands, seen = [], set()
    for g, n in cnt.most_common(2000):
        if n < min_count:
            break
        if g in known or NOISE.search(g) or not HINT.search(g):
            continue
        if any(g in s for s in seen):
            continue
        seen.add(g)
        cands.append({"phrase": g, "count": n})
        if len(cands) >= top:
            break

    total = len(titles) or 1
    return {
        "total_posts": len(titles),
        "neutral_posts": len(neutral),
        "neutral_ratio": round(len(neutral) / total * 100, 1),
        "suspicious_posts": len(suspicious),
        "suspicious_ratio": round(len(suspicious) / max(len(neutral), 1) * 100, 1),
        "candidates": cands,
    }


def format_markdown(r: dict) -> str:
    lines = [
        "## 词库体检",
        "",
        f"- 帖子总数：**{r['total_posts']}**",
        f"- 判为中性：**{r['neutral_posts']}**（{r['neutral_ratio']}%）",
        f"- 其中字面带情绪：**{r['suspicious_posts']}**"
        f"（占中性帖 {r['suspicious_ratio']}%）",
        "",
    ]
    if r["suspicious_ratio"] > 40:
        lines.append("> ⚠️ 超过 40% 的中性帖字面带情绪，说明词库有盲区，建议补词。")
    else:
        lines.append("> 盲区比例正常，词库暂时够用。")
    lines += ["", "### 候选短语（未收录，按出现帖数排序）", "",
              "| 短语 | 次数 | 短语 | 次数 |", "|---|---:|---|---:|"]
    c = r["candidates"]
    for i in range(0, len(c), 2):
        a = c[i]
        b = c[i + 1] if i + 1 < len(c) else {"phrase": "", "count": ""}
        lines.append(f"| `{a['phrase']}` | {a['count']} | "
                     f"`{b['phrase']}` | {b['count']} |")
    lines += [
        "", "### 怎么处理", "",
        "逐个判断归属，**不要批量加**——反讽词加错方向会让情绪分和现实相反。",
        "",
        "1. 明显看空/恐慌 → `bearish_phrases`",
        "2. 明显看多/兴奋 → `bullish_phrases`",
        "3. 反讽/阴阳怪气 → `bearish_regex`（写成「看多词 + 阴阳词」的组合）",
        "4. 中性名词 → 不用管",
        "",
        "改完跑 `python -m src.cli run` 看情绪分有没有变化。",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    posts = load_posts(args.data)
    if posts is None or posts.empty:
        print("没有股吧数据，先跑 python -m src.cli fetch", file=sys.stderr)
        return 1
    from src.kit import text_sentiment as ts
    lexicon = ts.load_lexicon()
    r = report(posts, lexicon, args.top, args.min_count)
    print(json.dumps(r, ensure_ascii=False, indent=2) if args.json
          else format_markdown(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
