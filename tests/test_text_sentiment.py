"""股吧文本情绪：词典打分的边界与聚合测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.kit import text_sentiment as ts

LEX = {
    "bullish_words": ["涨停", "起飞", "抄底"],
    "bearish_words": ["跌停", "暴跌", "割肉"],
    "bullish_phrases": ["关灯吃面"],
    "bearish_phrases": ["牛回速归"],
    "negation_words": ["不", "没", "别"],
    "spam_keywords": ["配资", "加群"],
}


class ScoreTitleTest(unittest.TestCase):
    def test_bullish(self):
        self.assertGreater(ts.score_title("明天要涨停了", LEX), 0)

    def test_bearish(self):
        self.assertLess(ts.score_title("又是一根暴跌", LEX), 0)

    def test_neutral(self):
        self.assertEqual(ts.score_title("今天天气不错", LEX), 0)

    def test_negation_flips_bullish_word(self):
        """「不涨停」应该算看空，而不是看多。"""
        self.assertLess(ts.score_title("这股不涨停", LEX), 0)

    def test_negation_flips_bearish_word(self):
        """「不会跌停」应该算看多。"""
        self.assertGreater(ts.score_title("不会跌停", LEX), 0)

    def test_bearish_phrase_takes_precedence(self):
        # 「牛回速归」是空头短语，不应被拆成单词
        self.assertLess(ts.score_title("牛回速归", LEX), 0)

    def test_bullish_phrase_is_removed_before_words(self):
        # 「关灯吃面」本身是看空梗，词典里设为多头短语仅用于验证短语优先
        score = ts.score_title("关灯吃面", LEX)
        self.assertEqual(score, 1)

    def test_empty_title(self):
        self.assertEqual(ts.score_title("", LEX), 0)


class SpamTest(unittest.TestCase):
    def test_detects_spam(self):
        self.assertTrue(ts.is_spam("加群带你炒股", LEX))
        self.assertFalse(ts.is_spam("今天亏麻了", LEX))


class AnalyzeTest(unittest.TestCase):
    def make(self, titles):
        return pd.DataFrame({"title": titles, "publish_date": "2026-10-19"})

    def test_aggregate_sentiment(self):
        posts = self.make(["涨停", "起飞", "抄底", "暴跌"])
        result = ts.analyze(posts, LEX)
        self.assertEqual(result["bull_posts"], 3)
        self.assertEqual(result["bear_posts"], 1)
        self.assertGreater(result["sentiment"], 0)
        self.assertGreater(result["score"], 50)

    def test_spam_excluded(self):
        posts = self.make(["涨停", "加群领牛股", "配资找我"])
        result = ts.analyze(posts, LEX)
        self.assertEqual(result["spam_posts"], 2)
        self.assertEqual(result["scored"], 1)

    def test_all_neutral_yields_zero(self):
        posts = self.make(["今天天气不错", "随便说说"])
        result = ts.analyze(posts, LEX)
        self.assertEqual(result["sentiment"], 0.0)
        self.assertEqual(result["score"], 50.0)

    def test_empty_input(self):
        result = ts.analyze(pd.DataFrame(), LEX)
        self.assertIsNone(result["sentiment"])
        self.assertEqual(result["posts"], 0)

    def test_sentiment_bounds(self):
        only_bull = ts.analyze(self.make(["涨停", "起飞"]), LEX)
        self.assertAlmostEqual(only_bull["sentiment"], 1.0)
        self.assertAlmostEqual(only_bull["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
