"""股吧页面解析测试（不联网，只验证解析与过滤）。"""
from __future__ import annotations

import json
import unittest

import pandas as pd

from src.data import guba


def make_page(posts):
    payload = json.dumps({"re": posts, "count": len(posts)}, ensure_ascii=False)
    return f"<html><script>var article_list={payload};</script></html>"


class ParsePageTest(unittest.TestCase):
    def test_parses_embedded_json(self):
        html = make_page([{"post_id": 1, "post_title": "看多",
                           "post_publish_time": "2026-10-19 10:00:00"}])
        posts = guba._parse_page(html)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["post_title"], "看多")

    def test_missing_article_list_returns_empty(self):
        self.assertEqual(guba._parse_page("<html>no data</html>"), [])

    def test_broken_json_returns_empty(self):
        html = "<script>var article_list={\"re\": [ broken };</script>"
        self.assertEqual(guba._parse_page(html), [])


class ToFrameTest(unittest.TestCase):
    def test_maps_fields_and_drops_incomplete(self):
        frame = guba._to_frame([
            {"post_id": 1, "post_title": "标题A", "post_publish_time": "2026-10-19 09:30:00",
             "user_nickname": "张三", "post_click_count": 12, "post_comment_count": 3},
            {"post_id": 2, "post_title": "", "post_publish_time": "2026-10-19 09:31:00"},
            {"post_id": 3, "post_title": "标题B", "post_publish_time": ""},
        ])
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["title"], "标题A")
        self.assertEqual(frame.iloc[0]["clicks"], 12)

    def test_empty_input(self):
        self.assertTrue(guba._to_frame([]).empty)


class FilterByDateTest(unittest.TestCase):
    def test_keeps_only_target_day(self):
        frame = pd.DataFrame({
            "post_id": [1, 2, 3],
            "title": ["a", "b", "c"],
            "publish_time": ["2026-10-19 10:00:00", "2026-10-18 23:00:00",
                             "2026-10-19 21:00:00"],
            "publish_date": ["2026-10-19", "2026-10-18", "2026-10-19"],
        })
        kept = guba.filter_by_date(frame, "2026-10-19")
        self.assertEqual(list(kept["post_id"]), [1, 3])

    def test_empty_frame_passthrough(self):
        self.assertTrue(guba.filter_by_date(pd.DataFrame(), "2026-10-19").empty)


if __name__ == "__main__":
    unittest.main()
