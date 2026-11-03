"""命令行入口：fetch / run / serve / status / ping。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import config
from .config import DATA_DIR, OUTPUT_DIR, ensure_dirs, load_env, load_thresholds
from .context import Ctx
from .data import calendar, fetcher
from .data.client import TushareClient, TushareError, find_token
from .data.store import ParquetStore
from .modules.base import STATUS_ERROR, STATUS_OK
from .modules.registry import Registry
from .render import build as render_build


def _store() -> ParquetStore:
    return ParquetStore(DATA_DIR)


def _client(required: bool = True) -> Optional[TushareClient]:
    try:
        return TushareClient()
    except TushareError as exc:
        if required:
            print(f"✗ {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"⚠ {exc}（继续用本地已有数据）", file=sys.stderr)
        return None


# ---------------- fetch ----------------
def cmd_fetch(args) -> int:
    ensure_dirs()
    store = _store()
    client = _client()

    if getattr(args, "lite", False):
        # 与 `run` 用同一个交易日：周末/节假日回退到最近的开市日
        raw = (args.date or calendar.today_compact()).replace("-", "")
        target = calendar.latest_open_date(store, on_or_before=raw)
        print(f"│ 轻量抓取（定投看板所需，目标 {target}）")
        fetcher.fetch_lite(client, store, target)
        return 0

    print("│ 拉取交易日历")
    calendar.ensure_trade_cal(client, store, start_year=2015,
                              force=args.force_cal)

    target = args.date
    print("│ 增量拉取")
    results = fetcher.fetch(
        client, store,
        target=target,
        backfill_days=args.backfill_days,
        backfill_years=args.backfill,
        tables=args.table or None,
        all_tables=args.all_tables,
        with_dividend=getattr(args, "with_dividend", False),
        dividend_all=getattr(args, "dividend_all", False),
        dividend_limit=8000,
    )
    print("└ 完成")
    print(fetcher.summarize(results))
    failed = sum(r.failed for r in results.values())
    return 1 if failed and args.strict else 0


# ---------------- run ----------------
def cmd_run(args) -> int:
    ensure_dirs()
    store = _store()
    thresholds = load_thresholds()

    if calendar.calendar_frame(store).empty:
        print("✗ 本地没有交易日历。先执行：make fetch", file=sys.stderr)
        return 2

    trade_date = fetcher.resolve_target(store, args.date)
    ctx = Ctx(trade_date=trade_date, store=store, thresholds=thresholds)

    only = args.only.split(",") if args.only else None
    registry = Registry(thresholds, only=only)
    if len(registry) == 0:
        print("✗ 没有启用任何模块，检查 config/modules.yaml", file=sys.stderr)
        return 2

    print(f"│ 交易日 {ctx.trade_date_display} · 模块 {len(registry)} 个")
    results = []
    for module in registry:
        result = module.run(ctx)
        results.append(result)
        mark = {"ok": "✓", "empty": "·", "stale": "⚠", "error": "✗"}.get(result.status, "?")
        print(f"│  {mark} {result.title:12s} {result.status:6s} {result.elapsed_ms:5d}ms  {result.summary[:60]}")

    paths = render_build.build(results, ctx.trade_date_display)
    print(f"└ 已生成 {paths['index']}（{paths['index'].stat().st_size // 1024} KB）")

    errors = [r for r in results if r.status == STATUS_ERROR]
    if args.require_ok and errors:
        print(f"✗ {len(errors)} 个模块异常", file=sys.stderr)
        return 1
    return 0


# ---------------- serve ----------------
def cmd_serve(args) -> int:
    import functools
    import http.server
    import socketserver

    index = OUTPUT_DIR / "index.html"
    if not index.exists():
        print("✗ 还没有看板，先执行 make run", file=sys.stderr)
        return 2

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(OUTPUT_DIR))
    # 绑 0.0.0.0 而不是 127.0.0.1：手机在同一 Wi-Fi 下也能打开
    import socket
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        lan_ip = probe.getsockname()[0]
        probe.close()
    except Exception:                        # noqa: BLE001
        lan_ip = ""
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"电脑  http://127.0.0.1:{args.port}/")
        if lan_ip:
            print(f"手机  http://{lan_ip}:{args.port}/   （同一 Wi-Fi）")
        print("Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
    return 0


# ---------------- status ----------------
def cmd_status(args) -> int:
    store = _store()
    stats = store.stats()
    if not stats:
        print("本地无数据。执行 make fetch 开始拉取。")
        return 0

    print(f"{'表':22s}{'模式':8s}{'文件':>6s}   {'最早':10s} {'最新':10s}")
    print("-" * 62)
    for name, info in stats.items():
        print(f"{name:22s}{info['mode']:8s}{info['files']:>6d}   "
              f"{str(info['first'] or '-'):10s} {str(info['last'] or '-'):10s}")

    cal = calendar.calendar_frame(store)
    if not cal.empty:
        latest = calendar.latest_open_date(store, on_or_before=calendar.today_compact())
        print(f"\n最近开市日: {calendar.to_display(latest) if latest else '-'}")
    return 0


# ---------------- ping ----------------
def cmd_ping(args) -> int:
    token = find_token()
    if not token:
        print("✗ 未找到 TUSHARE_TOKEN", file=sys.stderr)
        return 2
    print(f"✓ Token 已找到（长度 {len(token)}）")
    client = _client()
    try:
        df = client.query("trade_cal", exchange="SSE",
                          start_date="20240101", end_date="20240110")
        print(f"✓ 接口连通，trade_cal 返回 {len(df)} 行")
    except TushareError as exc:
        print(f"✗ 接口调用失败: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------- backtest ----------------
def cmd_backtest(args) -> int:
    import importlib

    from . import backtest as bt
    from .config import load_modules

    store = _store()
    thresholds = load_thresholds()

    cfg = next((m for m in load_modules() if m.get("id") == args.module), None)
    if cfg is None:
        print(f"✗ config/modules.yaml 里没有模块 {args.module}", file=sys.stderr)
        return 2
    try:
        mod = importlib.import_module(f"src.modules.{args.module}")
    except ModuleNotFoundError:
        print(f"✗ 模块文件 src/modules/{args.module}.py 不存在", file=sys.stderr)
        return 2

    horizons = [int(x) for x in args.horizons.split(",")]
    print(f"│ 回测 {args.module} · 未来 {horizons} 日 · "
          f"{'次日开盘' if args.entry == 'next_open' else '当日收盘'}买入")
    report = bt.run(store, thresholds, args.module, mod.MODULE, cfg,
                    days=args.days, step=args.step, horizons=horizons,
                    entry=args.entry, verbose=not args.quiet)
    if report is None:
        print("✗ 历史数据不足，无法回测（需要至少 140 个交易日预热）",
              file=sys.stderr)
        return 1
    print()
    print(report.table())
    return 0


# ---------------- 周期行业扫描 ----------------

# ---------------- 规则回测 ----------------

# ---------------- 单股体检 ----------------

# ---------------- 本地应用 ----------------

# ---------------- 交易流水 ----------------
def _latest_price(store, code: str) -> Optional[float]:
    """本地有 etf_daily 就用缓存，没有就现拉一次（CI 里没有缓存）。"""
    try:
        df = store.read_single("etf_daily")
    except Exception:                      # noqa: BLE001
        df = None
    if df is not None and not df.empty:
        sub = df[df["ts_code"] == code].sort_values("trade_date")
        if not sub.empty:
            return float(sub["close"].iloc[-1])
    try:
        client = _client()
        today = calendar.today_compact()
        fresh = client.query("fund_daily", ts_code=code,
                             start_date=calendar.shift_days(today, -20),
                             end_date=today)
        if fresh is not None and not fresh.empty:
            return float(fresh.sort_values("trade_date")["close"].iloc[-1])
    except Exception:                      # noqa: BLE001
        return None
    return None






# ---------------- portfolio ----------------

# ---------------- 参数 ----------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a-share-dashboard",
                                     description="A股看板：数据拉取与看板生成")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="增量拉取数据")
    p_fetch.add_argument("--date", help="目标日期 YYYY-MM-DD")
    p_fetch.add_argument("--backfill", type=int, default=0, metavar="YEARS",
                         help="回补 N 年")
    p_fetch.add_argument("--backfill-days", type=int, default=0, metavar="DAYS",
                         help="回补 N 个自然日")
    p_fetch.add_argument("--table", action="append", help="只拉指定表（可多次）")
    p_fetch.add_argument("--dividend-all", action="store_true",
                         help="分红缓存不限股息率门槛，全市场拉一遍（一次性）")
    p_fetch.add_argument("--with-dividend", action="store_true",
                         help="同时更新分红历史缓存（逐股接口，较慢，建议每周一次）")
    p_fetch.add_argument("--all-tables", action="store_true",
                         help="拉取全部已定义的表（默认只拉看板用到的 4 张）")
    p_fetch.add_argument("--lite", action="store_true",
                         help="只拉定投看板需要的数据（约 100MB，适合 CI）")
    p_fetch.add_argument("--force-cal", action="store_true", help="强制刷新交易日历")
    p_fetch.add_argument("--strict", action="store_true", help="有失败即退出非 0")
    p_fetch.set_defaults(func=cmd_fetch)

    p_run = sub.add_parser("run", help="运行模块并生成看板")
    p_run.add_argument("--date", help="目标日期 YYYY-MM-DD")
    p_run.add_argument("--only", help="只跑指定模块，逗号分隔")
    p_run.add_argument("--smoke", action="store_true", help="冒烟：任何模块异常即失败")
    p_run.add_argument("--require-ok", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="本地预览")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_status = sub.add_parser("status", help="查看数据水位线")
    p_status.set_defaults(func=cmd_status)

    p_ping = sub.add_parser("ping", help="检查 token 与接口连通")
    p_ping.set_defaults(func=cmd_ping)

    p_bt = sub.add_parser("backtest", help="历史回测某个模块")
    p_bt.add_argument("--module", required=True, help="模块 id，如 breakout / dragon")
    p_bt.add_argument("--days", type=int, default=200, help="最近多少个交易日")
    p_bt.add_argument("--step", type=int, default=5, help="抽样步长")
    p_bt.add_argument("--horizons", default="5,10,20", help="持有期，逗号分隔")
    p_bt.add_argument("--entry", default="next_open",
                      choices=["next_open", "close"], help="买入价口径")
    p_bt.add_argument("--quiet", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and getattr(args, "smoke", False):
        args.require_ok = True
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
