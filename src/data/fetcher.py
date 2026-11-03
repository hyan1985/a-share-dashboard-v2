"""增量拉取：按日分区、水位线、行数校验、断点续传。

设计要点：
- 每张表按 trade_date 一次拉全市场，绝不逐股循环。
- 落盘成功才算完成；已存在的日期直接跳过。
- 行数低于期望视为「上游还没入库」，退避重试，不写脏数据。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import calendar
from .store import ParquetStore

# 只拉上证综指：总览、风险、回测都只读它。
# 深成/创业板/科创50 目前无人使用，且科创50 2019-07 才发布，
# 混在一起会制造"行数不足"的假故障。
INDEX_CODES = ("000001.SH",)


@dataclass(frozen=True)
class TableSpec:
    name: str
    api: str
    mode: str = "by_date"          # by_date | by_period
    date_kwarg: str = "trade_date"
    expect_rows: int = 1           # 低于此值视为未入库
    codes: Sequence[str] = field(default_factory=tuple)
    label: str = ""
    # 写入时只保留这些列。全市场十年数据很大，但模块实际只用到少数几列，
    # 例如 daily_basic 有 18 列，两个模块只读 ts_code/pe_ttm/dv_ttm/total_mv。
    keep: Sequence[str] = field(default_factory=tuple)


TABLES: Tuple[TableSpec, ...] = (
    TableSpec("daily", "daily", expect_rows=4000, label="日线行情",
              keep=("ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "pct_chg", "vol", "amount")),
    TableSpec("daily_basic", "daily_basic", expect_rows=4000, label="每日指标",
              keep=("ts_code", "trade_date", "close", "pe", "pe_ttm", "pb",
                    "dv_ttm", "total_mv")),
    # 注意：这个打板专题接口早期年份（2016-2019）返回 0 行是**正常**的，
    # 不是"上游未入库"。expect_rows=1 会让每个历史日期白白重试 3 次。
    TableSpec("limit_list_d", "limit_list_d", expect_rows=0, label="涨跌停/炸板"),
    TableSpec("index_daily", "index_daily", expect_rows=1, codes=INDEX_CODES,
              label="指数日线"),
    # 以下暂未使用，需要时用 --all-tables 拉取
    TableSpec("adj_factor", "adj_factor", expect_rows=4000, label="复权因子"),
)

# 默认只拉看板真正用到的表：每日约 6 次请求
DEFAULT_TABLES: Tuple[str, ...] = ("daily", "daily_basic", "limit_list_d", "index_daily")

TABLE_MAP: Dict[str, TableSpec] = {t.name: t for t in TABLES}

# 参考表：无日期维度，整体覆盖写
REFERENCE_TABLES = ("stock_basic",)
FINANCIAL_TABLE = "fina_indicator"
DIVIDEND_TABLE = "dividend"      # 以 ts_code 为分区键（不是日期）
GUBA_TABLE = "guba"              # 股吧帖子，按交易日分区
GAP_SCAN_DAYS = 30                       # 增量时只补最近一个月的缺口

FINANCIAL_API = "fina_indicator_vip"     # 按报告期一次拉全市场，需 5000 积分


def fetch_dividend_history(client, store: ParquetStore, trade_date: str,
                           dv_threshold: float = 3.0,
                           max_age_days: int = 120,
                           limit: int = 800,
                           verbose: bool = True) -> Dict[str, int]:
    """预取高股息股票的分红历史，供「吃息」模块离线校验连续分红。

    tushare 不提供全市场按日批量拉分红历史（dividend 只支持按 ts_code 或
    公告日查询），所以用「按需缓存」：只取当日股息率 ≥ 阈值的股票，
    并且 120 天内已拉过的跳过。首次约 300~500 次请求，之后每天几次。
    """
    basic = store.read_dated("daily_basic", [trade_date])
    if basic.empty or "dv_ttm" not in basic.columns:
        return {"fetched": 0, "skipped": 0, "failed": 0, "candidates": 0}

    dv = pd.to_numeric(basic["dv_ttm"], errors="coerce").fillna(0.0)
    codes = basic.loc[dv >= dv_threshold, "ts_code"].astype(str).tolist()
    codes = codes[:limit]

    fetched = skipped = failed = 0
    for code in codes:
        age = store.age_days(DIVIDEND_TABLE, code)
        if age is not None and age <= max_age_days:
            skipped += 1
            continue
        try:
            df = client.query("dividend", ts_code=code)
        except Exception:
            failed += 1
            continue
        store.write_dated(DIVIDEND_TABLE, code, df)
        fetched += 1
    if verbose and (fetched or failed):
        print(f"    分红历史：新拉 {fetched}，跳过 {skipped}，失败 {failed}")
    return {"fetched": fetched, "skipped": skipped, "failed": failed,
            "candidates": len(codes)}


def fetch_guba(store: ParquetStore, trade_date: str,
               max_pages: int = 40, force: bool = False,
               verbose: bool = True) -> Dict[str, int]:
    """抓股吧帖子并只保留目标交易日。抓不到就不写，留待下次重试。"""
    from . import guba as guba_crawler

    if store.has_dated(GUBA_TABLE, trade_date) and not force:
        return {"fetched": 0, "skipped": 1, "failed": 0, "posts": 0}

    day = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    try:
        raw = guba_crawler.fetch_posts(target_date=day, max_pages=max_pages,
                                       verbose=verbose)
    except Exception as exc:                     # noqa: BLE001
        if verbose:
            print(f"    ✗ 股吧抓取失败: {type(exc).__name__}: {exc}")
        return {"fetched": 0, "skipped": 0, "failed": 1, "posts": 0}

    if raw.empty:
        return {"fetched": 0, "skipped": 0, "failed": 1, "posts": 0}

    today = guba_crawler.filter_by_date(raw, day)
    if today.empty:
        if verbose:
            print(f"    ⚠ 股吧抓到 {len(raw)} 条，但没有 {day} 的帖子")
        return {"fetched": 0, "skipped": 0, "failed": 1, "posts": 0}

    store.write_dated(GUBA_TABLE, trade_date, today)
    if verbose:
        print(f"    股吧：{day} 共 {len(today)} 条")
    return {"fetched": 1, "skipped": 0, "failed": 0, "posts": int(len(today))}


def annual_periods(end_year: int, years: int) -> List[str]:
    return [f"{y}1231" for y in range(end_year - years + 1, end_year + 1)]


def fetch_financials(client, store: ParquetStore, trade_date_compact: str,
                     years: int = 5, verbose: bool = True) -> Dict[str, int]:
    """按报告期拉财务指标（年报）。一次一报告期，不逐股循环。"""
    periods = annual_periods(int(trade_date_compact[:4]), years)
    fetched = skipped = failed = 0
    for period in periods:
        if store.has_dated(FINANCIAL_TABLE, period):
            skipped += 1
            continue
        if verbose:
            print(f"  → 财务指标 {period}")
        try:
            df = client.query(FINANCIAL_API, period=period)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if verbose:
                print(f"    ✗ {period}: {exc}")
            continue
        if df.empty:
            failed += 1
            continue
        store.write_dated(FINANCIAL_TABLE, period, df)
        fetched += 1
    return {"fetched": fetched, "skipped": skipped, "failed": failed,
            "periods": len(periods)}



CASHFLOW_API = "cashflow_vip"
INCOME_API = "income_vip"


def fetch_cashflow_income(client, store: ParquetStore, trade_date_compact: str,
                          years: int = 2, verbose: bool = True) -> Dict[str, int]:
    """拉现金流量表和利润表（年报），用于算「收现比」。

    收现比 = 销售商品收到的现金 ÷ 营业收入。fina_indicator 里没有
    销售商品收到的现金，必须单独拉 cashflow；营业收入在 income 里。

    只拉最近几年：筛选用的是最新一期，历史用来看趋势。
    """
    periods = annual_periods(int(trade_date_compact[:4]), years)
    out = {"cashflow": 0, "income": 0, "skipped": 0, "failed": 0}
    for period in periods:
        for table, api, fields in (
            ("cashflow", CASHFLOW_API,
             "ts_code,end_date,c_fr_sale_sg,n_cashflow_act"),
            ("income", INCOME_API, "ts_code,end_date,revenue,total_revenue"),
        ):
            if store.has_dated(table, period):
                out["skipped"] += 1
                continue
            try:
                df = client.query(api, period=period, fields=fields)
            except Exception as exc:  # noqa: BLE001
                out["failed"] += 1
                if verbose:
                    print(f"    ✗ {table} {period}: {exc}")
                continue
            if df is None or df.empty:
                out["failed"] += 1
                continue
            store.write_dated(table, period, df)
            out[table] += 1
            if verbose:
                print(f"  → {table} {period}: {len(df)} 行")
    return out


def fetch_reference(client, store: ParquetStore, verbose: bool = True) -> None:
    """拉取无日期维度的参考表（股票列表等）。"""
    if store.read_single("stock_basic").empty:
        if verbose:
            print("  → 股票基础信息 (stock_basic)")
        df = client.query("stock_basic", exchange="", list_status="L",
                          fields="ts_code,symbol,name,area,industry,market,list_date")
        if not df.empty:
            store.write_single("stock_basic", df)


@dataclass
class FetchResult:
    table: str
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    failed_dates: List[str] = field(default_factory=list)


def _pull_one(client, spec: TableSpec, date: str, retries: int = 3,
              expect: int = None) -> pd.DataFrame:
    """拉取单个日期；行数不足时退避重试。空结果可能是上游未入库。"""
    last: pd.DataFrame = pd.DataFrame()
    for attempt in range(retries):
        if spec.codes:
            frames = []
            for code in spec.codes:
                part = client.query(spec.api, ts_code=code, **{spec.date_kwarg: date})
                if not part.empty:
                    frames.append(part)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            df = client.query(spec.api, **{spec.date_kwarg: date})

        last = df
        if len(df) >= (expect if expect is not None else spec.expect_rows):
            return df
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    return last


def _adaptive_threshold(store: ParquetStore, spec: TableSpec) -> int:
    """行数阈值必须随市场规模变化。

    踩过的坑：硬编码 expect_rows=4000 是按 2025 年（5400 只）定的，
    2016 年只有 2764 只，导致每一个历史日期都被判成"上游未入库"，
    重试 3 次后放弃 —— 1677 个日期全废，且看不出任何进展。
    这里改成跟最近一次成功拉取的行数比对（允许早期市场只有 35%）。
    """
    floor = spec.expect_rows
    existing = store.list_dates(spec.name)
    if not existing:
        return floor
    try:
        recent = store.read_dated(spec.name, [existing[-1]])
    except Exception:                    # noqa: BLE001
        return floor
    if recent.empty:
        return floor
    if floor <= 0:
        return 0            # 阈值 0 表示"允许空结果"，不要自作聪明加限制
    if spec.codes:
        return len(spec.codes)      # 固定代码集：行数应等于代码数
    # 小表（如指数只有 3~4 行）按比例算；大盘表有 0.35 的容差和 200 行下限
    if floor <= 5:
        return max(1, int(len(recent) * 0.75))
    return min(floor, max(200, int(len(recent) * 0.35)))


def fetch_table(client, store: ParquetStore, spec: TableSpec,
                dates: Sequence[str], verbose: bool = True) -> FetchResult:
    result = FetchResult(table=spec.name)
    expect = _adaptive_threshold(store, spec)
    for date in dates:
        if store.has_dated(spec.name, date):
            result.skipped += 1
            continue
        try:
            df = _pull_one(client, spec, date, expect=expect)
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.failed_dates.append(date)
            if verbose:
                print(f"    ✗ {spec.name} {date}: {exc}")
            continue

        if len(df) < expect:
            result.failed += 1
            result.failed_dates.append(date)
            if verbose:
                print(f"    ⚠ {spec.name} {date}: 仅 {len(df)} 行"
                      f"（期望 ≥{expect}），判定未入库")
            continue

        if spec.keep:
            cols = [c for c in spec.keep if c in df.columns]
            if len(cols) >= max(3, len(spec.keep) // 2):
                df = df[cols]
        store.write_dated(spec.name, date, df)
        store.set_watermark(spec.name, date)
        result.fetched += 1
        result.rows += len(df)
    return result


def resolve_target(store: ParquetStore, target: Optional[str]) -> str:
    """把用户给的日期对齐到最近的已开市日。"""
    if target:
        compact = target.replace("-", "")[:8]
        resolved = calendar.latest_open_date(store, on_or_before=compact)
        if not resolved:
            raise ValueError(f"{target} 及之前没有开市日，检查 trade_cal 是否已拉取")
        return resolved
    resolved = calendar.latest_open_date(store, on_or_before=calendar.today_compact())
    if not resolved:
        raise ValueError("trade_cal 为空，请先执行 fetch")
    return resolved


def fetch(client, store: ParquetStore, target: Optional[str] = None,
          backfill_days: int = 0, backfill_years: int = 0,
          tables: Optional[Sequence[str]] = None,
          all_tables: bool = False,
          with_dividend: bool = False,
          dividend_limit: int = 400,
          dividend_all: bool = False,
          verbose: bool = True) -> Dict[str, FetchResult]:
    """增量拉取。backfill_* 指定回补范围。"""
    calendar.ensure_trade_cal(client, store, start_year=2008, force=False)
    fetch_reference(client, store, verbose=verbose)
    target_date = resolve_target(store, target)

    if backfill_years:
        start = calendar.shift_years(target_date, -backfill_years)
    elif backfill_days:
        start = calendar.shift_days(target_date, -backfill_days * 2)
    else:
        start = None

    dates = calendar.open_dates(store, start=start, end=target_date)

    if tables:
        specs = [TABLE_MAP[t] for t in tables]
    elif all_tables:
        specs = list(TABLES)
    else:
        specs = [TABLE_MAP[t] for t in DEFAULT_TABLES]

    # 每张表各自算要拉哪些日期。原来的写法有两个坑：
    #   1) floor = min(watermarks)：只要有一张表水位线旧，所有表都从头重拉；
    #   2) 缺口扫描扫 750 个交易日，而 limit_list_d 在 2016-2019 返回 0 行是
    #      正常的，这些日期永远"缺失"、每次都被重试——增量跑成了全量。
    selected_by_spec: Dict[str, List[str]] = {}
    if backfill_years or backfill_days:
        for spec in specs:
            selected_by_spec[spec.name] = list(dates)
        selected = dates
    else:
        gap_window = dates[-GAP_SCAN_DAYS:]        # 只补最近一个月的缺口
        for spec in specs:
            wm = store.get_watermark(spec.name)
            after = [d for d in dates if wm is None or d > wm]
            existing = set(store.list_dates(spec.name))
            gaps = [d for d in gap_window if d not in existing]
            spec_dates = sorted(set(after) | set(gaps))
            # 已经入库的跳过，免得把「水位线之后全部重算」变成全量
            spec_dates = [d for d in spec_dates if d not in existing] or (
                [target_date] if target_date not in existing else [])
            selected_by_spec[spec.name] = spec_dates
        all_dates = sorted({d for v in selected_by_spec.values() for d in v})
        selected = all_dates

    if verbose:
        print(f"  目标交易日: {target_date}")
        print(f"  待检查日期: {selected[0]} ~ {selected[-1]}（{len(selected)} 个）")
        print(f"  数据表: {', '.join(s.name for s in specs)}")

    results: Dict[str, FetchResult] = {}
    for spec in specs:
        if verbose:
            print(f"  → {spec.label or spec.name} ({spec.api})")
        spec_dates = selected_by_spec.get(spec.name, selected)
        if not spec_dates:
            results[spec.name] = FetchResult(table=spec.name)
            continue
        results[spec.name] = fetch_table(client, store, spec, spec_dates, verbose=verbose)

    if verbose:
        print("  → 财务指标（按年报，一次一报告期）")
    fin = fetch_financials(client, store, target_date, years=12, verbose=verbose)
    if verbose and fin["fetched"]:
        print(f"    新拉 {fin['fetched']} 期，跳过 {fin['skipped']} 期"
              f"，失败 {fin['failed']} 期")

    if verbose:
        print("  → 分红历史（按需缓存，只取高股息标的）")
    try:
        fetch_cashflow_income(client, store, target_date, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  ✗ 现金流/利润表拉取失败：{exc}")
    # 分红是逐股接口（一只一次请求），全市场跑一遍要好几分钟。
    # 它只需要每周更新一次，所以默认跳过，由 --with-dividend 触发。
    if with_dividend:
        fetch_dividend_history(client, store, target_date,
                               # dividend_all=True 时不按股息率筛，
                               # 把全市场都缓存一遍（一次性，约 5500 次请求）
                               dv_threshold=0.0 if dividend_all else 3.0,
                               limit=dividend_limit,
                               verbose=verbose)
    elif verbose:
        print("  - 分红历史：跳过（每周单独更新，加 --with-dividend）")

    if verbose:
        print("  → 股吧情绪（公开列表页）")
    fetch_guba(store, target_date, verbose=verbose)

    return results


def summarize(results: Dict[str, FetchResult]) -> str:
    lines = []
    total_req = 0
    for name, r in results.items():
        total_req += r.fetched
        lines.append(
            f"  {name:16s} 新拉 {r.fetched:4d}  跳过 {r.skipped:4d}  "
            f"失败 {r.failed:3d}  行数 {r.rows:8d}"
        )
    lines.append(f"  合计发起 {total_req} 次日级请求")
    return "\n".join(lines)


# ---------------- 轻量抓取（托管用）----------------
# 定投看板只需要四类数据，全部加起来不到 100MB，可以塞进 GitHub Actions 缓存：
#   1. daily_basic：估值分位用的月度采样（约 121 天）+ 今天
#   2. index_broad：极端信号用的近 250 天
#   3. fund_share：宽基 ETF 份额，近 250 天
#   4. etf_daily / etf_div：持仓标的的价格与分红（各一只一次请求）
BROAD_ETFS = ("510300.SH", "159919.SZ", "510310.SH", "510050.SH",
              "510500.SH", "512500.SH", "512100.SH", "159845.SZ",
              "588000.SH", "159915.SZ")
BROAD_INDICES = ("000001.SH", "000016.SH", "000300.SH", "000905.SH",
                 "000852.SH", "000922.CSI", "000932.SH", "399001.SZ",
                 "399006.SZ", "000688.SH")
HOLDING_ETFS = ("513530.SH", "563020.SH")


def _month_end_samples(dates: Sequence[str]) -> List[str]:
    """每月最后一个交易日，用于估值分位的历史采样。"""
    buckets: Dict[str, str] = {}
    for d in dates:
        buckets[d[:6]] = d
    return sorted(buckets.values())


def fetch_lite(client, store: ParquetStore, target_date: str,
               history_days: int = 2600, verbose: bool = True) -> Dict[str, int]:
    """只拉定投看板需要的数据。首次约 620 次请求，日常增量 3~4 次。"""
    import pandas as pd

    calendar.ensure_trade_cal(client, store, start_year=2008, force=False)
    stats: Dict[str, int] = {}

    all_dates = calendar.open_dates(store, end=target_date)
    recent = calendar.recent_open_dates(store, 260, end=target_date)
    samples = _month_end_samples(calendar.recent_open_dates(
        store, history_days, end=target_date))

    def _put(table: str, date: str, frame) -> None:
        if frame is None or frame.empty:
            return
        store.write_dated(table, date, frame)
        stats[table] = stats.get(table, 0) + 1

    # 1) daily_basic：月度采样 + 今天
    need = [d for d in samples + [target_date]
            if not store.has_dated("daily_basic", d)]
    if verbose:
        print(f"  → 估值采样 {len(need)} 天", flush=True)
    for d in need:
        _put("daily_basic", d,
             _safe_query(client, "daily_basic", trade_date=d))

    # 2) index_broad：近 250 天
    need = [d for d in recent if not store.has_dated("index_broad", d)]
    if verbose:
        print(f"  → 宽基指数 {len(need)} 天", flush=True)
    for d in need:
        df = _safe_query(client, "index_daily", trade_date=d)
        if df is not None and not df.empty:
            _put("index_broad", d, df[df["ts_code"].isin(BROAD_INDICES)])

    # 3) fund_share：近 250 天
    need = [d for d in recent if not store.has_dated("fund_share", d)]
    if verbose:
        print(f"  → ETF 份额 {len(need)} 天", flush=True)
    for d in need:
        df = _safe_query(client, "fund_share", trade_date=d)
        if df is not None and not df.empty:
            _put("fund_share", d, df[df["ts_code"].isin(BROAD_ETFS)])

    # 4) 持仓 ETF：价格 + 分红（一次请求一只，全覆盖）
    prices, divs = [], []
    for code in HOLDING_ETFS:
        p = _safe_query(client, "fund_daily", ts_code=code,
                        start_date="20190101", end_date=target_date)
        if p is not None and not p.empty:
            prices.append(p)
        d = _safe_query(client, "fund_div", ts_code=code)
        if d is not None and not d.empty:
            d = d[d.get("div_proc") == "实施"].drop_duplicates(
                subset=["ts_code", "ex_date", "div_cash"])
            divs.append(d)
    if prices:
        store.write_single("etf_daily", pd.concat(prices, ignore_index=True))
        stats["etf_daily"] = len(prices)
    if divs:
        store.write_single("etf_div", pd.concat(divs, ignore_index=True))
        stats["etf_div"] = len(divs)

    if verbose:
        print("  完成：" + "，".join(f"{k} +{v}" for k, v in stats.items()))
    return stats


def _safe_query(client, api: str, **kwargs):
    try:
        return client.query(api, **kwargs)
    except Exception:                           # noqa: BLE001
        return None
