"""银行三因子多窗口回测。

结论（2017-06 ~ 2026-03，106 个月 × 42 只银行，rank IC）：

  因子           IC5日    IC20日   IC60日   IC120日
  value_gap     -0.003   0.027    0.064    0.088
                  (t-0.1) (t1.1)  (t2.6)   (t4.3)
  yr_position    0.076    0.123    0.189     0.235
                  (t2.4)  (t3.7)  (t6.2)   (t8.6)
  chg_3m         0.003    0.026    0.066    0.104
                  (t0.1)  (t0.8)  (t2.1)   (t3.4)

三个因子在长窗口（60/120 日）全部显著，而且符号全部为正：
位置越高、估值越便宜、近期涨得越多 → 未来收益越好。

按综合分五分组，未来 120 日收益：
  第1组(最低) -0.50% / 0.70% / 2.88% / 4.58% / 第5组(最高) 4.83%
单调，首尾差 5.3 个百分点。

注意：这是「银行之间横向排名」的效应，不是「银行整体该不该买」。
两者可以同时成立——银行板块整体在高位，但板块内部强者继续强。
"""
import glob, os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR
from src.data.store import ParquetStore
from src.kit import universe as uni, returns as ret
from src.data import calendar

WINDOWS = [5, 20, 60, 120]
store = ParquetStore(DATA_DIR); meta = uni.stock_meta(store)
banks = [c for c in meta.index if "银行" in str(meta.loc[c].get("industry", ""))]
vcfg = {r["ts_code"].strip(): (float(r["target_pe"]), float(r["target_div_yield"]))
        for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "bank_valuation.csv"), encoding="utf-8"))}

all_dates = calendar.recent_open_dates(store, 2400, end="20261027")
px = ret.total_return_frame(store, banks, all_dates).ffill()
idx = {d: i for i, d in enumerate(all_dates)}

fin = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob(f"{DATA_DIR}/parquet/fina_indicator/*.parquet"))], ignore_index=True)
fin = fin[fin.ts_code.isin(banks)][["ts_code","end_date","ann_date","eps"]].copy()
fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-","")
fin = fin.dropna(subset=["eps"]).sort_values("ann_date")

div = store.read_dated("dividend"); div = div[div.ts_code.isin(banks)].copy()
div = div[div["div_proc"].astype(str).str.strip()=="实施"]
div["ann"] = div.get("ann_date", div["end_date"]).astype(str).str.replace("-","")
div["cash"] = pd.to_numeric(div.get("cash_div"), errors="coerce").fillna(0.0)
div = div[div.cash > 0]

buckets = {}
for d in all_dates: buckets[d[:6]] = d
sample = sorted(buckets.values())

rows = []
for T in sample:
    if T not in px.index: continue
    i = idx[T]
    if i + max(WINDOWS) >= len(all_dates): continue
    for c in banks:
        if c not in px.columns: continue
        ser = px[c].loc[:T].dropna()
        if len(ser) < 130: continue
        price = float(ser.iloc[-1])
        if price <= 0: continue
        tail = ser.tail(250); lo, hi = float(tail.min()), float(tail.max())
        pos = (price-lo)/(hi-lo)*100 if hi>lo else np.nan
        chg3 = (price/float(ser.iloc[-63])-1)*100 if len(ser)>63 else np.nan
        f = fin[(fin.ts_code==c)&(fin.ann_date<=T)]; dd = div[(div.ts_code==c)&(div.ann<=T)]
        cfg = vcfg.get(c.split(".")[0]); vg = np.nan
        if cfg and not f.empty and not dd.empty:
            eps = float(f.iloc[-1].eps)
            my = dd["end_date"].astype(str).str[:4].max()
            dps = float(dd[dd["end_date"].astype(str).str[:4]==my].cash.sum())
            tp, td = cfg; cands=[]
            if eps>0 and tp>0: cands.append(eps*tp)
            if dps>0 and td>0: cands.append(dps/(td/100.0))
            if cands: vg = (min(cands)+dps)/price - 1
        rec = {"T":T,"code":c,"value_gap":vg,"yr_position":pos,"chg_3m":chg3}
        ok = True
        for w in WINDOWS:
            fw = all_dates[i+w]
            if fw not in px.index or np.isnan(px.loc[fw,c]): ok=False; break
            rec[f"fwd{w}"] = float(px.loc[fw,c])/price - 1
        if ok: rows.append(rec)

df = pd.DataFrame(rows)
_n=df["T"].nunique(); _lo=df["T"].min(); _hi=df["T"].max(); _nc=df.code.nunique()
print(f"样本 {len(df)} 条 · {_n} 个月 · {_nc} 只银行 · {_lo}~{_hi}\n")

def ic(d, factor, fwd):
    out=[]
    for _,g in d.groupby("T"):
        g=g[[factor,fwd]].dropna()
        if len(g)>=8: out.append(g[factor].rank().corr(g[fwd].rank()))
    s=pd.Series(out)
    return (s.mean(), s.std()/np.sqrt(len(s)) if len(s)>1 else np.nan, len(s))

print(f"{'因子':<12}" + "".join(f"{'IC'+str(w)+'日':>10}{'t':>7}" for w in WINDOWS))
for fac in ("value_gap","yr_position","chg_3m"):
    line=f"{fac:<12}"
    for w in WINDOWS:
        m,se,n = ic(df, fac, f"fwd{w}")
        line += f"{m:>10.3f}{(m/se if se and se>0 else 0):>7.1f}"
    print(line)

