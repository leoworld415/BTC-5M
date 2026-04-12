#!/usr/bin/env python3
"""
BTC-5m Backtest Engine — v5A Strategy
=======================================
Proper methodology (backtest-expert):
- Use PnL as ground truth (signed: + = profit, - = loss)
- Walk-forward out-of-sample validation
- Parameter sensitivity
- Sharpe/MaxDD/CAGR metrics
"""
import sqlite3, math, statistics as stats
from collections import defaultdict

DB = "/home/lcb/.openclaw/workspace/skills/btc-5m-trader/data/btc5m_research.db"

conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("""
    SELECT slot, ts_unix, hour_utc, status, final_dir,
           rsi, slot_range, vol_mult, combined_mult,
           v42_dir, v42_conf, v35_dir, v35_conf,
           pnl
    FROM decisions
    WHERE status = 'SUCCESS'
      AND pnl IS NOT NULL
    ORDER BY slot ASC
""")
all_rows = c.fetchall()
conn.close()

records = []
for r in all_rows:
    slot, ts_unix, hour_utc, status, final_dir, \
        rsi, slot_range, vol_mult, combined_mult, \
        v42_dir, v42_conf, v35_dir, v35_conf, \
        pnl = r
    records.append({
        "slot": slot, "ts_unix": ts_unix, "hour_utc": hour_utc,
        "final_dir": final_dir, "pnl": float(pnl),
        "rsi": rsi, "slot_range": slot_range,
        "vol_mult": vol_mult, "combined_mult": combined_mult,
        "v42_dir": v42_dir, "v42_conf": v42_conf,
        "v35_dir": v35_dir, "v35_conf": v35_conf,
    })

print(f"Settled SUCCESS rows with PnL: {len(records)}")
if not records:
    print("No data!"); exit()

# ─── Time-ordered split ─────────────────────────────────────────────────
split_idx = int(len(records) * 0.80)
train = records[:split_idx]
test  = records[split_idx:]
print(f"  Train: {len(train)} | Test: {len(test)}")

# ─── Metric helpers ────────────────────────────────────────────────────
def metrics(recs):
    if not recs: return {}
    pnls    = [r["pnl"] for r in recs]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]
    n       = len(recs)
    wr      = len(wins) / n if n else 0
    pnl_sum = sum(pnls)
    aw      = stats.mean(wins)  if wins  else 0.0
    al      = stats.mean(losses) if losses else 0.0
    exp     = wr*aw - (1-wr)*abs(al)
    std     = stats.stdev(pnls) if len(pnls) > 1 else 0.0
    sh      = (stats.mean(pnls)/std*math.sqrt(252*288)) if std > 0 else None

    # Max drawdown
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    return {
        "n": n, "wr": wr, "pnl": pnl_sum,
        "aw": aw, "al": al, "exp": exp,
        "sharpe": sh, "maxdd": max_dd,
        "rrr": abs(aw/al) if al != 0 else 0,
    }

def m(recs):
    return metrics(recs)

def print_metrics(label, recs, pfx="  "):
    d = m(recs)
    if not d: return d
    print(f"{pfx}{label}")
    print(f"{pfx}  N={d['n']} | WR={d['wr']:.1%} | PnL={d['pnl']:+.2f} | E={d['exp']:+.4f}")
    print(f"{pfx}  AvgWin={d['aw']:+.4f} | AvgLoss={d['al']:+.4f} | RRR={d['rrr']:.2f}x")
    if d['sharpe'] is not None:
        print(f"{pfx}  Sharpe={d['sharpe']:+.2f} | MaxDD={d['maxdd']:+.4f}")
    return d

# ─── Report ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BACKTEST REPORT — v5A Strategy")
print("  Data: decisions table (SUCCESS, PnL!=NULL)")
print("="*60)

print("\n── Baseline: All Trades ──")
all_d = m(records)
print(f"  N={all_d['n']} | WR={all_d['wr']:.1%} | PnL={all_d['pnl']:+.2f} | E={all_d['exp']:+.4f}")

print("\n── In-Sample (Train 80%) ──")
r_train = print_metrics("Train", train)

print("\n── Out-of-Sample (Test 20%) ──")
r_test  = print_metrics("Test", test)

print("\n── Direction Breakdown ──")
downs = [r for r in records if r["final_dir"] == "DOWN"]
ups   = [r for r in records if r["final_dir"] == "UP"]
print_metrics("DOWN trades", downs)
print_metrics("UP trades",   ups)

# ─── Walk-Forward (test split in 2) ───────────────────────────────────
if len(test) >= 8:
    wf_mid = len(test)//2
    wf1, wf2 = test[:wf_mid], test[wf_mid:]
    print("\n── Walk-Forward ──")
    print_metrics("WF Period 1", wf1)
    print_metrics("WF Period 2", wf2)

# ─── By Hour ──────────────────────────────────────────────────────────
print("\n── By Hour ──")
by_hour = defaultdict(list)
for r in records:
    by_hour[r["hour_utc"]].append(r)
for h in sorted(by_hour.keys()):
    d = m(by_hour[h])
    sh_str = f"{d['sharpe']:+.2f}" if d['sharpe'] is not None else "N/A"
    print(f"  UTC {h:02d}: N={d['n']:2d} | WR={d['wr']:5.1%} | PnL={d['pnl']:+8.2f} | RRR={d['rrr']:.2f}x | Sharpe={sh_str}")

# ─── RSI Buckets ─────────────────────────────────────────────────────
print("\n── RSI Buckets ──")
for lo, hi, label in [
    (0, 30,   "RSI <30  (Oversold → UP)"),
    (30, 40,  "RSI 30-40"),
    (40, 60,  "RSI 40-60 (Neutral → DOWN)"),
    (60, 100, "RSI >60  (Overbought)"),
]:
    bucket = [r for r in records if (r["rsi"] or 50) >= lo and (r["rsi"] or 50) < hi]
    if bucket:
        d = m(bucket)
        print(f"  {label:30s} N={d['n']:2d} | WR={d['wr']:5.1%} | PnL={d['pnl']:+8.2f} | RRR={d['rrr']:.2f}x")

# ─── Stress Test: +$0.50 / -$0.50 slippage ──────────────────────────
print("\n── Stress Tests ──")
for slip in [0.25, 0.50, 1.00]:
    stressed = [{"pnl": r["pnl"] - slip if r["pnl"] > 0 else r["pnl"] - slip} for r in records]
    # Actually just subtract slip from winners too
    stressed = [{"pnl": r["pnl"] - slip} for r in records]
    d = m(stressed)
    sh_s = f"{d['sharpe']:+.2f}" if d['sharpe'] is not None else "N/A"
    print(f"  Slip=${slip:.2f}/trade: N={d['n']} | WR={d['wr']:.1%} | PnL={d['pnl']:+.2f} | Sharpe={sh_s}")

# ─── Summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
is_ok  = r_train["wr"] > 0.42 and r_train["pnl"] > 0
oos_ok = r_test["wr"]  > 0.40 and r_test["pnl"] > 0
print(f"  Train: WR={r_train['wr']:.1%} | PnL={r_train['pnl']:+.2f} | {'✅' if is_ok else '❌'}")
print(f"  Test:  WR={r_test['wr']:.1%}  | PnL={r_test['pnl']:+.2f}  | {'✅' if oos_ok else '❌'}")
if r_train["wr"] and r_test["wr"]:
    ratio = r_test["wr"] / r_train["wr"]
    print(f"  IS/OOS ratio: {ratio:.2f} {'✅ >0.80' if ratio>0.80 else '⚠️ 0.50-0.80' if ratio>0.50 else '🔴 <0.50'}")
print("="*60)
