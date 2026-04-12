#!/usr/bin/env python3
"""
MEXC P1 — 15-Minute Paper Trader
==================================
MEXC紙質交易，窗口從5分鐘改為15分鐘。

為什麼15分鐘：
  - 5分鐘太短：entry≈exit，幾乎無波動空間
  - 15分鐘：足夠BTC走 $50-200，有實質PnL

改動：
  - SLOT_DURATION: 300 → 900 (15分鐘)
  - klines: interval=5m → interval=15m
  - HOLD threshold: 10% → 8% (15分鐘波幅更大，門檻略緊)
  - Position sizing: 8% balance → 15% (15分鐘有更多時間，倉位略大)

其他全部保留 ConservativeV1 信號邏輯。
"""
import os, json, time, requests, pickle
from datetime import datetime, timezone
import numpy as np

BASE = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader")
LOG  = os.path.join(BASE, "logs/mexc_p1_paper_trades_15m.jsonl")
HELD = os.path.join(BASE, "logs/mexc_held_position_15m.json")
TRACK = os.path.join(BASE, "logs/traded_slots_15m.json")

BINANCE = "https://api.binance.com"

# ── 15分鐘參數 ────────────────────────────────────────────────────────────────
SLOT_DURATION      = 900       # 15分鐘
MIN_ENTRY_SECS     = 120       # 至少2分鐘緩衝
LEVERAGE           = 3
BALANCE            = 100.0
POSITION_PCT       = 0.08      # 8% balance per trade (略高於5m版)
HOLD_LOSS_THRESH   = 0.08      # 8% loss → hold
PROFIT_EXIT_THRESH = 0.001      # 任何利潤 → exit

# ML
MODEL_V17 = os.path.join(BASE, "data/ml_model_v17.pkl")
SCALER_V17 = os.path.join(BASE, "data/scaler_v17.pkl")
MODEL_V35 = os.path.join(BASE, "data/ml_model_v35.pkl")
SCALER_V35 = os.path.join(BASE, "data/scaler_v35.pkl")

_mv17 = _sv17 = _mv35 = _sv35 = None

# ── Indicators ────────────────────────────────────────────────────────────────
def ema(prices, n):
    if len(prices) < n: return None
    k = 2.0/(n+1); e = float(prices[0])
    for p in prices[1:]: e = p*k + e*(1-k)
    return e

def rsi(closes, p=14):
    if len(closes) < p+1: return 50.0
    arr = np.asarray(closes[-p-1:], dtype=np.float64).ravel()
    d = np.diff(arr)
    g = float(np.sum(d[d>0])); l = abs(float(np.sum(d[d<0])))
    return 100-100/(1+g/l) if l else 100.0

def macd_hist(closes):
    if len(closes) < 26: return 0.0
    e12 = ema(closes, 12); e26 = ema(closes, 26); e9 = ema(closes, 9)
    if None in (e12, e26, e9): return 0.0
    return float((e12-e26)-e9)

# ── ML ──────────────────────────────────────────────────────────────────────
def load_v17():
    global _mv17
    if _mv17 is None and os.path.exists(MODEL_V17):
        with open(MODEL_V17,"rb") as f: _mv17 = pickle.load(f)
    return _mv17

def load_sv17():
    global _sv17
    if _sv17 is None and os.path.exists(SCALER_V17):
        with open(SCALER_V17,"rb") as f: _sv17 = pickle.load(f)
    return _sv17

def load_v35():
    global _mv35
    if _mv35 is None and os.path.exists(MODEL_V35):
        with open(MODEL_V35,"rb") as f: _mv35 = pickle.load(f)
    return _mv35

def load_sv35():
    global _sv35
    if _sv35 is None and os.path.exists(SCALER_V35):
        with open(SCALER_V35,"rb") as f: _sv35 = pickle.load(f)
    return _sv35

def pred_v17(rsi_v, macd_v, ema_x, vol_d, mom5, mom1, stoch, bb, atr, willr, rsi_f, taker, taker_avg):
    m = load_v17(); s = load_sv17()
    if m is None or s is None: return "N/A", 0.5
    X = np.array([[rsi_v,macd_v,ema_x,vol_d,mom5,mom1,stoch,bb,atr,willr,rsi_f,taker,taker_avg,0.0,0.0]], dtype=np.float32)
    p = m.predict_proba(s.transform(X))[0]
    return ("UP" if p[1]>=p[0] else "DOWN", max(float(p[1]),float(p[0])))

def pred_v35(rsi_v, macd_v, ema_x, vol_d, stoch, bb, atr, willr, mom1, mom15, taker, hour):
    m = load_v35(); s = load_sv35()
    if m is None or s is None: return "N/A", 0.5
    X = np.array([[rsi_v,macd_v,ema_x,vol_d,stoch,bb,atr,willr,mom1,mom15,taker,0.0,hour]], dtype=np.float32)
    p = m.predict_proba(s.transform(X))[0]
    return ("UP" if p[1]>=p[0] else "DOWN", max(float(p[1]),float(p[0])))

# ── Data Fetchers ────────────────────────────────────────────────────────────
def klines_1m(limit=100):
    try:
        r = requests.get(f"{BINANCE}/api/v3/klines",
                        params={"symbol":"BTCUSDT","interval":"1m","limit":limit}, timeout=8)
        data = r.json()
        return [[float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in data]
    except: return []

def klines_15m(limit=60):
    try:
        r = requests.get(f"{BINANCE}/api/v3/klines",
                        params={"symbol":"BTCUSDT","interval":"15m","limit":limit}, timeout=8)
        data = r.json()
        return [[float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in data]
    except: return []

def live_price():
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/price",
                        params={"symbol":"BTCUSDT"}, timeout=8)
        return float(r.json()["price"])
    except: return None

def bid_ask():
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/bookTicker",
                        params={"symbol":"BTCUSDT"}, timeout=8)
        d = r.json()
        return float(d["bidPrice"]), float(d["askPrice"])
    except: return None, None

# ── PnL ─────────────────────────────────────────────────────────────────────
def calc_pnl(entry, exit_px, direction, size):
    if direction == "LONG":
        ret = (exit_px - entry) / entry
    else:
        ret = (entry - exit_px) / entry
    return size * LEVERAGE * ret, ret

# ── Main ────────────────────────────────────────────────────────────────────
def run():
    now = int(time.time())
    next_slot = (now // SLOT_DURATION) * SLOT_DURATION + SLOT_DURATION
    current_slot = (now // SLOT_DURATION) * SLOT_DURATION
    secs_left = next_slot - now

    print(f"[15m] now={now} slot={current_slot} next={next_slot} left={secs_left}s")

    if secs_left < MIN_ENTRY_SECS:
        print(f"  ⏳ too close to close ({secs_left}s < {MIN_ENTRY_SECS}s), skip")
        return

    # ── Data ────────────────────────────────────────────────────────────────
    k1 = klines_1m(100)
    k15 = klines_15m(60)
    if not k1 or not k15:
        print("  ❌ no data"); return

    bp, ap = bid_ask()
    lp = live_price()
    cp = bp or lp or k1[-1][4]

    closes_1m = [float(x[4]) for x in k1]
    closes_15m = [float(x[4]) for x in k15]

    # Indicators
    rsi_v  = rsi(closes_1m, 14)
    macd_v = macd_hist(closes_1m)
    ema9   = ema(closes_1m, 9) or 0
    ema21  = ema(closes_1m, 21) or 0
    ema_x  = (ema9 - ema21) / ema21 * 100 if ema21 else 0
    mom5   = (closes_1m[-1] - closes_1m[-6]) / closes_1m[-6] * 100 if len(closes_1m)>=6 else 0
    mom1   = (closes_1m[-1] - closes_1m[-2]) / closes_1m[-2] * 100
    mom15  = (closes_1m[-1] - closes_1m[-16]) / closes_1m[-16] * 100 if len(closes_1m)>=16 else mom5
    vol_d  = (closes_1m[-1] - closes_1m[0]) / closes_1m[0] * 100

    # Stoch
    lo14 = min(float(x[3]) for x in k1[-14:])
    hi14 = max(float(x[2]) for x in k1[-14:])
    stoch_v = 100*(closes_1m[-1]-lo14)/(hi14-lo14) if hi14!=lo14 else 50.0

    # ATR(14) on 15m
    trs = [max(float(k15[i][2])-float(k15[i][3]),
                abs(float(k15[i][2])-float(k15[i-1][4])),
                abs(float(k15[i][3])-float(k15[i-1][4])))
           for i in range(1,len(k15))]
    atr_v = float(np.mean(trs[-14:])) if len(trs)>=14 else 0

    # BB
    bb_mu  = np.mean(closes_1m[-20:])
    bb_std = np.std(closes_1m[-20:])
    bb_pos = (closes_1m[-1]-(bb_mu-2*bb_std))/(4*bb_std+1e-9) if bb_std else 0.5

    taker = 1.0
    hour  = now // 3600 % 24

    v35_dir, v35_conf = pred_v35(rsi_v, macd_v, ema_x, vol_d, stoch_v, bb_pos,
                                  atr_v/closes_1m[-1], -100*(hi14-closes_1m[-1])/(hi14-lo14+1e-9),
                                  mom1, mom15, taker, hour)
    v17_dir, v17_conf = pred_v17(rsi_v, macd_v, ema_x, vol_d, mom5, mom1, stoch_v, bb_pos,
                                  atr_v/closes_1m[-1],
                                  -100*(hi14-closes_1m[-1])/(hi14-lo14+1e-9),
                                  rsi_v, taker, taker)

    # 2/3 vote
    votes = [d for d in [v35_dir, v17_dir] if d in ("UP","DOWN")]
    if votes.count("UP") > votes.count("DOWN"):
        signal_dir = "UP"
    elif votes.count("DOWN") > votes.count("UP"):
        signal_dir = "DOWN"
    else:
        signal_dir = "WAIT"

    signal_conf = max(v35_conf, v17_conf) if signal_dir != "WAIT" else 0.5

    # ── Slot dedup ─────────────────────────────────────────────────────────
    traded = set(json.load(open(TRACK)) if os.path.exists(TRACK) else [])
    if current_slot in traded:
        print(f"  ⏳ already traded slot {current_slot}")
        return
    traded.add(current_slot)
    json.dump(list(traded), open(TRACK,"w"))

    # ── Held position ──────────────────────────────────────────────────────
    held = json.load(open(HELD)) if os.path.exists(HELD) else None
    pos_size = BALANCE * POSITION_PCT

    entry_px = 0.0; exit_px = 0.0; pnl_val = 0.0
    position = "WAIT"; note = ""; new_held = None; closed_15m = False

    if held:
        held_dir  = held["direction"]
        held_ent  = held["entry_price"]
        held_slot = held["slot"]
        held_sz   = held.get("position_size", pos_size)

        cur_pnl, cur_ret = calc_pnl(held_ent, cp, held_dir, held_sz)
        ret_pct = abs(cur_ret) * 100

        print(f"  📌 Held: {held_dir} @{held_ent:.2f} → {cp:.2f} | PnL={cur_pnl:+.2f} ({ret_pct:.1f}%)")

        if cur_pnl > 0:
            exit_px = cp
            pnl_val, _ = calc_pnl(held_ent, exit_px, held_dir, held_sz)
            position = f"EXIT_{held_dir}"; note = f"Held→exit profit ${pnl_val:+.2f}"
            print(f"  ✅ EXIT {held_dir} profit ${pnl_val:+.2f}")
            new_held = None
        elif cur_ret <= -HOLD_LOSS_THRESH:
            exit_px = cp
            pnl_val, _ = calc_pnl(held_ent, exit_px, held_dir, held_sz)
            position = f"LOSS_CLOSE_{held_dir}"; note = f"Hold→force close ${pnl_val:+.2f}"
            print(f"  ❌ FORCE CLOSE {held_dir} loss ${pnl_val:+.2f} ({ret_pct:.1f}% > 8%)")
            new_held = None
        else:
            position = f"HOLDING_{held_dir}"
            entry_px = held_ent
            new_held = {
                "direction": held_dir, "entry_price": held_ent,
                "slot": held_slot, "position_size": held_sz,
                "opened_at": held.get("opened_at", datetime.now(timezone.utc).isoformat()),
                "hold_count": held.get("hold_count", 0) + 1,
                "worst_pnl": min(held.get("worst_pnl", 0), cur_pnl),
            }
            note = f"Holding {held_dir} loss={cur_pnl:+.2f} ({ret_pct:.1f}% < 8%)"
            print(f"  ⏸️ HOLD {held_dir} loss={cur_pnl:+.2f}, wait")
    else:
        # ── New entry ──────────────────────────────────────────────────────
        if signal_dir == "WAIT":
            print(f"  ⏳ no signal (WAIT)")
        else:
            # Check if 15m candle for this slot has CLOSED
            slot_end_ms = (current_slot + SLOT_DURATION) * 1000
            try:
                r = requests.get(f"{BINANCE}/api/v3/klines",
                               params={"symbol":"BTCUSDT","interval":"15m",
                                       "startTime": current_slot*1000,
                                       "endTime": slot_end_ms, "limit":1}, timeout=8)
                cdl = r.json()
                if cdl and len(cdl) > 0:
                    ct = int(cdl[0][6])
                    if ct < now * 1000:
                        # Candle HAS closed - can calculate real P&L
                        entry_px = float(cdl[0][0])  # open price of the 15m candle
                        exit_px = float(cdl[0][4])  # close price of the 15m candle
                        closed_15m = True
                        position = "LONG" if signal_dir == "UP" else "SHORT"
                    else:
                        # Candle not closed yet - wait for next slot
                        print(f"  ⏳ 15m candle not closed yet, skipping")
                        closed_15m = False
                        result_recorded = {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "slot": current_slot,
                            "signal_dir": signal_dir,
                            "position": "SKIP_WAIT_CANDLE",
                            "result": "-",
                            "pnl": 0.0,
                            "note": "candle not closed",
                            "closed_15m": False
                        }
                        with open(TRADES, "a") as f:
                            f.write(json.dumps(result_recorded, default=str) + "\n")
                        return  # Exit run() for this slot
                else:
                    print(f"  ⏳ no candle data, skip")
                    return  # Exit run() for this slot
            except Exception as e:
                print(f"  ⚠️ candle fetch error: {e}, skip")
                return  # Exit run() for this slot

            if not closed_15m:
                new_held = None
                closed_15m = False
                ret_pct = 0.0
            else:
                ret_pct = abs((exit_px - entry_px) / entry_px) * 100

            if cur_pnl > 0:
                pnl_val = cur_pnl
                note = f"15m WIN ${pnl_val:+.2f}"
                print(f"  ✅ WIN {position} @ ${entry_px:.2f}→${exit_px:.2f} PnL=${pnl_val:+.2f}")
                new_held = None
            elif cur_ret <= -HOLD_LOSS_THRESH:
                pnl_val = cur_pnl
                new_held = {
                    "direction": position, "entry_price": entry_px,
                    "slot": current_slot, "position_size": pos_size,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "hold_count": 0, "worst_pnl": pnl_val,
                }
                note = f"Hold: loss={pnl_val:.2f} ({ret_pct:.1f}%)"
                print(f"  ⏸️ HOLD {position} loss={pnl_val:.2f} ({ret_pct:.1f}%)")
            else:
                pnl_val = cur_pnl
                note = f"15m settle {position} PnL=${pnl_val:+.2f}"
                print(f"  ⚖️ {position} @ ${entry_px:.2f}→${exit_px:.2f} PnL=${pnl_val:+.2f}")
                new_held = None

    # ── Save state ──────────────────────────────────────────────────────────
    if new_held:
        json.dump(new_held, open(HELD,"w"), default=str)
    elif os.path.exists(HELD):
        os.remove(HELD)

    # ── Log ────────────────────────────────────────────────────────────────
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slot": current_slot,
        "price": cp, "bid": bp, "ask": ap,
        "direction": position,
        "signal_dir": signal_dir,
        "v17_dir": v17_dir, "v17_conf": round(v17_conf,4),
        "v35_dir": v35_dir, "v35_conf": round(v35_conf,4),
        "entry_price": entry_px,
        "exit_price": exit_px,
        "closed_15m": closed_15m,
        "pnl": round(pnl_val, 4) if pnl_val else 0.0,
        "note": note,
        "held": new_held,
    }
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG,"a") as f:
        f.write(json.dumps(log, default=str)+"\n")

    emoji = {"LONG":"🟢","SHORT":"🔴","WAIT":"⚪",
              "EXIT_LONG":"✅🟢","EXIT_SHORT":"✅🔴",
              "LOSS_CLOSE_LONG":"❌🟢","LOSS_CLOSE_SHORT":"❌🔴",
              "HOLDING_LONG":"⏸️🟢","HOLDING_SHORT":"⏸️🔴"}.get(position,"⚪")
    print(f"  {emoji} {position} @ ${cp:.2f} | PnL: {pnl_val:+.4f} | {note}")

if __name__ == "__main__":
    run()
