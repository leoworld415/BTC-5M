#!/usr/bin/env python3
"""
MEXC_P1 — MEXC Futures Paper Trader (ConservativeV1 Signal)

Features:
1. Fetch real-time BTCUSDT prices from Binance APIs
2. Calculate ConservativeV1 signal (V4 score + v17 ML)
3. Simulate Long/Short MEXC Futures positions with 3x leverage
4. Hold positions across slots if loss < 10%; sell only when next slot is profitable
5. Log everything to mexc_p1_paper_trades.jsonl
6. Run as a cron job every 5 minutes
"""
import os, json, time, requests, pickle
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

# Paths
BASE = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader")
LOG = os.path.join(BASE, "logs/mexc_p1_paper_trades.jsonl")
HELD_STATE = os.path.join(BASE, "logs/mexc_held_position.json")
TRACKER = os.path.join(BASE, "logs/traded_slots.json")

# MEXC API
MEXC_HOST = "https://api.mexc.com"
BINANCE_HOST = "https://api.binance.com"

# Model Paths
MODEL_PATH_V17 = os.path.join(BASE, "data/ml_model_v17.pkl")
SCALER_PATH_V17 = os.path.join(BASE, "data/scaler_v17.pkl")
MODEL_PATH_V35 = os.path.join(BASE, "data/ml_model_v35.pkl")
SCALER_PATH_V35 = os.path.join(BASE, "data/scaler_v35.pkl")

# ── Strategy Parameters ────────────────────────────────────────────────────
LEVERAGE = 3
MIN_ENTRY_SECS = 90
SLOT_DURATION = 300
BALANCE = 100.0           # Paper balance in USD
POSITION_PCT = 0.05        # 5% of balance per position
CONTRACT_MULTIPLIER = 1.0  # 1 contract per $1 of position_size
HOLD_LOSS_THRESHOLD = 0.10  # 10% loss → hold instead of close
PROFIT_EXIT_THRESHOLD = 0.001  # ANY profit → exit (even 0.1%)

# ML Model globals
_mv17 = None; _sv17 = None; _mv35 = None; _sv35 = None

# ── Indicator Functions ────────────────────────────────────────────────────
def ema_calc(prices, period):
    if len(prices) < period: return None
    arr = np.asarray(prices, dtype=np.float64).ravel()
    k = 2.0 / (period + 1)
    e = float(arr[0])
    for p in arr[1:]: e = p * k + e * (1 - k)
    return e

def rsi_calc(closes, p=14):
    if len(closes) < p + 1: return 50.0
    arr = np.asarray(closes[-p-1:], dtype=np.float64).ravel()
    deltas = np.diff(arr)
    gain = float(np.sum(deltas[deltas > 0]))
    loss = abs(float(np.sum(deltas[deltas < 0])))
    if loss == 0: return 100.0
    return 100 - 100 / (1 + gain / loss)

def macd_hist(closes):
    if len(closes) < 26: return 0.0
    e12 = ema_calc(closes, 12); e26 = ema_calc(closes, 26); e9 = ema_calc(closes, 9)
    if None in (e12, e26, e9): return 0.0
    return float((e12 - e26) - e9)

# ── ML Models ───────────────────────────────────────────────────────────────
def load_v17():
    global _mv17
    if _mv17 is None and os.path.exists(MODEL_PATH_V17):
        with open(MODEL_PATH_V17, "rb") as f: _mv17 = pickle.load(f)
    return _mv17

def load_scaler_v17():
    global _sv17
    if _sv17 is None and os.path.exists(SCALER_PATH_V17):
        with open(SCALER_PATH_V17, "rb") as f: _sv17 = pickle.load(f)
    return _sv17

def load_v35():
    global _mv35
    if _mv35 is None and os.path.exists(MODEL_PATH_V35):
        with open(MODEL_PATH_V35, "rb") as f: _mv35 = pickle.load(f)
    return _mv35

def load_scaler_v35():
    global _sv35
    if _sv35 is None and os.path.exists(SCALER_PATH_V35):
        with open(SCALER_PATH_V35, "rb") as f: _sv35 = pickle.load(f)
    return _sv35

def predict_v17(rsi, macd, ema_cross, vol_delta, mom_5m, mom_1m, stoch, bb, atr, willr, rsi_f, taker, taker_avg):
    mv17 = load_v17(); sc17 = load_scaler_v17()
    if mv17 is None or sc17 is None: return "N/A", 0.0
    X = np.array([[rsi, macd, ema_cross, vol_delta, mom_5m, mom_1m, stoch, bb, atr, willr, rsi_f, taker, taker_avg, 0.0, 0.0]], dtype=np.float32)
    X_sc = sc17.transform(X)
    p = mv17.predict_proba(X_sc)[0]
    return ("UP" if p[1] >= p[0] else "DOWN", max(float(p[1]), float(p[0])))

def predict_v35(rsi, macd, ema_cross, vol_delta, stoch, bb_pos, atr, willr, mom_1m, mom_15m, taker, hour):
    mv35 = load_v35(); sc35 = load_scaler_v35()
    if mv35 is None or sc35 is None: return "N/A", 0.0
    X = np.array([[rsi, macd, ema_cross, vol_delta, stoch, bb_pos, atr, willr, mom_1m, mom_15m, taker, 0.0, hour]], dtype=np.float32)
    X_sc = sc35.transform(X)
    p = mv35.predict_proba(X_sc)[0]
    return ("UP" if p[1] >= p[0] else "DOWN", max(float(p[1]), float(p[0])))

def v4_signal_score(closes_1m, trend2h):
    if len(closes_1m) < 30: return 0.0, 50.0, 0.0, 0.0, 0.0, 0.0
    rsi_val = rsi_calc(closes_1m, 14)
    macd_val = macd_hist(closes_1m)
    ema9 = ema_calc(closes_1m, 9); ema21 = ema_calc(closes_1m, 21)
    mom5 = (closes_1m[-1] - closes_1m[-6]) / closes_1m[-6] * 100 if len(closes_1m) >= 6 else 0
    score = 0.0
    if rsi_val < 30: score += 0.25
    elif rsi_val > 70: score -= 0.25
    elif rsi_val < 40: score += 0.12
    elif rsi_val > 60: score -= 0.12
    score += 0.20 if macd_val > 0 else -0.20
    score += 0.20 if ema9 > ema21 else -0.20
    if mom5 > 0.1: score += 0.15
    elif mom5 < -0.1: score -= 0.15
    if trend2h > 0.60: score += 0.20
    elif trend2h < 0.40: score -= 0.20
    return score, rsi_val, macd_val, ema9, ema21, mom5

# ── Market Data Fetchers ───────────────────────────────────────────────────
def fetch_binance_klines_1m(limit=100):
    try:
        r = requests.get(f"{BINANCE_HOST}/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit}, timeout=8)
        data = r.json()
        return [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]
    except: return []

def fetch_binance_klines_5m(limit=150):
    try:
        r = requests.get(f"{BINANCE_HOST}/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit}, timeout=8)
        data = r.json()
        return [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]
    except: return []

def fetch_live_ticker_price():
    try:
        r = requests.get(f"{BINANCE_HOST}/api/v3/ticker/price",
                        params={"symbol": "BTCUSDT"}, timeout=8)
        return float(r.json()["price"])
    except: return None

def fetch_live_bid_ask():
    """Fetch best bid/ask for BTCUSDT from Binance depth."""
    try:
        r = requests.get(f"{BINANCE_HOST}/api/v3/ticker/bookTicker",
                         params={"symbol": "BTCUSDT"}, timeout=8)
        d = r.json()
        return float(d["bidPrice"]), float(d["askPrice"])
    except: return None, None

def fetch_market_trend():
    now = int(time.time()); up_w, dn_w = 0, 0
    for offset in range(1, 25):
        slot = (now // 300) * 300 - offset * 300
        try:
            r = requests.get(f"{MEXC_HOST}/api/v3/markets",
                            params={"slug": f"btc-updown-5m-{slot}"}, timeout=8)
            if r.status_code == 200 and r.json():
                m = r.json()[0]
                if m.get("closed"):
                    p = json.loads(m["outcomePrices"])
                    up_w += 1 if float(p[0]) >= 0.5 else 0
                    dn_w += 1 if float(p[0]) < 0.5 else 0
        except: pass
    total = up_w + dn_w
    return up_w / total if total > 0 else 0.5

# ── Held Position State ────────────────────────────────────────────────────
def load_held():
    if os.path.exists(HELD_STATE):
        with open(HELD_STATE) as f:
            return json.load(f)
    return None

def save_held(held):
    with open(HELD_STATE, "w") as f:
        json.dump(held, f, indent=2, default=str)

def pnl_realistic(entry_price, exit_price, direction, position_size):
    """
    Realistic PnL calculation:
    - position_size = $ (dollar amount we're putting up as margin)
    - leverage multiplies the return
    - CONTRACT_MULTIPLIER = 1 contract per $1 of position_size
    """
    if direction == "LONG":
        ret_pct = (exit_price - entry_price) / entry_price
    else:  # SHORT
        ret_pct = (entry_price - exit_price) / entry_price
    # PnL in USD: position_size * leverage * return_pct
    pnl = position_size * LEVERAGE * ret_pct
    return pnl, ret_pct

# ── Main Run ───────────────────────────────────────────────────────────────
def run():
    now = int(time.time())
    next_slot = (now // SLOT_DURATION) * SLOT_DURATION + SLOT_DURATION
    current_slot = (now // SLOT_DURATION) * SLOT_DURATION
    secs_remaining = next_slot - now

    if secs_remaining < MIN_ENTRY_SECS:
        return

    # ── Fetch data ──────────────────────────────────────────────────────
    klines_1m = fetch_binance_klines_1m(100)
    klines_5m = fetch_binance_klines_5m(150)
    if not klines_1m or not klines_5m:
        return

    live_price = fetch_live_ticker_price()
    bid_price, ask_price = fetch_live_bid_ask()
    price_1m = live_price if live_price else klines_1m[-1][4]
    current_price = bid_price if bid_price else price_1m

    closes_1m = [float(x[4]) for x in klines_1m]
    rsi = rsi_calc(closes_1m, 14)
    macd = macd_hist(closes_1m)
    ema_cross = closes_1m[-1] - closes_1m[-2]
    vol_delta = (closes_1m[-1] - closes_1m[0]) / closes_1m[0] * 100
    mom_5m = (closes_1m[-1] - closes_1m[-6]) / closes_1m[-6] * 100 if len(closes_1m) >= 6 else 0
    mom_1m = (closes_1m[-1] - closes_1m[-2]) / closes_1m[-2] * 100
    stoch = 50.0; bb_pos = 0.5; atr = 0.1; willr = -50.0
    taker_ratio = 1.0

    trend2h = fetch_market_trend()
    v4_score, _, _, _, _, _ = v4_signal_score(closes_1m, trend2h)
    v35_dir, v35_conf = predict_v35(rsi, macd, ema_cross, vol_delta, stoch, bb_pos, atr, willr, mom_1m, mom_1m, taker_ratio, now // 3600)
    v17_dir, v17_conf = predict_v17(rsi, macd, ema_cross, vol_delta, mom_5m, mom_1m, stoch, bb_pos, atr, willr, rsi, taker_ratio, taker_ratio)

    signal_dir = v17_dir  # Primary signal from v17
    signal_conf = v17_conf

    # ── Slot deduplication ──────────────────────────────────────────────
    traded = set()
    if os.path.exists(TRACKER):
        with open(TRACKER) as f:
            traded = set(json.load(f))

    # ── Load held position from previous slot ────────────────────────────
    held = load_held()
    position_size = BALANCE * POSITION_PCT  # $5 per position

    entry_price = 0.0; exit_price = 0.0; pnl = 0.0; closed_5m = False
    position = "WAIT"; new_hold = None; note = ""

    if held:
        # ── We have a held position from previous slot ──────────────────
        held_dir = held["direction"]
        held_entry = held["entry_price"]
        held_slot = held["slot"]
        held_size = held.get("position_size", position_size)
        held_entry_price = held_entry

        # Current unrealized PnL
        cur_pnl, cur_ret = pnl_realistic(held_entry, current_price, held_dir, held_size)
        ret_pct = abs(cur_ret) * 100

        print(f"  📌 Held position: {held_dir} @ ${held_entry:.2f} → current ${current_price:.2f}")
        print(f"     Unrealized: ${cur_pnl:+.4f} ({held_dir} {ret_pct:.2f}%)")

        # Determine exit condition
        if cur_pnl > 0:
            # Profitable → EXIT
            exit_price = current_price
            pnl, _ = pnl_realistic(held_entry, exit_price, held_dir, held_size)
            position = f"HOLD_EXIT_{held_dir}"
            note = f"Held→exit profitable ${pnl:+.4f}"
            print(f"  ✅ HOLD策略觸發：有利潤，EXIT {held_dir} @ ${exit_price:.2f} PnL=${pnl:+.4f}")
            new_hold = None  # Clear held position
        elif cur_ret <= -HOLD_LOSS_THRESHOLD:
            # Loss exceeds threshold → CLOSE at loss
            exit_price = current_price
            pnl, _ = pnl_realistic(held_entry, exit_price, held_dir, held_size)
            position = f"HOLD_LOSS_{held_dir}"
            note = f"Held→forced close loss=${pnl:.4f} ({abs(cur_ret)*100:.1f}%)"
            print(f"  ❌ HOLD策略：虧損超10%({abs(cur_ret)*100:.1f}%)，FORCE CLOSE PnL=${pnl:.4f}")
            new_hold = None  # Clear held position
        else:
            # Within hold zone (< 10% loss, no profit) → keep holding
            position = f"HOLDING_{held_dir}"
            entry_price = held_entry
            new_hold = {
                "direction": held_dir,
                "entry_price": held_entry,
                "slot": held_slot,
                "position_size": held_size,
                "opened_at": held.get("opened_at"),
                "hold_count": held.get("hold_count", 0) + 1,
                "worst_pnl": min(held.get("worst_pnl", 0), cur_pnl),
            }
            note = f"Holding (loss={cur_pnl:+.4f}, {abs(cur_ret)*100:.1f}% < 10%)"
            print(f"  ⏸️ HOLD策略：保持在野，loss={cur_pnl:+.4f}，等下一slot")
    else:
        # ── No held position — decide new entry ───────────────────────────
        if signal_dir == "WAIT" or current_slot in traded:
            pass  # No signal or already traded this slot
        else:
            entry_price = klines_1m[-1][4]  # Entry at last closed 1m candle close
            position = "LONG" if signal_dir == "UP" else "SHORT" if signal_dir == "DOWN" else "WAIT"

            if position != "WAIT":
                traded.add(current_slot)
                with open(TRACKER, "w") as f:
                    json.dump(list(traded), f)

                # ── Evaluate 5m candle for this slot ──────────────────────
                slot_close_ms = (current_slot + 300) * 1000
                try:
                    r = requests.get(f"{BINANCE_HOST}/api/v3/klines",
                                   params={"symbol": "BTCUSDT", "interval": "5m",
                                           "startTime": current_slot * 1000,
                                           "endTime": slot_close_ms, "limit": 1}, timeout=8)
                    candle = r.json()
                    if candle and len(candle) > 0:
                        ct = int(candle[0][6])
                        if ct < now * 1000:  # Candle closed
                            exit_price = float(candle[0][4])
                            closed_5m = True
                        else:
                            exit_price = current_price
                except:
                    exit_price = current_price

                cur_pnl, cur_ret = pnl_realistic(entry_price, exit_price, position, position_size)
                ret_pct = abs(cur_ret) * 100

                # ── Decision: exit, hold, or keep? ───────────────────────
                if cur_pnl > 0:
                    # Profitable in same slot → close immediately
                    pnl = cur_pnl
                    note = f"Same-slot exit (profit)"
                    new_hold = None
                    print(f"  ✅ 同slot獲利：{position} @ ${entry_price:.2f}→${exit_price:.2f} PnL=${pnl:+.4f}")
                elif cur_ret <= -HOLD_LOSS_THRESHOLD:
                    # Loss exceeds threshold → HOLD
                    pnl = cur_pnl
                    new_hold = {
                        "direction": position,
                        "entry_price": entry_price,
                        "slot": current_slot,
                        "position_size": position_size,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "hold_count": 0,
                        "worst_pnl": cur_pnl,
                    }
                    note = f"Hold: loss={pnl:.4f} ({ret_pct:.1f}%), will exit when profitable"
                    print(f"  ⏸️ HOLD策略：同slot虧損{ret_pct:.1f}%(>10%)，HOLD等下一slot")
                else:
                    # Within hold zone → close normally
                    pnl = cur_pnl
                    note = f"Same-slot close (within hold zone)"
                    new_hold = None
                    print(f"  ⚖️ 同slot within hold zone: {position} PnL=${pnl:+.4f}")

    # ── Save held position state ─────────────────────────────────────────
    if new_hold:
        save_held(new_hold)
    else:
        # Clear held state
        if os.path.exists(HELD_STATE):
            os.remove(HELD_STATE)

    # ── Log record ───────────────────────────────────────────────────────
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slot": current_slot,
        "price": current_price,
        "bid": bid_price,
        "ask": ask_price,
        "direction": position,
        "signal_dir": signal_dir,
        "v4_score": v4_score,
        "v17_dir": v17_dir, "v17_conf": round(v17_conf, 4),
        "v35_dir": v35_dir, "v35_conf": round(v35_conf, 4),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "closed_5m": closed_5m,
        "pnl": round(pnl, 6) if pnl else 0.0,
        "note": note,
        "held": new_hold,
    }

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(log_record) + "\n")

    direction_emoji = {"LONG": "🟢", "SHORT": "🔴", "WAIT": "⚪",
                       "HOLD_EXIT_LONG": "✅🟢", "HOLD_EXIT_SHORT": "✅🔴",
                       "HOLD_LOSS_LONG": "❌🟢", "HOLD_LOSS_SHORT": "❌🔴",
                       "HOLDING_LONG": "⏸️🟢", "HOLDING_SHORT": "⏸️🔴"}
    emoji = direction_emoji.get(position, "⚪")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {emoji} {position} @ ${current_price:.2f} | PnL: {pnl:+.4f} | {note}")

if __name__ == "__main__":
    run()
