#!/usr/bin/env python3
"""
MEXC P1 15m Paper Trader — Beta V1 Primary
==========================================
Beta V1: RandomForest_d3, 35 features, HO AUC=0.6326
- No hour bias | UP=51.4%, DOWN=67.6% correct
- v42 is fallback
"""
import os, sys, json, time, math, datetime as dt
import requests, numpy as np, pickle

SKILL    = os.path.dirname(os.path.abspath(__file__))
STATE_F  = os.path.join(SKILL, "data", "mexc_p1_beta_v1_state.json")
LOG_F    = os.path.join(SKILL, "logs",   "mexc_p1_beta_v1_trades.jsonl")
MODEL_B  = os.path.join(SKILL, "data", "ml_model_beta_v1.pkl")
MODEL_V42 = os.path.join(SKILL, "data", "ml_model_v42_flaml.pkl")
SCAL_V42 = os.path.join(SKILL, "data", "scaler_v42_flaml.pkl")
MEXC     = "https://fapi.binance.com"
PAIR     = "BTCUSDT"
SLOT_MIN = 15
HOLD_MIN = 15
POS      = 0.95
LEV      = 3

_beta_m  = None
_v42_m   = None
_v42_sc  = None

# ─── Models ─────────────────────────────────────────────────────────────────
def load_beta():
    global _beta_m
    if _beta_m is not None: return _beta_m
    try:
        with open(MODEL_B, "rb") as f:
            _beta_m = pickle.load(f)
        print("   Beta V1 loaded")
    except Exception as e:
        print(f"   Beta V1 load error: {e}")
    return _beta_m

def load_v42():
    global _v42_m, _v42_sc
    if _v42_m is not None: return _v42_m, _v42_sc
    try:
        with open(MODEL_V42, "rb") as f:
            _v42_m = pickle.load(f)
        with open(SCAL_V42, "rb") as f:
            _v42_sc = pickle.load(f)
        print("   v42 fallback loaded")
    except Exception as e:
        print(f"   v42 load error: {e}")
    return _v42_m, _v42_sc

# ─── Fetchers ────────────────────────────────────────────────────────────
def fetch_klines(limit=120):
    url = f"{MEXC}/fapi/v1/klines"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"symbol": PAIR, "interval": "1m", "limit": limit}, timeout=8)
            data = r.json()
            closes  = [float(k[4]) for k in data]
            highs   = [float(k[2]) for k in data]
            lows    = [float(k[3]) for k in data]
            volumes = [float(k[5]) for k in data]
            times   = [int(k[0])//1000 for k in data]
            return times, closes, highs, lows, volumes
        except Exception as e:
            print(f"   Kline error: {e}")
            time.sleep(1)
    return None, [], [], [], []

def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng", timeout=5)
        return float(r.json()["data"][0]["value"])
    except:
        return 50

def fetch_ls():
    try:
        r = requests.get(f"{MEXC}/fapi/v1/globalLongShortAccountRatio",
                        params={"symbol": PAIR, "periodType": "1h", "limit": 1}, timeout=5)
        return float(r.json()[0]["longShortRatio"])
    except:
        return 1.0

# ─── Indicators ─────────────────────────────────────────────────────────
def ema_arr(arr, n):
    k = 2/(n+1)
    out = float(arr[0])
    for v in arr[1:]:
        out = float(v)*k + out*(1-k)
    return out

def calc_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = float(np.mean(gains[-period:]))
    avg_l = max(float(np.mean(losses[-period:])), 1e-8)
    return 100 - (100 / (1 + avg_g/avg_l))

def calc_stoch(highs, lows, closes, k_p=14):
    k_vals = []
    for i in range(len(closes)-k_p, len(closes)):
        wl = float(np.min(lows[i:i+k_p]))
        wh = float(np.max(highs[i:i+k_p]))
        k_vals.append(100*(closes[i]-wl)/(wh-wl+1e-8) if wh != wl else 50)
    k = float(np.mean(k_vals[-k_p:]))
    d = ema_arr(np.array(k_vals[-k_p:]), 3)
    return k, d

def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return float(np.mean(trs[-period:]))

# ─── Feature Builder (35 Beta V1 features) ───────────────────────────────
def build_beta_features(closes, highs, lows, volumes, fng_val, ls_ratio):
    c  = float(closes[-1])
    n  = len(closes)

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg14_g = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
    avg14_l = max(float(np.mean(losses[-14:])) if len(gains) >= 14 else float(np.mean(losses)), 1e-8)
    rsi14 = 100 - (100 / (1 + avg14_g/avg14_l))
    avg5_g = float(np.mean(gains[-5:])) if len(gains) >= 5 else avg14_g
    avg5_l = max(float(np.mean(losses[-5:])) if len(gains) >= 5 else avg14_l, 1e-8)
    rsi5  = 100 - (100 / (1 + avg5_g/avg5_l))

    low14  = float(np.min(lows[-14:]))
    high14 = float(np.max(highs[-14:]))
    sk_raw = 100*(c-low14)/(high14-low14+1e-8)
    willr  = -100*(high14-c)/(high14-low14+1e-8)

    sma20 = float(np.mean(closes[-20:]))
    std20 = float(np.std(closes[-20:]))
    bb    = (c-(sma20-2*std20))/(4*std20+1e-8)

    atr   = calc_atr(highs, lows, closes) / c * 100
    mom5  = (c-closes[-6])/closes[-6]*100 if n >= 6 else 0
    mom1  = (c-closes[-2])/closes[-2]*100 if n >= 2 else 0

    taker_n   = float(volumes[-1]) if len(volumes) > 0 else 0.5
    taker_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else taker_n
    obi      = (taker_n/taker_avg - 1) if taker_avg > 0 else 0

    ls = float(ls_ratio)

    def nf(v): return float(v)
    def clip(v, lo, hi): return max(lo, min(hi, v))

    return [
        # Continuous (normalized)
        clip(rsi14/100, 0, 1),
        clip(ls/2.5, 0, 1),
        clip(sk_raw/100, 0, 1),
        clip((willr+100)/100, 0, 1),
        clip(bb, 0, 1),
        clip(atr/0.01, 0, 1),
        clip(mom5/0.5, -1, 1),
        clip(obi+0.5, 0, 1),
        # RSI categorical
        1.0 if rsi14 < 30 else 0.0,
        1.0 if rsi14 < 40 else 0.0,
        1.0 if 40 <= rsi14 <= 60 else 0.0,
        1.0 if 60 < rsi14 <= 70 else 0.0,
        1.0 if rsi14 > 70 else 0.0,
        1.0 if rsi14 < 20 else 0.0,
        # L/S categorical
        1.0 if ls < 0.5 else 0.0,
        1.0 if 0.5 <= ls < 0.9 else 0.0,
        1.0 if 0.9 <= ls <= 1.1 else 0.0,
        1.0 if 1.1 <= ls < 1.3 else 0.0,
        1.0 if ls >= 1.3 else 0.0,
        # Momentum categorical
        1.0 if mom5 < -0.2 else 0.0,
        1.0 if mom5 < 0 else 0.0,
        1.0 if mom5 > 0 else 0.0,
        # Stochastic categorical
        1.0 if sk_raw < 20 else 0.0,
        1.0 if sk_raw > 80 else 0.0,
        # WillR categorical
        1.0 if willr < -80 else 0.0,
        # BB categorical
        1.0 if bb < 0.2 else 0.0,
        1.0 if bb > 0.8 else 0.0,
        # Range (placeholder)
        1.0,
        1.0,
        # Composite signals
        (1.0 if rsi14 < 30 else 0.0) * (1.0 if 1.1 <= ls < 1.3 else 0.0),
        (1.0 if 40 <= rsi14 <= 60 else 0.0) * (1.0 if 0.5 <= ls < 0.9 else 0.0),
        (1.0 if rsi14 < 30 else 0.0) * (1.0 if sk_raw < 20 else 0.0),
        (1.0 if sk_raw < 20 else 0.0) * (1.0 if mom5 < 0 else 0.0),
        (1.0 if ls < 0.5 else 0.0) * (1.0 if rsi14 < 30 else 0.0),
        1.0,
    ]

# ─── Predict ───────────────────────────────────────────────────────────
def predict_beta(closes, highs, lows, volumes, fng_val, ls_ratio):
    model = load_beta()
    if model is None: return "N/A", 0.0
    feats = build_beta_features(closes, highs, lows, volumes, fng_val, ls_ratio)
    x = np.array([feats])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        prob = model.predict_proba(x)[0]
        di   = int(model.predict(x)[0])
        conf = float(max(prob))
        return "UP" if di == 1 else "DOWN", conf
    except Exception as e:
        print(f"   Beta predict error: {e}")
        return "N/A", 0.0

# ─── Rule Signal ───────────────────────────────────────────────────────
def rule_signal(rsi_val):
    if rsi_val < 30:   return "UP",   0.5
    elif rsi_val < 40:  return "UP",   0.2
    elif rsi_val > 70:  return "DOWN", 0.5
    elif rsi_val > 60:  return "DOWN", 0.2
    elif 40 <= rsi_val <= 60: return "DOWN", 0.4
    return None, 0.0

# ─── State ─────────────────────────────────────────────────────────────
def load_state():
    try:
        with open(STATE_F) as f:
            return json.load(f)
    except:
        return {"slot": 0, "entry_price": 0, "position_mode": None, "pnl": 0.0}

def save_state(s):
    with open(STATE_F, "w") as f:
        json.dump(s, f, indent=2)

def log(entry):
    entry["ts"] = dt.datetime.now().isoformat()
    with open(LOG_F, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ─── Main ──────────────────────────────────────────────────────────────
def main():
    now_ts  = int(time.time())
    slot_ts = (now_ts // (SLOT_MIN*60)) * (SLOT_MIN*60)
    state   = load_state()

    print(f"\n{'='*55}")
    print(f"  MEXC P1 Beta V1 [{dt.datetime.now().strftime('%H:%M')}]")
    print(f"{'='*55}")

    times, closes, highs, lows, volumes = fetch_klines(limit=120)
    if not closes:
        print("  No kline data")
        return

    fng = fetch_fng()
    ls  = fetch_ls()
    btc = float(closes[-1])
    rsi = calc_rsi(closes)
    sk, _ = calc_stoch(highs, lows, closes)
    atr = calc_atr(highs, lows, closes)

    print(f"  BTC=${btc:,.0f} | RSI={rsi:.0f} | Stoch={sk:.0f}")
    print(f"  F&G={fng:.0f} | LS={ls:.2f}")

    # Predict
    beta_dir, beta_conf = predict_beta(closes, highs, lows, volumes, fng, ls)
    print(f"  Beta V1: {beta_dir} {beta_conf:.1%}")

    # Rule
    rule_dir, rule_conf = rule_signal(rsi)
    if rule_dir:
        print(f"  RSI rule: {rule_dir}")

    # Cascade: Rule PRIMARY, Beta V1 CONFIRMS
    final_dir = None
    reason = ""

    if rule_dir is None:
        reason = "RSI neutral (40-70)"
        print(f"  WAIT: {reason}")

    elif rule_dir == beta_dir or beta_dir == "N/A":
        final_dir = rule_dir
        reason = f"Rule+Beta agree: {final_dir}"
        print(f"  TRADE: {reason}")

    elif beta_dir != "N/A":
        reason = f"Rule({rule_dir}) != Beta({beta_dir} {beta_conf:.1%})"
        print(f"  WAIT: {reason}")

    # Position management
    has_pos = state["position_mode"] is not None

    if has_pos:
        entry  = state["entry_price"]
        mode   = state["position_mode"]
        pnl    = state["pnl"]

        if slot_ts > state["slot"]:
            close_px = btc
            if mode == "LONG":
                pnl = (close_px - entry) / entry * POS * LEV
            else:
                pnl = (entry - close_px) / entry * POS * LEV
            state["pnl"] = pnl
            state["position_mode"] = None
            save_state(state)
            log({"slot": state["slot"], "direction": "CLOSE",
                 "mode": mode, "entry_price": entry,
                 "close_price": close_px, "pnl": round(pnl, 6),
                 "beta_dir": beta_dir, "beta_conf": beta_conf,
                 "rule_dir": rule_dir})
            print(f"  CLOSE {mode} @ ${close_px:,.1f} | PnL: {pnl:+.4f}")
            has_pos = False
        else:
            print(f"  HOLD {mode} @ ${entry:,.1f} | PnL: {pnl:+.4f}")
            log({"slot": slot_ts, "direction": "HOLD", "mode": mode,
                 "entry_price": entry, "pnl": pnl,
                 "beta_dir": beta_dir, "beta_conf": beta_conf,
                 "rule_dir": rule_dir})

    # Open new
    if not has_pos and final_dir is not None:
        mode = "SHORT" if final_dir == "DOWN" else "LONG"
        state["slot"] = slot_ts
        state["entry_price"] = btc
        state["position_mode"] = mode
        state["pnl"] = 0.0
        save_state(state)
        log({"slot": slot_ts, "direction": final_dir, "mode": mode,
             "entry_price": btc, "pnl": 0,
             "beta_dir": beta_dir, "beta_conf": beta_conf,
             "rule_dir": rule_dir})
        print(f"  OPEN {mode} @ ${btc:,.1f}")

    if not has_pos and final_dir is None:
        log({"slot": slot_ts, "direction": "WAIT",
             "beta_dir": beta_dir, "beta_conf": beta_conf,
             "rule_dir": rule_dir})

if __name__ == "__main__":
    main()
