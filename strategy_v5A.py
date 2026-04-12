#!/usr/bin/env python3
"""
BTC-5m Option A Strategy — v5.1 Rules + ML Confirmation Hybrid
================================================================
Based on v5.1 (2026-03-20) + deep data analysis (2026-04-11)

PRINCIPLE: Rules come FIRST. ML CONFIRMS.
  - RSI-based rules are statistically significant (Z>1.96)
  - ML only used to confirm, not to override rules
  - Market Neutral Filter prevents overtrading sideways markets

Key differences from conservative:
  1. Rules are PRIMARY signal (RSI < 30 → UP, RSI 40-60 → DOWN)
  2. ML (v46/v42) must confirm rules OR wait
  3. Market Neutral Filter restored (EMA 9 vs 21 diff < 0.3% → wait)
  4. ATR dynamic confidence (0.51-0.80)
  5. Hard stop: 10% daily loss (v5.1 style)
  6. Proper logging of v46_dir/v46_conf

Statistically validated rules:
  ✅ RSI < 30 → market UP 57% (Z=2.66)
  ✅ RSI 40-60 → market DOWN 76% (Z=2.17)
  ✅ LS 1.1-1.3 → market UP 51% (Z=3.03)
  ✅ Range < 20 → market DOWN 84%
  ❌ F&G, Stoch, WillR, EMA, MACD → random (ignored)

2026-04-11
"""
import os, sys, json, time, requests, random, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pickle

# ─── PATHS ──────────────────────────────────────────────────────────────────
SKILL   = Path(__file__).parent
DATA    = SKILL / "data"
BASE    = SKILL
DB_PATH = DATA / "btc5m_research.db"

# ─── CHAINLINK ORACLE ───────────────────────────────────────────────────────
# BTC/USD Feed on Ethereum
# Proxy: 0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88C
# Function: latestAnswer() = 0x50d25bcd | decimals=8
# Polymarket settles based on Chainlink BTC/USD (confirmed)
# Binance is our signal source — CL deviation = settlement risk
#
# Dual Protection:
#   1. Deviation > 1.0%  → BLOCK all trades (market dislocation)
#   2. CL LAG DETECTED   → CL catching up = big momentum coming
#      - Binance trending + CL not following → CL LAG ALERT
#      - Enhanced confidence in Binance momentum direction

_cl_cache  = {"price": None, "ts": 0}
_cl_history = []   # [(timestamp, cl_price, binance_price), ...]

def get_chainlink_price():
    global _cl_cache
    now = time.time()
    if _cl_cache["price"] is not None and (now - _cl_cache["ts"]) < 60:
        return _cl_cache["price"]
    try:
        payload = {"jsonrpc": "2.0", "method": "eth_call",
                   "params": [{"to": "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88C",
                               "data": "0x50d25bcd"}, "latest"], "id": 1}
        r = requests.post("https://1rpc.io/eth", json=payload, timeout=8)
        d = r.json()
        res = d.get("result", "")
        if res and res != "0x":
            price = int(res[-16:], 16) / 1e8
            _cl_cache = {"price": price, "ts": now}
            return price
    except Exception:
        pass
    return _cl_cache["price"]

def chainlink_oracle_check(btc_price, binance_momentum_pct=None):
    """Enhanced oracle check with CL lag detection.
    
    Returns (alert_level, deviation_pct, cl_lag_signal, message)
    alert_level: 0=normal, 1=high_deviation, 2=extreme (BLOCK)
    cl_lag_signal: None, 'UP_MOMENTUM', 'DOWN_MOMENTUM'
      → CL is lagging behind Binance momentum; catch-up move incoming
    """
    global _cl_history
    cl_price = get_chainlink_price()
    if cl_price is None:
        return 0, 0.0, None, "chainlink_unavailable"

    now = time.time()
    # Record history (keep last 6 readings = ~30min)
    _cl_history.append((now, cl_price, btc_price))
    if len(_cl_history) > 6:
        _cl_history = _cl_history[-6:]

    deviation = abs(cl_price - btc_price) / btc_price * 100

    # Alert level
    if deviation > 1.0:
        alert = 2
        return alert, deviation, None, f"🔴 CL/BINANCE DIFF={deviation:+.2f}% — BLOCK"
    elif deviation > 0.5:
        alert = 1
        msg = f"🟡 CL/BINANCE DIFF={deviation:+.2f}% ⚠️"
    else:
        alert = 0
        msg = f"🟢 CL={cl_price:,.0f} BIN={btc_price:,.0f}"

    # CL Lag Detection
    # Pattern: Binance trending in one direction, CL barely moving
    # If Binance moved >0.3% in last 3 readings but CL moved <30% of that → CL LAG
    cl_lag_signal = None
    if len(_cl_history) >= 3:
        bx_move = btc_price - _cl_history[-3][2]   # Binance change in window
        cl_move = cl_price - _cl_history[-3][1]   # CL change in window
        bx_pct  = abs(bx_move) / _cl_history[-3][2] * 100
        cl_pct  = abs(cl_move) / (_cl_history[-3][1] or 1) * 100
        if bx_pct > 0.3 and cl_pct < bx_pct * 0.3:
            # CL lagging behind Binance momentum
            if bx_move > 0:
                cl_lag_signal = "UP_MOMENTUM"
                msg += " | 🚀 CL LAG: UP catch-up incoming"
            else:
                cl_lag_signal = "DOWN_MOMENTUM"
                msg += " | 🔻 CL LAG: DOWN catch-up incoming"

    return alert, deviation, cl_lag_signal, msg


# ─── ML MODEL LOADING ────────────────────────────────────────────────────────
_mv42 = _sv42 = _feat42 = None
_mv46 = _sv46 = _feat46 = None
_mbeta = _sbeta = _fbeta = None

def load_v42():
    global _mv42, _sv42, _feat42
    if _mv42 is not None: return _mv42, _sv42, _feat42
    try:
        with open(DATA / "ml_model_v42_flaml.pkl", "rb") as f:
            _mv42 = pickle.load(f)
        with open(DATA / "scaler_v42_flaml.pkl", "rb") as f:
            _sv42 = pickle.load(f)
        with open(DATA / "ml_features_v42.json") as f:
            _feat42 = json.load(f)
        print(f"   ⭐ v42 loaded: {len(_feat42)} features, {type(_mv42).__name__}")
    except Exception as e:
        print(f"   ⚠️ v42 load failed: {e}")
    return _mv42, _sv42, _feat42

def load_v46():
    global _mv46, _sv46, _feat46
    if _mv46 is not None: return _mv46, _sv46, _feat46
    try:
        with open(DATA / "ml_model_v46.pkl", "rb") as f:
            _mv46 = pickle.load(f)
        _sv46 = _mv46.named_steps["standardscaler"]
        with open(DATA / "ml_features_v46.json") as f:
            raw = json.load(f)
            _feat46 = raw["features"] if isinstance(raw, dict) else raw
        print(f"   🔶 v46 loaded: {len(_feat46)} features, {type(_mv46).__name__}")
    except Exception as e:
        print(f"   ⚠️ v46 load failed: {e}")
    return _mv46, _sv46, _feat46

def load_beta_v1():
    """Load Beta V1 (RF_d3, unbiased, no hour bias). HO AUC=0.6326."""
    global _mbeta, _sbeta, _fbeta
    if _mbeta is not None: return _mbeta, _sbeta, _fbeta
    try:
        import sklearn.base
        with open(DATA / "ml_model_beta_v1.pkl", "rb") as f:
            _mbeta = pickle.load(f)
        with open(DATA / "ml_features_beta_v1.json") as f:
            _fbeta = json.load(f)["feature_names"]
        if hasattr(_mbeta, 'named_steps'):
            _sbeta = _mbeta.named_steps.get('standardscaler')
        print(f"   🟢 Beta V1 loaded: {len(_fbeta)} features")
    except Exception as e:
        print(f"   ⚠️ Beta V1 load failed: {e}")
    return _mbeta, _sbeta, _fbeta


# ─── FETCHERS ────────────────────────────────────────────────────────────────
def fetch_btc_klines_5m(limit=200):
    """Fetch 5m Binance klines. Returns (klines_dict, times_list)."""
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit}, timeout=10)
        data = r.json()
        klines = {}
        for k in data:
            ts = int(k[0]) // 1000
            klines[ts] = {
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
                "quote_vol": float(k[7]) if len(k) > 7 else 0,
                "taker_buy_vol": float(k[8]) if len(k) > 8 else 0,
            }
        times = sorted(klines.keys())
        return klines, times
    except:
        return {}, []

def fetch_btc_klines_1m(limit=60):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit}, timeout=10)
        return [float(k[4]) for k in r.json()]
    except:
        return []

def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng", timeout=5)
        return float(r.json().get("data", [{}])[0].get("value", 50))
    except:
        return 50

def fetch_ls_ratio():
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 3}, timeout=5
        )
        return float(r.json()[0].get("longShortRatio", 1.0))
    except:
        return 1.0

def find_polymarket_slot(now_ts):
    """Find the active Polymarket slot for this 5m window."""
    for slot in [(now_ts // 300) * 300, (now_ts // 300) * 300 + 300]:
        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": f"btc-updown-5m-{slot}"}, timeout=10
            )
            if r.status_code == 200 and r.json():
                m = r.json()[0]
                if not m.get("acceptingOrders"):
                    continue
                ids = json.loads(m.get("clobTokenIds", "[]"))
                if len(ids) >= 2:
                    prices = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
                    return {
                        "slot": slot,
                        "up_token": ids[0], "down_token": ids[1],
                        "up_price": float(prices[0]),
                        "question": m.get("question", ""),
                        "min_size": float(m.get("orderMinSize", 5)),
                        "tick_size": str(m.get("orderPriceMinTickSize", "0.01")),
                    }
        except:
            pass
    return None

# ─── TECHNICAL INDICATORS ──────────────────────────────────────────────────
def ema_arr(arr, n):
    k = 2 / (n + 1)
    out = float(arr[0])
    for v in arr[1:]:
        out = float(v) * k + out * (1 - k)
    return out

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = float(np.mean(gains[-period:]))
    avg_l = max(float(np.mean(losses[-period:])), 1e-8)
    return 100 - (100 / (1 + avg_g / avg_l))

def calc_stoch(highs, lows, closes, k_p=14):
    if len(closes) < k_p + 1:
        return 50.0, 50.0
    k_vals = []
    for i in range(len(closes) - k_p, len(closes)):
        win_low = float(np.min(lows[i - k_p:i + 1]))
        win_high = float(np.max(highs[i - k_p:i + 1]))
        if win_high > win_low:
            k_vals.append(100 * (closes[i] - win_low) / (win_high - win_low))
        else:
            k_vals.append(50)
    k = float(np.mean(k_vals[-k_p:]))
    d = ema_arr(np.array(k_vals[-k_p:]), 3)
    return k, d

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 100.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    return float(np.mean(trs[-period:]))

# ─── RULE-BASED SIGNAL (v5.1 style, statistically validated) ──────────────
def rule_signal(rsi_val, ls_ratio, ema_diff_pct, hour_utc, slot_range):
    """
    Returns (direction, confidence, reason).
    Direction is PRIMARY from rules. ML only confirms.
    
    Rules based on statistical analysis (Z > 1.96):
      ✅ RSI < 30 → UP (market UP 57%, Z=2.66)
      ✅ RSI 40-60 → DOWN (market DOWN 76%, Z=2.17)
      ✅ LS 1.1-1.3 → UP edge (market UP 51%)
      ✅ Range < 20 → DOWN (market DOWN 84%)
      
    Market Neutral Filter:
      EMA 9 vs 21 diff < 0.3% → sideways → WAIT
    """
    reasons = []
    score = 0.0

    # ── Market Neutral Filter ─────────────────────────────────────────────
    if abs(ema_diff_pct) < 0.3:
        return "WAIT", 0.0, "Market Neutral (EMA diff < 0.3%)"

    # ── Primary Rules ──────────────────────────────────────────────────────
    # RSI-based (statistically validated)
    if rsi_val < 30:
        score += 0.5
        reasons.append(f"RSI={rsi_val:.0f}<30 (oversold)")
    elif 30 <= rsi_val < 40:
        score += 0.2
        reasons.append(f"RSI={rsi_val:.0f} 30-40 (weak oversold)")
    elif 40 <= rsi_val <= 60:
        score -= 0.4
        reasons.append(f"RSI={rsi_val:.0f} 40-60 (neutral→DOWN)")
    elif 60 < rsi_val <= 70:
        score -= 0.2
        reasons.append(f"RSI={rsi_val:.0f} 60-70 (weak overbought)")
    elif rsi_val > 70:
        score -= 0.4
        reasons.append(f"RSI={rsi_val:.0f}>70 (overbought)")

    # L/S Ratio (from analysis: LS 1.1-1.3 → UP edge)
    if ls_ratio >= 1.1 and ls_ratio <= 1.3:
        score += 0.15
        reasons.append(f"LS={ls_ratio:.2f} 1.1-1.3 (bullish crowd)")
    elif ls_ratio > 1.3:
        score += 0.1
        reasons.append(f"LS={ls_ratio:.2f}>1.3 (crowded long)")
    elif ls_ratio < 0.5:
        score -= 0.15
        reasons.append(f"LS={ls_ratio:.2f}<0.5 (crowded short)")

    # Slot range (tight range → strong DOWN)
    if slot_range is not None and slot_range < 20:
        score -= 0.3
        reasons.append(f"Range=${slot_range:.0f}<20 (tight→DOWN)")

    # Hour filter (UTC 06-09, 17 → low quality)
    if hour_utc in (6, 7, 8, 9, 17):
        score *= 0.5
        reasons.append(f"UTC{hour_utc} (low alpha hours)")

    if abs(score) < 0.1:
        return "WAIT", abs(score), " | ".join(reasons) if reasons else "Score too low"

    direction = "UP" if score > 0 else "DOWN"
    confidence = min(abs(score), 1.0)

    return direction, confidence, " | ".join(reasons)

# ─── ML PREDICTION ──────────────────────────────────────────────────────────
def build_v46_features(klines_dict, times, slot_ts, fng, ls_ratio):
    """Build 67 features for v46 from klines + external signals."""
    closes = np.array([klines_dict[t]["close"] for t in times])
    highs  = np.array([klines_dict[t]["high"]  for t in times])
    lows   = np.array([klines_dict[t]["low"]   for t in times])
    vols   = np.array([klines_dict[t]["volume"] for t in times])
    taker_arr = np.array([klines_dict[t].get("taker_buy_vol", 0) for t in times])

    c = float(closes[-1])
    n = len(closes)
    ts = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    hour_utc = float(ts.hour)

    def ema(a, per):
        return ema_arr(a, per)

    f = {}

    # EMA
    ema5  = ema(closes, 5)
    ema9  = ema(closes, 9)
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema26 = ema(closes, 26)
    ema12 = ema(closes, 12)

    # ATR
    trs = np.maximum(highs[1:] - lows[1:],
                     np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    atr_n = float(np.mean(trs[-14:])) / c * 100 if len(trs) >= 14 else 0.001

    # BB
    sma20 = float(np.mean(closes[-20:]))
    std20 = float(np.std(closes[-20:]))
    bb_pos_n = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    # Consec
    consec_up = consec_down = 0
    for x in reversed(closes[:-1]):
        if x < closes[-1]: consec_up += 1
        else: break
    for x in reversed(closes[:-1]):
        if x > closes[-1]: consec_down += 1
        else: break

    # RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg14_g = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
    avg14_l = max(float(np.mean(losses[-14:])) if len(gains) >= 14 else float(np.mean(losses)), 1e-8)
    rsi14_n = (100 - (100 / (1 + avg14_g / avg14_l))) / 100
    avg5_g = float(np.mean(gains[-5:])) if len(gains) >= 5 else avg14_g
    avg5_l = max(float(np.mean(losses[-5:])) if len(gains) >= 5 else avg14_l, 1e-8)
    rsi5_n = (100 - (100 / (1 + avg5_g / avg5_l))) / 100

    # Stochastic
    low14  = float(np.min(lows[-14:]))
    high14 = float(np.max(highs[-14:]))
    sk_raw = 100 * (c - low14) / (high14 - low14 + 1e-8)
    sk = sk_raw / 100
    sd = ema_arr(np.array([sk_raw] * 3), 3) / 100
    sk_arr = np.array([
        100 * (closes[i] - float(np.min(lows[max(0,i-13):i+1]))) /
              (float(np.max(highs[max(0,i-13):i+1])) - float(np.min(lows[max(0,i-13):i+1])) + 1e-8)
        for i in [-5, -4, -3, -2, -1]
        if len(closes) + i >= 14 and i + 1 > 0
    ])
    sk_m_sd_n = float((sk_raw - np.mean(sk_arr)) / (np.std(sk_arr) + 1e-8)) if len(sk_arr) > 1 else 0

    # WillR
    willr_n = -100 * (high14 - c) / (high14 - low14 + 1e-8)

    # Momentum
    mom1_n  = float((c - closes[-2])  / closes[-2]  * 100) if n >= 2  else 0
    mom5_n  = float((c - closes[-6])  / closes[-6]  * 100) if n >= 6  else 0
    mom10_n = float((c - closes[-11]) / closes[-11] * 100) if n >= 11 else 0
    mom15_n = float((c - closes[-16]) / closes[-16] * 100) if n >= 16 else 0

    # Range
    range_lower = float(np.min(lows[-20:]))
    range_upper = float(np.max(highs[-20:]))
    range_middle = (range_upper + range_lower) / 2
    range_pos_n = (c - range_lower) / (range_upper - range_lower + 1e-8)

    # Taker
    taker_n   = float(taker_arr[-1]) if len(taker_arr) > 0 else 0.5
    taker_avg = float(np.mean(taker_arr[-20:])) if len(taker_arr) >= 20 else taker_n

    # Vol delta
    vol_delta_n = float(np.mean(np.diff(vols[-6:]))) / (float(np.std(vols[-6:])) + 1e-8)

    # EMA features
    ec_raw = (ema5 - ema20) / c * 100
    macd_n = (ema12 - ema26) / c * 100
    price_slope = float(np.polyfit(range(20), closes[-20:], 1)[0]) / c * 100

    # RSI bins
    rsi14_val = rsi14_n * 100
    rsi5_val  = rsi5_n  * 100

    # Build feature dict
    f.update({
        "atr_n": atr_n, "bb_pos_n": float(bb_pos_n),
        "consec_up": float(consec_up), "consec_down": float(consec_down),
        "consec_up_only": float(consec_up - consec_down),
        "consec_down_only": float(consec_down - consec_up),
        "ema5_n": float(ema5 / c), "ema20_n": float(ema20 / c),
        "ema_cross_raw": float(ec_raw),
        "ema_cross_9_20": float((ema9 - ema20) / c * 100),
        "ema_positive": 1.0 if ema5 > ema20 else 0.0,
        "ema_negative": 1.0 if ema5 < ema20 else 0.0,
        "price_vs_ema5_n":  float((c - ema5)  / c * 100),
        "price_vs_ema20_n": float((c - ema20) / c * 100),
        "price_vs_ema50_n": float((c - ema50) / c * 100),
        "macd_n": float(macd_n),
        "rsi14_n": float(rsi14_n), "rsi5_n": float(rsi5_n),
        "rsi5_m14": float(rsi5_n - rsi14_n),
        "rsi_30_40": 1.0 if 30 <= rsi14_val < 40 else 0.0,
        "rsi_50_60": 1.0 if 50 <= rsi14_val < 60 else 0.0,
        "rsi_60_70": 1.0 if 60 <= rsi14_val < 70 else 0.0,
        "rsi_70_100": 1.0 if rsi14_val >= 70 else 0.0,
        "rsi_extreme": 1.0 if rsi14_val < 30 or rsi14_val > 70 else 0.0,
        "rsi_oversold": 1.0 if rsi14_val < 40 else 0.0,
        "rsi_mom_bullish": 1.0 if rsi5_val > rsi14_val and closes[-1] > closes[-6] else 0.0,
        "rsi_mom_bearish": 1.0 if rsi5_val < rsi14_val and closes[-1] < closes[-6] else 0.0,
        "stoch_k_n": float(sk), "stoch_d_n": float(sd),
        "stoch_overbought": 1.0 if sk_raw > 80 else 0.0,
        "stoch_oversold": 1.0 if sk_raw < 20 else 0.0,
        "sk_m_sd_n": float(sk_m_sd_n),
        "willr_n": float(willr_n),
        "mom1_n": float(mom1_n), "mom5_n": float(mom5_n),
        "mom10_n": float(mom10_n), "mom15_n": float(mom15_n),
        "mom_positive": 1.0 if mom5_n > 0 else 0.0,
        "mom_negative": 1.0 if mom5_n < 0 else 0.0,
        "mom_flat": 1.0 if abs(mom5_n) < 0.1 else 0.0,
        "mom_mixed": 1.0 if mom5_n * mom10_n < 0 else 0.0,
        "range_lower": float((c - range_lower) / (range_upper - range_lower + 1e-8)),
        "range_upper": float((range_upper - c) / (range_upper - range_lower + 1e-8)),
        "range_middle": float((c - range_middle) / c * 100),
        "range_pos_n": float(range_pos_n),
        "taker_n": float(taker_n), "taker_avg_n": float(taker_avg),
        "obi_positive": 1.0 if taker_n > taker_avg else 0.0,
        "vol_delta_n": float(vol_delta_n),
        "vol_delta_pos": 1.0 if vol_delta_n > 0 else 0.0,
        "vol_delta_neg": 1.0 if vol_delta_n < 0 else 0.0,
        "vol_drifting_up": 1.0 if np.mean(vols[-5:]) > np.mean(vols[-10:-5]) else 0.0,
        "vol_drifting_down": 1.0 if np.mean(vols[-5:]) < np.mean(vols[-10:-5]) else 0.0,
        "hour_utc": hour_utc,
        "hour_cos": math.cos(2 * math.pi * hour_utc / 24),
        "hour_sin": math.sin(2 * math.pi * hour_utc / 24),
        "hour_good": 1.0 if hour_utc in (1, 2, 4, 10, 11) else 0.0,
        "hour_bad":  1.0 if hour_utc in (0, 6, 7, 8, 9, 17) else 0.0,
        "fng": float(fng), "fng_extreme_fear": 1.0 if fng < 20 else 0.0,
        "fng_fear": 1.0 if fng < 40 else 0.0,
        "ls_ratio": float(ls_ratio),
        "ls_bullish": 1.0 if 1.1 <= ls_ratio < 1.3 else 0.0,
        "ls_crowded": 1.0 if ls_ratio >= 1.3 else 0.0,
        "fear_crowded_contrarian": 1.0 if fng < 25 and ls_ratio > 2.0 else 0.0,
        "price_slope20_n": float(price_slope),
    })
    return f

def ml_predict_v46(klines_dict, times, slot_ts, fng, ls_ratio):
    """Predict with v46 (GB_d3, 67F). Returns (direction, confidence)."""
    model, scaler, features = load_v46()
    if model is None or scaler is None:
        return "N/A", 0.0

    feats = build_v46_features(klines_dict, times, slot_ts, fng, ls_ratio)
    x = np.array([[feats.get(feat, 0.0) for feat in features]])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        x_s = scaler.transform(x)
        dir_int = int(model.predict(x_s)[0])
        prob = model.predict_proba(x_s)[0]
        conf = float(max(prob))
        direction = "UP" if dir_int == 1 else "DOWN"
        return direction, conf
    except Exception as e:
        print(f"   ⚠️ v46 predict error: {e}")
        return "N/A", 0.0

def ml_predict_v42(klines_dict, times, slot_ts):
    """Predict with v42 (FLAML LR, 43F). Returns (direction, confidence)."""
    model, scaler, features = load_v42()
    if model is None or scaler is None:
        return "N/A", 0.0

    closes = np.array([klines_dict[t]["close"] for t in times])
    highs  = np.array([klines_dict[t]["high"]  for t in times])
    lows   = np.array([klines_dict[t]["low"]   for t in times])
    vols   = np.array([klines_dict[t]["volume"] for t in times])
    taker_arr = np.array([klines_dict[t].get("taker_buy_vol", 0) for t in times])
    c = float(closes[-1])
    n = len(closes)
    ts = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    hour_utc = ts.hour

    ema5  = ema_arr(closes, 5)
    ema9  = ema_arr(closes, 9)
    ema20 = ema_arr(closes, 20)
    ema50 = ema_arr(closes, 50)
    ema12 = ema_arr(closes, 12)
    ema26 = ema_arr(closes, 26)

    trs = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    atr14 = float(np.mean(trs[-14:])) / c * 100 if len(trs) >= 14 else 0.001
    atr5  = float(np.mean(trs[-5:]))  / c * 100 if len(trs) >= 5  else atr14

    sma20 = float(np.mean(closes[-20:]))
    std20 = float(np.std(closes[-20:]))
    bb_pos = (c - (sma20 - 2*std20)) / (4*std20 + 1e-8)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg14_g = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
    avg14_l = max(float(np.mean(losses[-14:])) if len(gains) >= 14 else float(np.mean(gains)), 1e-8)
    rsi14 = (100 - (100 / (1 + avg14_g / avg14_l))) / 100
    avg5_g = float(np.mean(gains[-5:])) if len(gains) >= 5 else avg14_g
    avg5_l = max(float(np.mean(losses[-5:])) if len(gains) >= 5 else avg14_l, 1e-8)
    rsi5 = (100 - (100 / (1 + avg5_g / avg5_l))) / 100

    low14  = float(np.min(lows[-14:]))
    high14 = float(np.max(highs[-14:]))
    sk_raw = 100 * (c - low14) / (high14 - low14 + 1e-8)
    sd = ema_arr(np.array([sk_raw]*3), 3)

    consec_up = 0
    for x in reversed(closes[:-1]):
        if x < closes[-1]: consec_up += 1
        else: break

    rolling_high = float(np.max(closes[-30:]))
    rolling_low  = float(np.min(closes[-30:]))
    daily_range = (c - float(closes[0])) / float(closes[0]) * 100 if n > 0 else 0

    rets5  = np.diff(closes[-6:]) / closes[-6:-1]
    rets10 = np.diff(closes[-11:]) / closes[-11:-1]
    vol_ratio = float(np.std(rets5) / (np.std(rets10) + 1e-8))

    taker_ratio = float(np.mean(taker_arr[-5:])) if len(taker_arr) >= 5 else float(np.mean(taker_arr))
    obi_taker = float(taker_arr[-1] / (np.mean(taker_arr) + 1e-8) - 1)

    trend = abs(ema5 - ema20) / c * 100
    vwap = float(np.sum(((highs + lows + closes) / 3 * vols)[-20:]) / (np.sum(vols[-20:]) + 1e-8))

    f = {
        "atr14": atr14, "atr5": atr5, "bb_position": float(bb_pos),
        "consec_up": float(consec_up / 10.0),
        "daily_range_pos": float(daily_range / 100),
        "day_of_week": float(ts.weekday() / 6.0),
        "ema20_n": float(ema20 / c), "ema5_n": float(ema5 / c),
        "ema_cross_9_20": float(max(min((ema9 - ema20) / c, 0.01), -0.01) / 0.01),
        "ema_cross_raw": float(max(min((ema5 - ema20) / c, 0.01), -0.01) / 0.01),
        "hour_cos": math.cos(2 * math.pi * hour_utc / 24),
        "hour_sin": math.sin(2 * math.pi * hour_utc / 24),
        "hour_utc": float(hour_utc),
        "inside_bar": 0.0, "is_asia": 1.0 if hour_utc >= 22 or hour_utc <= 9 else 0.0,
        "is_hammer": 0.0, "is_shooting_star": 0.0,
        "is_us": 1.0 if 13 <= hour_utc <= 21 else 0.0,
        "is_weekend": 1.0 if ts.weekday() >= 5 else 0.0,
        "macd": float(max(min((ema12 - ema26) / c * 100, 1.0), -1.0)),
        "mom1": float(max(min((c - closes[-2]) / closes[-2] * 100, 0.005), -0.005) / 0.005),
        "mom3": float(max(min((c - closes[-4]) / closes[-4] * 100, 0.005), -0.005) / 0.005),
        "mom5": float(max(min((c - closes[-6]) / closes[-6] * 100, 0.005), -0.005) / 0.005),
        "mom10": float(max(min((c - closes[-11]) / closes[-11] * 100, 0.01), -0.01) / 0.01),
        "obi_taker": obi_taker,
        "price_vs_ema20": float(max(min((c - ema20) / c, 0.01), -0.01) / 0.01),
        "price_vs_ema5":  float(max(min((c - ema5)  / c, 0.005), -0.005) / 0.005),
        "price_vs_ema50": float(max(min((c - ema50) / c, 0.02), -0.02) / 0.02),
        "price_vs_rolling_high": float((c - rolling_high) / c * 100),
        "price_vs_rolling_low":  float((c - rolling_low)  / c * 100),
        "ret_std5":  float(min(float(np.std(rets5)  * 100), 0.005) / 0.005),
        "ret_std10": float(min(float(np.std(rets10) * 100), 0.01)  / 0.01),
        "rsi14": float(rsi14), "rsi5": float(rsi5),
        "rsi5_minus_14": float(rsi5 - rsi14),
        "stoch_d": float(sd / 100), "stoch_k": float(sk_raw / 100),
        "taker_ratio": taker_ratio,
        "taker_ratio5": float(np.mean(taker_arr[-5:])) if len(taker_arr) >= 5 else taker_ratio,
        "trend_strength": float(trend),
        "vol_ratio": float(min(vol_ratio, 5.0) / 5.0),
        "vol_ratio20": float(min(vol_ratio, 5.0) / 5.0),
        "vwap_deviation": float((c - vwap) / c * 100),
    }

    x = np.array([[f.get(feat, 0.0) for feat in features]])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        x_s = scaler.transform(x)
        dir_int = int(model.predict(x_s)[0])
        prob = model.predict_proba(x_s)[0]
        conf = float(max(prob))
        direction = "UP" if dir_int == 1 else "DOWN"
        return direction, conf
    except Exception as e:
        print(f"   ⚠️ v42 predict error: {e}")
        return "N/A", 0.0



def predict_beta_v1(klines_dict, times, slot_ts, fng, ls_ratio):
    """Predict with Beta V1 (RF_d3, no hour bias, HO AUC=0.6326)."""
    model, scaler, features = load_beta_v1()
    if model is None: return "N/A", 0.0

    closes = np.array([klines_dict[t]["close"] for t in times])
    highs  = np.array([klines_dict[t]["high"]  for t in times])
    lows   = np.array([klines_dict[t]["low"]   for t in times])
    vols   = np.array([klines_dict[t]["volume"] for t in times])
    taker_arr = np.array([klines_dict[t].get("taker_buy_vol", 0) for t in times])
    c = float(closes[-1])
    n = len(closes)

    def ema_a(arr, per):
        k = 2/(per+1); out=float(arr[0])
        for v in arr[1:]: out=float(v)*k+out*(1-k)
        return out

    # RSI
    deltas = np.diff(closes)
    gains=np.where(deltas>0,deltas,0.0); losses=np.where(deltas<0,-deltas,0.0)
    avg14_g=float(np.mean(gains[-14:])); avg14_l=max(float(np.mean(losses[-14:])),1e-8)
    rsi14=100-(100/(1+avg14_g/avg14_l))
    avg5_g=float(np.mean(gains[-5:])); avg5_l=max(float(np.mean(losses[-5:])),1e-8)
    rsi5=100-(100/(1+avg5_g/avg5_l))

    # Stochastic
    low14=float(np.min(lows[-14:])); high14=float(np.max(highs[-14:]))
    sk_raw=100*(c-low14)/(high14-low14+1e-8)
    sk=sk_raw/100

    # WillR
    willr=-100*(high14-c)/(high14-low14+1e-8)

    # BB
    sma20=float(np.mean(closes[-20:])); std20=float(np.std(closes[-20:]))
    bb=(c-(sma20-2*std20))/(4*std20+1e-8)

    # ATR
    trs=np.maximum(highs[1:]-lows[1:],np.maximum(abs(highs[1:]-closes[:-1]),abs(lows[1:]-closes[:-1])))
    atr=float(np.mean(trs[-14:]))/c*100

    # Momentum
    mom5=(c-closes[-6])/closes[-6]*100

    # OBI
    taker_n=float(taker_arr[-1]) if len(taker_arr)>0 else 0.5
    taker_avg=float(np.mean(taker_arr[-20:])) if len(taker_arr)>=20 else taker_n

    # L/S ratio
    ls = float(ls_ratio)

    # Build 35 features matching Beta V1 training
    def nf(val): return float(val)

    def build_beta_features():
        rsi_n  = nf(min(max(rsi14/100, 0), 1))
        ls_n   = nf(min(max(ls/2.5, 0), 1))
        stoch_n = nf(min(max(sk, 0), 1))
        willr_n = nf(min(max((willr+100)/100, 0), 1))
        bb_n   = nf(min(max(bb, 0), 1))
        atr_n  = nf(min(max(atr/0.01, 0), 1))
        mom5_n = nf(min(max(mom5/0.5, -1), 1))
        obi_n  = nf(min(max(taker_n+0.5, 0), 1))

        return [
            rsi_n, ls_n, stoch_n, willr_n, bb_n, atr_n, mom5_n, obi_n,
            1.0 if rsi14 < 30 else 0.0,       # rsi_os
            1.0 if rsi14 < 40 else 0.0,          # rsi_weak_os
            1.0 if 40 <= rsi14 <= 60 else 0.0,   # rsi_neutral
            1.0 if 60 < rsi14 <= 70 else 0.0,    # rsi_weak_ob
            1.0 if rsi14 > 70 else 0.0,          # rsi_ob
            1.0 if rsi14 < 20 else 0.0,          # rsi_extreme_os
            1.0 if ls < 0.5 else 0.0,            # ls_crowded_short
            1.0 if 0.5 <= ls < 0.9 else 0.0,    # ls_moderate_short
            1.0 if 0.9 <= ls <= 1.1 else 0.0,   # ls_neutral
            1.0 if 1.1 <= ls < 1.3 else 0.0,    # ls_moderate_long
            1.0 if ls >= 1.3 else 0.0,           # ls_crowded_long
            1.0 if mom5 < -0.2 else 0.0,         # mom_strong_neg
            1.0 if mom5 < 0 else 0.0,            # mom_neg
            1.0 if mom5 > 0 else 0.0,             # mom_pos
            1.0 if sk < 20 else 0.0,             # stoch_os
            1.0 if sk > 80 else 0.0,              # stoch_ob
            1.0 if willr < -80 else 0.0,          # willr_os
            1.0 if bb < 0.2 else 0.0,             # bb_lower
            1.0 if bb > 0.8 else 0.0,            # bb_upper
            1.0,  # range_tight (placeholder)
            1.0,  # range_wide (placeholder)
            # Composite signals
            (1.0 if rsi14 < 30 else 0.0) * (1.0 if 1.1 <= ls < 1.3 else 0.0),  # sig_up_RSI_LS
            (1.0 if 40 <= rsi14 <= 60 else 0.0) * (1.0 if 0.5 <= ls < 0.9 else 0.0),  # sig_down_RSI_LS
            (1.0 if rsi14 < 30 else 0.0) * (1.0 if sk < 20 else 0.0),  # sig_up_RSI_STOCH
            (1.0 if sk < 20 else 0.0) * (1.0 if mom5 < 0 else 0.0),  # sig_up_STOCH_MOM
            (1.0 if ls < 0.5 else 0.0) * (1.0 if rsi14 < 30 else 0.0),  # sig_up_LS_RSI
            1.0,  # sig_down_RANGE (placeholder)
        ]

    feat_vec = np.array([build_beta_features()])
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if scaler:
            feat_s = scaler.transform(feat_vec)
        else:
            feat_s = feat_vec
        dir_int = int(model.predict(feat_s)[0])
        prob = model.predict_proba(feat_s)[0]
        conf = float(max(prob))
        direction = "UP" if dir_int == 1 else "DOWN"
        return direction, conf
    except Exception as e:
        print(f"   ⚠️ Beta V1 predict error: {e}")
        return "N/A", 0.0


# ─── RISK MANAGEMENT (v5.1 style) ──────────────────────────────────────────
# ATR dynamic confidence
MIN_CONF_HIGH = 0.51   # normal
MIN_CONF_ATR  = 0.80   # high volatility

def get_dynamic_confidence(atr_ratio):
    """ATR-based dynamic confidence threshold."""
    if atr_ratio > 1.5:
        return 0.80
    elif atr_ratio > 1.2:
        return 0.60
    else:
        return 0.51

# Kelly bet sizing
def kelly_bet(win_rate=0.61, reward_ratio=0.97):
    """Kelly criterion. Returns fraction of bankroll."""
    p = win_rate
    b = reward_ratio
    if b <= 0 or p <= 0:
        return 0.01
    kelly = (p * (b + 1) - 1) / b
    kelly_half = kelly * 0.5
    if p < 0.5:
        return 0.01
    return min(max(kelly_half, 0.01), 0.05)

# State management
STATE_FILE = DATA / "trade_state_v5A.json"

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"consec_loss": 0, "pause_until": 0, "day_start_bal": 0, "day": "", "total_pnl": 0}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ─── CLOB / POLYMARKET CLIENT ──────────────────────────────────────────────
def init_client():
    key_path = Path(os.path.expanduser("~/.openclaw/eth_key.txt"))
    creds_path = Path(os.path.expanduser("~/.openclaw/fresh_creds.json"))
    if not key_path.exists() or not creds_path.exists():
        return None, None
    key = key_path.read_text().strip()
    with open(creds_path) as f:
        c = json.load(f)
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
    creds = ApiCreds(c["api_key"], c["api_secret"], c["api_passphrase"])
    client = ClobClient("https://clob.polymarket.com", chain_id=137, key=key,
                         creds=creds, signature_type=2,
                         funder="0x8d8BA13d2c3D1935bF0b8Bd2052AC73e8E329376")
    return client, BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)

def get_balance(client, params):
    try:
        b = client.get_balance_allowance(params=params)
        return int(b.get("balance", "0")) / 1e6
    except:
        return 0.0

# ─── LOGGING ────────────────────────────────────────────────────────────────
def log_decision(dec):
    """Log decision to SQLite decisions table."""
    import sqlite3
    from pathlib import Path
    DB = Path(__file__).parent / "data" / "btc5m_research.db"
    conn = sqlite3.connect(str(DB))
    c = conn.cursor()
    # Auto-create table if missing columns (v46 was added later)
    c.execute("PRAGMA table_info(decisions)")
    existing_cols = [r[1] for r in c.fetchall()]
    needed_cols = ["v46_dir", "v46_conf", "cl_lag"]
    for col in needed_cols:
        if col not in existing_cols:
            try:
                c.execute(f"ALTER TABLE decisions ADD COLUMN {col} TEXT")
            except:
                pass
    try:
        cols = ["slot","hour_utc","status","sim_mode","btc_price",
                "final_dir","bet_amount","entry_price",
                "v42_dir","v42_conf","v43_dir","v43_conf","v46_dir","v46_conf",
                "v17_dir","v17_conf","v35_dir","v35_conf","v41_dir","v41_conf","v4_score","v45_dir","v45_conf",
                "rsi","macd","ema_cross","stoch","willr","bb_pos","atr",
                "mom_5m","mom_1m","mom_15m","taker",
                "fng_value","fng_class","ls_ratio",
                "slot_range","vol_class","vol_mult","conf_mult","combined_mult",
                "div_up","div_down","div_note","clob_override",
                "cl_lag",
                "note","actual","pnl"]
        vals = [
            dec.get("slot"), dec.get("hour_utc"), dec.get("status","WAIT"),
            dec.get("sim_mode",False), dec.get("btc_price"),
            dec.get("final_dir","WAIT"), dec.get("bet_amount",0),
            dec.get("entry_price",0),
            dec.get("v42_dir","N/A"), dec.get("v42_conf",0),
            dec.get("v43_dir","N/A"), dec.get("v43_conf",0),
            dec.get("v46_dir","N/A"), dec.get("v46_conf",0),
            dec.get("v17_dir","N/A"), dec.get("v17_conf",0),
            dec.get("v35_dir","N/A"), dec.get("v35_conf",0),
            dec.get("v41_dir","N/A"), dec.get("v41_conf",0),
            dec.get("v4_score",0),
            dec.get("v45_dir","N/A"), dec.get("v45_conf",0),
            dec.get("rsi"), dec.get("macd"), dec.get("ema_cross"),
            dec.get("stoch"), dec.get("willr"), dec.get("bb_pos"), dec.get("atr"),
            dec.get("mom_5m"), dec.get("mom_1m"), dec.get("mom_15m"), dec.get("taker"),
            dec.get("fng_value"), dec.get("fng_class",""), dec.get("ls_ratio"),
            dec.get("slot_range"), dec.get("vol_class",""),
            dec.get("vol_mult",1.0), dec.get("conf_mult",1.0), dec.get("combined_mult",1.0),
            dec.get("div_up",0), dec.get("div_down",0), dec.get("div_note",""),
            int(dec.get("clob_override",False) or 0),
            dec.get("cl_lag","N/A"),
            dec.get("note",""),
            dec.get("actual"), dec.get("pnl",0),
        ]
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        sql = f"INSERT INTO decisions ({col_names}) VALUES ({placeholders})"
        c.execute(sql, vals)
        conn.commit()
    except Exception:
        pass
    conn.close()


# ─── MAIN TRADING FUNCTION ──────────────────────────────────────────────────
def run():
    now = int(time.time())
    now_dt = datetime.now(timezone.utc)
    hour_utc = now_dt.hour

    print(f"\n{'='*60}")
    print(f"  [{now_dt.strftime('%H:%M:%S')}] BTC-5m Option A Strategy v5A")
    print(f"  Rules PRIMARY | ML CONFIRMS | Market Neutral Filter ON")
    print(f"{'='*60}")

    state = load_state()
    today = now_dt.strftime("%Y-%m-%d")

    # Daily reset
    if state.get("day") != today:
        state["day"] = today
        state["consec_loss"] = 0
        state["pause_until"] = 0
        state["day_start_bal"] = 0
        state["total_pnl"] = 0
        save_state(state)

    client, bal_params = init_client()

    # Balance check
    balance = get_balance(client, bal_params) if client else 0.0
    if balance <= 0:
        print(f"   ⚠️ No balance or not in live mode")
        sim_mode = True
    else:
        sim_mode = state.get("sim_mode", False)
        if state["day_start_bal"] == 0:
            state["day_start_bal"] = balance
            save_state(state)

    print(f"   Mode: {'SIM' if sim_mode else 'LIVE'} | Balance: ${balance:.2f}" if balance > 0 else f"   Mode: SIM (no wallet)")

    # ── Hard stops (v5.1 style) ────────────────────────────────────────────
    if balance > 0 and state["day_start_bal"] > 0:
        day_loss_pct = (state["day_start_bal"] - balance) / state["day_start_bal"]
        if day_loss_pct >= 0.10:
            print(f"   🛑 HARD STOP: Day loss {day_loss_pct:.1%} ≥ 10% | Pausing 2hr")
            state["pause_until"] = now + 2 * 3600
            save_state(state)
            return
        if now < state.get("pause_until", 0):
            resume = datetime.fromtimestamp(state["pause_until"], tz=timezone.utc).strftime("%H:%M")
            print(f"   ⏸️ Paused until {resume}")
            return

    # ── Fetch data ────────────────────────────────────────────────────────
    klines_5m_dict, klines_5m_times = fetch_btc_klines_5m(limit=200)
    closes_1m = fetch_btc_klines_1m(limit=60)
    fng_val = fetch_fng()
    ls_ratio = fetch_ls_ratio()

    if len(klines_5m_times) < 30 or len(closes_1m) < 30:
        print("   ❌ Insufficient kline data")
        return

    slot = (now // 300) * 300
    btc_price = float(klines_5m_dict[klines_5m_times[-1]]["close"])

    # Latest 5m bar for slot_range
    if klines_5m_times[-1] == slot:
        slot_bar = klines_5m_dict[klines_5m_times[-1]]
        slot_range = slot_bar["high"] - slot_bar["low"]
    else:
        slot_range = None

    closes_5m = [klines_5m_dict[t]["close"] for t in klines_5m_times]
    highs_5m  = [klines_5m_dict[t]["high"]  for t in klines_5m_times]
    lows_5m   = [klines_5m_dict[t]["low"]   for t in klines_5m_times]
    vols_5m   = [klines_5m_dict[t]["volume"] for t in klines_5m_times]

    rsi_val = calc_rsi(closes_5m)
    sk_val, sd_val = calc_stoch(highs_5m, lows_5m, closes_5m)
    atr14, atr_mean, atr_ratio = calc_atr(highs_5m, lows_5m, closes_5m), \
                                  np.mean([calc_atr(highs_5m, lows_5m, closes_5m)]), \
                                  calc_atr(highs_5m, lows_5m, closes_5m) / max(np.mean([calc_atr(highs_5m, lows_5m, closes_5m)]), 1)

    # ATR ratio
    trs_all = []
    for i in range(1, len(closes_5m)):
        trs_all.append(max(
            highs_5m[i] - lows_5m[i],
            abs(highs_5m[i] - closes_5m[i-1]),
            abs(lows_5m[i] - closes_5m[i-1])
        ))
    trs_all = np.array(trs_all)
    atr14_val = float(np.mean(trs_all[-14:]))
    atr_mean_val = float(np.mean(trs_all))
    atr_ratio_val = atr14_val / max(atr_mean_val, 1)

    # EMA for Market Neutral Filter
    ema9_5m  = ema_arr(np.array(closes_5m), 9)
    ema21_5m = ema_arr(np.array(closes_5m), 21)
    ema_diff_pct = abs(ema9_5m - ema21_5m) / ema21_5m * 100

    # MACD
    macd_val = ema_arr(np.array(closes_5m), 12) - ema_arr(np.array(closes_5m), 26)

    willr_val = -100 * (float(np.max(highs_5m[-14:])) - closes_5m[-1]) / \
                 (float(np.max(highs_5m[-14:])) - float(np.min(lows_5m[-14:])) + 1e-8)

    bb_sma = np.mean(closes_5m[-20:])
    bb_std = np.std(closes_5m[-20:])
    bb_pos = (closes_5m[-1] - (bb_sma - 2*bb_std)) / (4*bb_std + 1e-8)

    mom_5m_val = (closes_5m[-1] - closes_5m[-6]) / closes_5m[-6] * 100
    mom_1m_val = (closes_5m[-1] - closes_5m[-2]) / closes_5m[-2] * 100

    print(f"   BTC: ${btc_price:,.0f} | RSI={rsi_val:.0f} | ATR={atr_ratio_val:.2f}x | EMA_diff={ema_diff_pct:.2f}%")
    slot_range_str = f"{slot_range:.0f}" if slot_range else "?"
    print(f"   F&G={fng_val:.0f} | LS={ls_ratio:.2f} | Range=${slot_range_str}")

    # ── STEP 1: RULE SIGNAL (PRIMARY) ─────────────────────────────────────
    rule_dir, rule_conf, rule_reason = rule_signal(
        rsi_val, ls_ratio, ema_diff_pct, hour_utc,
        (slot_range / btc_price * 100) if slot_range else None
    )
    print(f"   📋 RULE SIGNAL: {rule_dir} (conf={rule_conf:.2f}) | {rule_reason}")

    # ── STEP 2: ML PREDICTIONS ─────────────────────────────────────────────
    v42_dir, v42_conf = ml_predict_v42(klines_5m_dict, klines_5m_times, slot)
    beta_dir, beta_conf = predict_beta_v1(klines_5m_dict, klines_5m_times, slot, fng_val, ls_ratio)
    print(f"   ⭐ v42 ML: {v42_dir} {v42_conf:.2%}")
    print(f"   🟢 Beta V1: {beta_dir} {beta_conf:.2%}")

    # ── STEP 2b: CHAINLINK ORACLE CHECK ────────────────────────────────────
    cl_alert, cl_dev, cl_lag, cl_msg = chainlink_oracle_check(btc_price, mom_5m_val)
    print(f"   ⛓ Oracle: {cl_msg}")
    if cl_alert >= 2:
        print(f"   🔴 ORACLE BLOCK — diff {cl_dev:+.2f}% exceeds 1.0% threshold")
        note_parts.append(f"ORACLE BLOCK: diff={cl_dev:+.2f}%")
        final_dir = None
        decision = {
            "slot": slot, "hour_utc": hour_utc,
            "status": "WAIT", "sim_mode": sim_mode,
            "btc_price": round(btc_price, 2),
            "final_dir": "WAIT",
            "v42_dir": v42_dir, "v42_conf": v42_conf,
            "v46_dir": beta_dir, "v46_conf": beta_conf,
            "cl_lag": cl_lag,
            "note": "chainlink_extreme_deviation",
        }
        log_decision(decision)
        return

    # ── STEP 3: CASCADE — Rules PRIMARY, Beta V1 CONFIRMS ─────────────────
    # Enhanced with CL LAG: if CL is lagging Binance momentum, boost ML confidence
    final_dir = None
    note_parts = []
    ml_confirmed = False

    if rule_dir == "WAIT":
        final_dir = None
        note_parts.append(f"RULE WAIT: {rule_reason}")
        cascade_note = "Rule said WAIT"

    elif rule_dir in ("UP", "DOWN"):
        # Beta V1 is PRIMARY ML (HO AUC=0.6326, no hour bias)
        ml_dir = beta_dir if beta_dir != "N/A" else v42_dir
        ml_conf = beta_conf if beta_dir != "N/A" else v42_conf
        ml_src = "BetaV1" if beta_dir != "N/A" else "v42"

        # CL LAG boosts confidence: if CL is catching up to Binance momentum,
        # the momentum direction is STRONG and reliable
        cl_lag_boost = False
        if cl_lag in ("UP_MOMENTUM", "DOWN_MOMENTUM"):
            lag_dir = "UP" if cl_lag == "UP_MOMENTUM" else "DOWN"
            if lag_dir == rule_dir:
                cl_lag_boost = True
                ml_conf = min(ml_conf + 0.08, 0.70)  # Boost ML confidence
                note_parts.append(f"🚀 CL-LAG BOOST: {rule_dir} momentum confirmed +8%")

        if ml_dir == rule_dir and ml_conf >= 0.52:
            final_dir = rule_dir
            ml_confirmed = True
            boost_str = " [CL-LAG BOOST]" if cl_lag_boost else ""
            note_parts.append(f"✅ RULE+{ml_src} AGREE: {rule_dir} | {ml_src}={ml_conf:.2%}{boost_str} | {rule_reason}")
            cascade_note = f"Rule+{ml_src} agree {rule_dir}{boost_str}"

        elif ml_dir != rule_dir and ml_dir != "N/A":
            # ML contradicts rule BUT if CL LAG agrees with rule, still trade
            if cl_lag_boost:
                final_dir = rule_dir
                note_parts.append(f"⚠️ ML({ml_dir}) ≠ RULE({rule_dir}) but CL-LAG confirms RULE → TRADE{boost_str}")
                cascade_note = f"CL-LAG overrides ML contradiction ({rule_dir})"
            else:
                final_dir = None
                note_parts.append(f"⏸️ RULE({rule_dir}) ≠ {ml_src}({ml_dir} {ml_conf:.2%}) → WAIT")
                cascade_note = f"ML contradicted rule ({rule_dir} vs {ml_dir})"

        else:
            final_dir = rule_dir
            note_parts.append(f"⚠️ RULE ONLY (ML N/A): {rule_dir} | {rule_reason}")
            cascade_note = f"Rule only {rule_dir}"

    print(f"   🎯 CASCADE: {cascade_note}")

    # ── STEP 4: Bet sizing ────────────────────────────────────────────────
    min_conf = get_dynamic_confidence(atr_ratio_val)
    bet_pct_base = kelly_bet(win_rate=0.61, reward_ratio=0.97)

    if final_dir is None:
        print(f"   ⏸️ Final decision: WAIT")
        print(f"   💀 我的策略：「在市場中性時耐心等待」——現實：「我只是牆上的一張示意圖。」")
        decision = {
            "slot": slot, "hour_utc": hour_utc,
            "status": "WAIT", "sim_mode": sim_mode,
            "btc_price": round(btc_price, 2),
            "final_dir": "WAIT",
            "v42_dir": v42_dir, "v42_conf": v42_conf,
            "v46_dir": beta_dir, "v46_conf": beta_conf,
            "v4_score": rule_conf,
            "rsi": round(rsi_val, 2),
            "macd": round(macd_val, 6),
            "ema_cross": round((ema9_5m - ema21_5m) / ema21_5m * 100, 4),
            "stoch": round(sk_val, 2),
            "willr": round(willr_val, 2),
            "bb_pos": round(bb_pos, 4),
            "atr": round(atr_ratio_val, 4),
            "mom_5m": round(mom_5m_val, 4),
            "mom_1m": round(mom_1m_val, 4),
            "mom_15m": 0,
            "fng_value": fng_val,
            "ls_ratio": ls_ratio,
            "slot_range": round((slot_range / btc_price * 100) if slot_range else 0, 2),
            "cl_lag": cl_lag,
            "note": " | ".join(note_parts),
        }
        log_decision(decision)
        return

    # Determine bet size
    consec = state.get("consec_loss", 0)
    if consec >= 5:
        bet_pct = bet_pct_base * 0.25
        print(f"   ⚠️ Consec loss {consec}≥5 → ×0.25")
    elif consec >= 3:
        bet_pct = bet_pct_base * 0.5
        print(f"   ⚠️ Consec loss {consec}≥3 → ×0.5")
    else:
        bet_pct = bet_pct_base

    if atr_ratio_val > 1.5:
        bet_pct *= 0.8
        print(f"   ⚠️ High ATR {atr_ratio_val:.2f}x → ×0.8")

    bet_amount = balance * bet_pct if balance > 0 else 0

    print(f"   💰 Bet: ${bet_amount:.2f} ({bet_pct:.1%}) | ATR conf≥{min_conf:.0%}")

    # ── STEP 5: Execute trade ─────────────────────────────────────────────
    market = find_polymarket_slot(now)
    if not market:
        print(f"   ⚠️ No Polymarket slot found")
        return

    up_price = market["up_price"]
    mkt_direction = "UP" if up_price >= 0.5 else "DOWN"
    print(f"   🎯 Market: {market['question'][:40]}")
    print(f"   Up={up_price:.3f} ({'UP' if mkt_direction=='UP' else 'DOWN'} market)")

    if sim_mode or balance < 5:
        print(f"   🟡 SIM MODE — would {'BUY' if final_dir=='UP' else 'SELL'} {final_dir}")
        print(f"   💀 紙上交易最完美的部分？利潤不用扣除現實世界的「插針」和「流動性風險」。")
        decision = {
            "slot": slot, "hour_utc": hour_utc,
            "status": "SIM", "sim_mode": True,
            "btc_price": round(btc_price, 2),
            "final_dir": final_dir,
            "bet_amount": round(bet_amount, 2),
            "entry_price": round(up_price, 4),
            "v42_dir": v42_dir, "v42_conf": round(v42_conf, 4),
            "v46_dir": beta_dir, "v46_conf": round(beta_conf, 4),
            "v4_score": round(rule_conf, 4),
            "rsi": round(rsi_val, 2),
            "macd": round(macd_val, 6),
            "ema_cross": round((ema9_5m - ema21_5m) / ema21_5m * 100, 4),
            "stoch": round(sk_val, 2),
            "willr": round(willr_val, 2),
            "bb_pos": round(bb_pos, 4),
            "atr": round(atr_ratio_val, 4),
            "mom_5m": round(mom_5m_val, 4),
            "mom_1m": round(mom_1m_val, 4),
            "fng_value": fng_val,
            "ls_ratio": ls_ratio,
            "slot_range": round((slot_range / btc_price * 100) if slot_range else 0, 2),
            "cl_lag": cl_lag,
            "note": " | ".join(note_parts),
        }
        log_decision(decision)
        return

    # ── LIVE TRADE ────────────────────────────────────────────────────────
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
    from py_clob_client.order_builder.constants import BUY

    token = market["up_token"] if final_dir == "UP" else market["down_token"]
    mkt_px = up_price if final_dir == "UP" else (1 - up_price)

    try:
        ob = client.get_order_book(token)
        best_px = float(ob.asks[0].price) if ob.asks else mkt_px
    except:
        best_px = mkt_px

    best_px = min(max(best_px, 0.01), 0.99)
    size = max(market["min_size"], round(bet_amount / best_px, 2))
    cost = size * best_px

    print(f"   📌 {final_dir} @ {best_px:.3f} | size={size} | cost=${cost:.2f}")

    order = OrderArgs(token_id=token, price=best_px, size=size, side=BUY)
    opts  = PartialCreateOrderOptions(tick_size=market["tick_size"])

    try:
        signed = client.create_order(order, opts)
        result = client.post_order(signed)
        if result.get("success"):
            order_id = result.get("orderID", "")
            tx_hash = result.get("transactionsHashes", [None])[0]
            print(f"   ✅ Order placed: {order_id}")
            print(f"   🔗 TX: {tx_hash}")
            print(f"   💀 區塊鏈從不說謊——但你的止損執行速度取決於Gas Fee夠不夠高。")

            decision = {
                "slot": slot, "hour_utc": hour_utc,
                "status": "SUCCESS", "sim_mode": False,
                "btc_price": round(btc_price, 2),
                "final_dir": final_dir,
                "bet_amount": round(cost, 2),
                "entry_price": round(best_px, 4),
                "v42_dir": v42_dir, "v42_conf": round(v42_conf, 4),
                "v46_dir": beta_dir, "v46_conf": round(beta_conf, 4),
                "v4_score": round(rule_conf, 4),
                "rsi": round(rsi_val, 2),
                "macd": round(macd_val, 6),
                "ema_cross": round((ema9_5m - ema21_5m) / ema21_5m * 100, 4),
                "stoch": round(sk_val, 2),
                "willr": round(willr_val, 2),
                "bb_pos": round(bb_pos, 4),
                "atr": round(atr_ratio_val, 4),
                "mom_5m": round(mom_5m_val, 4),
                "mom_1m": round(mom_1m_val, 4),
                "fng_value": fng_val,
                "ls_ratio": ls_ratio,
                "slot_range": round((slot_range / btc_price * 100) if slot_range else 0, 2),
                "cl_lag": cl_lag,
                "note": " | ".join(note_parts),
            }
            log_decision(decision)

            # Update balance
            new_bal = get_balance(client, bal_params)
            print(f"   💰 Balance: ${balance:.2f} → ${new_bal:.2f}")

            # Update state
            state["total_pnl"] = state.get("total_pnl", 0)
            save_state(state)

        else:
            print(f"   ❌ Order failed: {result}")
    except Exception as e:
        print(f"   ❌ Trade error: {e}")

if __name__ == "__main__":
    run()
