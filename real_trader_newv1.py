#!/usr/bin/env python3
"""
BTC-5m v5B — Beta V1 ML優先 + RSI方向修復 + UTC04加倉 + Fear&Greed + L/S Ratio
2026-04-12 HKT (基於市場數據修正)

設計哲學:
  「RSI 是市場方向領先指標，ML 是二級確認。
   順 RSI 市場慣性交易，逆 RSI 必須有超強 ML 信號。」

核心邏輯:
  1. RSI 前置過濾（v5B 修復）:
     - RSI<25  → UP（逆勢抄底）
     - RSI<35  → DOWN（市場慣性偏空，ML 服從）
     - RSI 25-65 → WAIT（中性區）
     - RSI>65  → UP（市場動能向上，ML 服從）
     - RSI>75  → DOWN（極端超買反轉）
  2. ML Cascade：Beta V1 PRIMARY → v42 → v39 → v35 → v17 → v7 → v6
  3. RSI+ML 方向整合：ML 服從 RSI 市場傾向
  4. CLOB 逆向：可翻轉方向（強衝突時）
  5. 熔斷：連虧3次/日虧40% → 硬停止
  6. 結算穿透：自動更新所有交易的WIN/LOSS
"""

import os, json, time, requests, pickle, numpy as np, math, random, socket
# [v5B] 全域 socket timeout — 防止 API 卡死
socket.setdefaulttimeout(15)
from datetime import datetime, timezone
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (ApiCreds, OrderArgs, PartialCreateOrderOptions,
                                         BalanceAllowanceParams, AssetType)
from py_clob_client.order_builder.constants import BUY

# ── 配置 ─────────────────────────────────────────────────────────────────────
HOST   = "https://clob.polymarket.com"
CHAIN  = 137
FUNDER = "0x8d8BA13d2c3D1935bF0b8Bd2052AC73e8E329376"
BASE   = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader")
LOG    = os.path.join(BASE, "logs/real_trades_log.jsonl")
STATE  = os.path.join(BASE, "data/trade_state.json")
CACHE  = os.path.join(BASE, "data/settlement_cache.json")
NEWV1_STATE = os.path.join(BASE, "data/trade_state_newv1.json")

# 模型路徑
MODEL_PATH_V4  = os.path.join(BASE, "data/ml_model_v4.pkl")
MODEL_PATH_V5B = os.path.join(BASE, "data/ml_model_v5b.pkl")
MODEL_PATH_V6  = os.path.join(BASE, "data/ml_model_v6.pkl")
MODEL_PATH_V7  = os.path.join(BASE, "data/ml_model_v7.pkl")
MODEL_PATH_V17 = os.path.join(BASE, "data/ml_model_v17.pkl")
MODEL_PATH_V18 = os.path.join(BASE, "data/ml_model_v18.pkl")
MODEL_PATH_V35   = os.path.join(BASE, "data/ml_model_v35.pkl")
MODEL_PATH_V39   = os.path.join(BASE, "data/ml_model_v39.pkl")
MODEL_PATH_V42   = os.path.join(BASE, "data/ml_model_v42_flaml.pkl")
SCALER_PATH_V4  = os.path.join(BASE, "data/scaler_v4.pkl")
SCALER_PATH_V5B = os.path.join(BASE, "data/scaler_v5b.pkl")
SCALER_PATH_V6  = os.path.join(BASE, "data/scaler_v6.pkl")
SCALER_PATH_V7  = os.path.join(BASE, "data/scaler_v7.pkl")
SCALER_PATH_V17 = os.path.join(BASE, "data/scaler_v17.pkl")
SCALER_PATH_V35   = os.path.join(BASE, "data/scaler_v35.pkl")
SCALER_PATH_V39   = os.path.join(BASE, "data/scaler_v39.pkl")
SCALER_PATH_V42   = os.path.join(BASE, "data/scaler_v42_flaml.pkl")
SCALER_PATH_BETA  = os.path.join(BASE, "data/scaler_beta_v1.pkl")
MODEL_PATH_BETA   = os.path.join(BASE, "data/ml_model_beta_v1.pkl")
FEATURES_PATH_V42 = os.path.join(BASE, "data/ml_features_v42.json")

# ════════════════════════════════════════════════════════════════════════════════
# [New V1.4] 風險參數 — 純倉位控制
# ════════════════════════════════════════════════════════════════════════════════
BASE_BET_PCT    = 0.10    # 基礎倉位: balance × 10% (Leo要求)
# ── [New V1.4] 預測模式開關 ───────────────────────────────────────────────
PREDICT_ONLY    = False    # True = 只跑預測不交易 (Leo要求收集數據)
# PREDICT_ONLY = False  # 正常交易模式
MAX_BET_USDC    = 50.0   # 每筆最大下注 ($)
MIN_BET_USDC    = 2.0    # 每筆最小下注 ($)
MAX_DAILY_STAKE = None   # [v1.4] 每日下注上限: 已移除（直接注釋掉檢查）
HARD_STOP_PCT   = 0.40   # [v1.3] 日虧熔斷: 40%
PAUSE_HOURS     = 3      # 連虧熔斷後暫停 (小時)
MAX_CONSEC_LOSS = 3      # 連虧觸發暫停
MIN_ENTRY_SECS  = 120    # 距收市最少秒數

# [v39 風控] v39 新模型上線 6 小時內，強制最多 10% balance
# v39 primary 上線時間：2026-04-02 00:05 HKT = 2026-04-01 16:05 UTC = 1775059500
# 6 小時後 = 2026-04-01 22:05 UTC = 1775066700
V39_SAFETY_END = 1775066700  # 2026-04-02 06:05 HKT
V39_MAX_BET_PCT = 0.10       # 6 小時內最多 10% balance

# ════════════════════════════════════════════════════════════════════════════════
# v33 (2026-03-30): 更新時段 Alpha（720筆真實數據）
# WR ≥ 60%: 1.00 | WR 55-59%: 0.75 | WR 50-54%: 0.50 | WR < 50%: BLOCK
# ════════════════════════════════════════════════════════════════════════════════
HOUR_BLOCK = {6, 8, 9, 17}  # WR < 40%，直接跳過

HOUR_BET_MULT = {
    # UTC: (WR%, n) → bet_multiplier
    # 2026-03-30 更新基於720筆真實結算
    0:  0.50,   # WR 44.2%, n=43
    1:  1.00,   # WR 60.9%, n=23  ⭐ UP from 0.25
    2:  0.50,   # WR 50.0%, n=18
    3:  0.50,   # WR 50.0%, n=18
    4:  1.00,   # WR 74.4%, n=39  ⭐⭐ #1 ALPHA
    5:  0.50,   # WR 51.4%, n=37
    6:  0.00,   # WR 35.3%, n=17  ❌ BLOCK
    7:  0.50,   # WR 47.8%, n=23
    8:  0.00,   # WR 39.4%, n=33  ❌ BLOCK
    9:  0.00,   # WR 39.1%, n=23  ❌ BLOCK
    10: 1.00,   # WR 60.0%, n=25  ⭐ UP from 0.25
    11: 0.75,   # WR 55.3%, n=47  UP from 0.25
    12: 0.50,   # WR 51.0%, n=49
    13: 0.50,   # WR 41.9%, n=31  DOWN from 1.00
    14: 0.75,   # WR 55.6%, n=27
    15: 0.75,   # WR 57.9%, n=38  UP from 0.25
    16: 0.75,   # WR 54.8%, n=31
    17: 0.00,   # WR 38.9%, n=18  ❌ BLOCK
    18: 0.50,   # WR 52.6%, n=38
    19: 0.50,   # WR 45.0%, n=40
    20: 0.50,   # WR 52.5%, n=40
    21: 0.50,   # WR 47.1%, n=34
    22: 0.50,   # WR 45.0%, n=20
    23: 0.50,   # WR 45.0%, n=20
}

# CLOB OBI 最佳區間過濾（720筆數據驗證）
# OBI -0.3~-0.1: WR 62.7% ⭐ 最佳切入點
# OBI +0.1~+0.3: WR 44.4% ⚠️ 避開
OBI_GOOD_RANGE = (-0.30, -0.10)  # 順勢增強區間
OBI_BAD_RANGE  = (0.10, 1.00)     # 逆向阻擋區間

# RSI 倉位係數（RSI 極端時降低倉位）
def rsi_bet_mult(rsi):
    if rsi >= 70 or rsi <= 30:
        return 0.50   # 極端市場，倉位減半
    if rsi >= 65 or rsi <= 35:
        return 0.75
    return 1.0

# CLOB 逆向規則（已驗證：群眾永遠是錯的）
# obi > 0.20 → 市場走 DOWN (78%準確率)
# obi < -0.20 → 市場走 UP (90%準確率)
CLOB_STRONG_THRESH = 0.20
CLOB_WEAK_THRESH   = 0.10

# ════════════════════════════════════════════════════════════════════════════════
# 模型加載 (懶加載)
# ════════════════════════════════════════════════════════════════════════════════
_mv4 = _mv5b = _mv6 = _mv7 = _mv17 = _mv18 = _mv35 = _mv39 = _mv42 = _mbeta = None
_sv4 = _sv5b = _sv6 = _sv7 = _sv17 = _sv35 = _sv39 = _sv42 = _sbeta = None
_fv42 = None  # feature names for v42

def load_v4():
    global _mv4
    if _mv4 is None:
        if os.path.exists(MODEL_PATH_V4):
            with open(MODEL_PATH_V4, "rb") as f:
                _mv4 = pickle.load(f)
            print(f"🤖 v4 loaded")
    return _mv4

def load_v5b():
    global _mv5b
    if _mv5b is None:
        if os.path.exists(MODEL_PATH_V5B):
            with open(MODEL_PATH_V5B, "rb") as f:
                _mv5b = pickle.load(f)
            print(f"🤖 v5b loaded")
    return _mv5b

def load_v6():
    global _mv6
    if _mv6 is None:
        if os.path.exists(MODEL_PATH_V6):
            with open(MODEL_PATH_V6, "rb") as f:
                _mv6 = pickle.load(f)
            print(f"🤖 v6 loaded")
    return _mv6

def load_v7():
    global _mv7
    if _mv7 is None:
        if os.path.exists(MODEL_PATH_V7):
            with open(MODEL_PATH_V7, "rb") as f:
                _mv7 = pickle.load(f)
            print(f"🤖 v7 loaded")
    return _mv7

def load_v17():
    global _mv17
    if _mv17 is None:
        if os.path.exists(MODEL_PATH_V17):
            with open(MODEL_PATH_V17, "rb") as f:
                _mv17 = pickle.load(f)
            print(f"🤖 v17 loaded")
    return _mv17

def load_scaler_v35():
    global _sv35
    if _sv35 is None:
        if os.path.exists(SCALER_PATH_V35):
            with open(SCALER_PATH_V35, "rb") as f:
                _sv35 = pickle.load(f)
            print(f"🤖 v35 scaler loaded")
    return _sv35

def load_v35():
    global _mv35
    if _mv35 is None:
        if os.path.exists(MODEL_PATH_V35):
            with open(MODEL_PATH_V35, "rb") as f:
                _mv35 = pickle.load(f)
            print(f"🤖 v35 loaded")
    return _mv35

def load_scaler_v39():
    global _sv39
    if _sv39 is None:
        if os.path.exists(SCALER_PATH_V39):
            with open(SCALER_PATH_V39, "rb") as f:
                _sv39 = pickle.load(f)
            print(f"🤖 v39 scaler loaded")
    return _sv39

def load_v39():
    global _mv39
    if _mv39 is None:
        if os.path.exists(MODEL_PATH_V39):
            with open(MODEL_PATH_V39, "rb") as f:
                _mv39 = pickle.load(f)
            print(f"🤖 v39 loaded (LogisticRegression C=0.001, 43 features)")
    return _mv39

# ── FLAML v42 (PRIMARY) ──────────────────────────────────────────────────────
def load_v42():
    global _mv42
    if _mv42 is None:
        if os.path.exists(MODEL_PATH_V42):
            with open(MODEL_PATH_V42, "rb") as f:
                _mv42 = pickle.load(f)
            print(f"🤖 FLAML v42 loaded (extra_tree, 43F, CV=53.5%, 3756 samples)")
    return _mv42

def load_scaler_v42():
    global _sv42
    if _sv42 is None:
        if os.path.exists(SCALER_PATH_V42):
            with open(SCALER_PATH_V42, "rb") as f:
                _sv42 = pickle.load(f)
    return _sv42

def load_features_v42():
    global _fv42
    if _fv42 is None:
        if os.path.exists(FEATURES_PATH_V42):
            with open(FEATURES_PATH_V42) as f:
                _fv42 = json.load(f)
    return _fv42

# ── Beta V1 (RF_d3, AUC=0.6326, 35 features) ───────────────────────────────
def load_beta_v1():
    global _mbeta
    if _mbeta is None:
        if os.path.exists(MODEL_PATH_BETA):
            with open(MODEL_PATH_BETA, "rb") as f:
                _mbeta = pickle.load(f)
            print(f"🤖 Beta V1 loaded (RF_d3, AUC=0.6326, 35F)")
    return _mbeta

def load_scaler_beta_v1():
    global _sbeta
    if _sbeta is None:
        if os.path.exists(SCALER_PATH_BETA):
            with open(SCALER_PATH_BETA, "rb") as f:
                _sbeta = pickle.load(f)
            print(f"🤖 Beta V1 scaler loaded")
    return _sbeta

def load_scaler_v17():
    global _sv17
    if _sv17 is None:
        if os.path.exists(SCALER_PATH_V17):
            with open(SCALER_PATH_V17, "rb") as f:
                _sv17 = pickle.load(f)
    return _sv17

def load_scaler_v4():
    global _sv4
    if _sv4 is None and os.path.exists(SCALER_PATH_V4):
        with open(SCALER_PATH_V4, "rb") as f:
            _sv4 = pickle.load(f)
    return _sv4

def load_scaler_v5b():
    global _sv5b
    if _sv5b is None and os.path.exists(SCALER_PATH_V5B):
        with open(SCALER_PATH_V5B, "rb") as f:
            _sv5b = pickle.load(f)
    return _sv5b

def load_scaler_v6():
    global _sv6
    if _sv6 is None and os.path.exists(SCALER_PATH_V6):
        with open(SCALER_PATH_V6, "rb") as f:
            _sv6 = pickle.load(f)
    return _sv6

def load_scaler_v7():
    global _sv7
    if _sv7 is None and os.path.exists(SCALER_PATH_V7):
        with open(SCALER_PATH_V7, "rb") as f:
            _sv7 = pickle.load(f)
    return _sv7

# ════════════════════════════════════════════════════════════════════════════════
# [v39] 5m 乾淨特徵計算（無數據洩漏）
# ════════════════════════════════════════════════════════════════════════════════
def fetch_btc_klines_5m(limit=150):
    """Fetch Binance 5m klines — for v39 ML model."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol":"BTCUSDT","interval":"5m","limit":limit},
            timeout=8
        )
        data = r.json()
        # Returns list of [open_time, open, high, low, close, volume, close_time, ...]
        klines = {}
        for x in data:
            ts = int(x[0]) // 1000  # ms to sec, already 5m-aligned
            klines[ts] = {
                'open': float(x[1]), 'high': float(x[2]), 'low': float(x[3]),
                'close': float(x[4]), 'volume': float(x[5]),
                'quote_vol': float(x[7]), 'trades': int(x[8]),
                'taker_buy_vol': float(x[9])
            }
        times = sorted(klines.keys())
        return klines, times
    except Exception as e:
        print(f"⚠️ 5m klines error: {e}")
        return {}, []

def _ema_calc(prices, period):
    if len(prices) < period: return None
    k = 2.0 / (period + 1)
    ema = float(prices[0])
    for p in prices[1:]: ema = p * k + ema * (1 - k)
    return ema

def _rsi_calc(closes_arr, period=14):
    if len(closes_arr) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes_arr)):
        d = closes_arr[i] - closes_arr[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))

def _atr_calc(highs_arr, lows_arr, closes_arr, period=14):
    if len(highs_arr) < period + 1: return None
    trs = []
    for i in range(1, len(highs_arr)):
        h, l, cp = highs_arr[i], lows_arr[i], closes_arr[i-1]
        trs.append(max(h-l, abs(h-cp), abs(l-cp)))
    if len(trs) < period: return None
    return np.mean(trs[-period:])

def _stoch_calc(highs_arr, lows_arr, closes_arr, k_p=14, d_p=3):
    if len(highs_arr) < k_p: return None, None
    k_vals = []
    for i in range(k_p-1, len(closes_arr)):
        h = max(highs_arr[i-k_p+1:i+1])
        l = min(lows_arr[i-k_p+1:i+1])
        k_vals.append(100.0*(closes_arr[i]-l)/(h-l) if h!=l else 50.0)
    if len(k_vals) < d_p: return k_vals[-1], k_vals[-1]
    return k_vals[-1], np.mean(k_vals[-d_p:])

def _build_v39_features_legacy(klines, times):
    """
    [LEGACY - NOT USED] Old v39 feature builder with dict-style klines.
    Replaced by build_v39_features(klines_5m, slot_ts) at line ~789.
    Kept for reference only.
    times[-1] = last COMPLETED 5m bar (this is the "current" bar for prediction).
    closes[-1] = close of last completed bar = last KNOWN price.
    closes[-2] = close of 5min before = used for feature computation.
    """
    if len(times) < 30: return None
    
    # closes for features: up to closes[-2] (excludes the very last bar which would be "now")
    lookback_times = times[max(0, len(times)-150):len(times)-1]
    if len(lookback_times) < 20: return None
    
    closes  = [klines[t]['close'] for t in lookback_times]
    highs   = [klines[t]['high']  for t in lookback_times]
    lows    = [klines[t]['low']   for t in lookback_times]
    volumes = [klines[t]['volume'] for t in lookback_times]
    taker_b = [klines[t]['taker_buy_vol'] for t in lookback_times]
    quotes  = [klines[t]['quote_vol'] for t in lookback_times]
    
    # FIX: closes[-1] = close of last completed bar (lookback excludes times[-1])
    price = closes[-1]
    
    # EMA features
    ema5  = _ema_calc(closes, 5)
    ema9  = _ema_calc(closes, 9)
    ema12 = _ema_calc(closes, 12)
    ema20 = _ema_calc(closes, 20)
    ema50 = _ema_calc(closes, 50)
    
    ema_cross_raw   = (ema5 - ema20) / price if ema20 else 0.0
    ema_cross_9_20  = (ema9  - ema20) / price if ema20 else 0.0
    price_vs_ema5   = (price - ema5)  / price if ema5  else 0.0
    price_vs_ema20  = (price - ema20) / price if ema20 else 0.0
    price_vs_ema50  = (price - ema50) / price if ema50 else 0.0
    ema5_n  = ema5  / price if ema5  else 1.0
    ema20_n = ema20 / price if ema20 else 1.0
    
    # RSI
    rsi5_v   = _rsi_calc(closes, 5)
    rsi14_v  = _rsi_calc(closes, 14)
    rsi5_n   = rsi5_v  / 100.0 if rsi5_v  else 0.5
    rsi14_n  = rsi14_v / 100.0 if rsi14_v else 0.5
    rsi5_m14 = (rsi5_v - rsi14_v) / 100.0 if (rsi5_v and rsi14_v) else 0.0
    
    # MACD
    ema12_v = _ema_calc(closes, 12)
    ema26_v = _ema_calc(closes, 26)
    macd_n   = (ema12_v - ema26_v) / price if (ema12_v and ema26_v) else 0.0
    
    # Stochastic
    sk, sd = _stoch_calc(highs, lows, closes)
    stoch_k_n = sk / 100.0 if sk else 0.5
    stoch_d_n = sd / 100.0 if sd else 0.5
    
    # ATR
    at5_v  = _atr_calc(highs, lows, closes, 5)
    at14_v = _atr_calc(highs, lows, closes, 14)
    at5_n  = at5_v  / price if at5_v  else 0.0
    at14_n = at14_v / price if at14_v else 0.0
    
    # Bollinger position
    if len(closes) >= 20:
        bb_ma = np.mean(closes[-20:])
        bb_sd = np.std(closes[-20:], ddof=0)
        bb_h = bb_ma + 2*bb_sd; bb_l = bb_ma - 2*bb_sd
        bb_pos = (price - bb_l) / (bb_h - bb_l) if bb_h != bb_l else 0.5
    else:
        bb_pos = 0.5
    
    # Volume
    last_vol = klines[lookback_times[-1]]['volume']
    avg_vol5  = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
    avg_vol20 = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else volumes[-1]
    vol_r   = last_vol / avg_vol5   if avg_vol5  > 0 else 1.0
    vol_r20 = last_vol / avg_vol20  if avg_vol20 > 0 else 1.0
    
    last_tb = klines[lookback_times[-1]]['taker_buy_vol']
    last_v  = last_vol
    taker_r   = (last_tb / last_v) if last_v > 0 else 0.5
    avg_tb5 = np.mean(taker_b[-6:-1]) if len(taker_b) >= 6 else last_tb
    avg_v5  = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else last_v
    taker_r5 = (avg_tb5 / avg_v5) if avg_v5 > 0 else 0.5
    
    # Momentum (from closes[-1] = last completed close)
    mom1_v  = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0
    mom3_v  = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0.0
    mom5_v  = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0.0
    mom10_v = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0.0
    
    # Range
    rh = max(highs[-25:]); rl = min(lows[-25:])
    daily_pos = (price - rl) / (rh - rl) if rh > rl else 0.5
    
    # Volatility
    ret_std5  = np.std([(closes[i]-closes[i-1])/closes[i-1] for i in range(-5,-1)]) if len(closes)>=6 else 0.0
    ret_std10 = np.std([(closes[i]-closes[i-1])/closes[i-1] for i in range(-10,-1)]) if len(closes)>=11 else 0.0
    
    # Consecutive up closes
    consec_up = 0
    for i in range(len(closes)-1, max(0, len(closes)-12), -1):
        if closes[i] > closes[i-1]: consec_up += 1
        else: break
    
    # VWAP
    vwap20 = np.sum(quotes[-21:-1]) / np.sum(volumes[-21:-1]) if len(volumes)>=21 else price
    vwap_dev = (price - vwap20) / price if vwap20 else 0.0
    
    # Time features
    dt = datetime.fromtimestamp(times[-1], tz=timezone.utc)
    h_utc = dt.hour; dow = dt.weekday()
    hour_sin = np.sin(2*np.pi*h_utc/24)
    hour_cos = np.cos(2*np.pi*h_utc/24)
    is_asia = 1 if 1 <= h_utc <= 8 else 0
    is_us   = 1 if 13 <= h_utc <= 20 else 0
    is_wknd = 1 if dow >= 5 else 0
    
    # Patterns (last completed bar)
    lo = klines[lookback_times[-1]]
    lo_o, lo_h, lo_l, lo_c = lo['open'], lo['high'], lo['low'], lo['close']
    body_top = max(lo_c, lo_o); body_bot = min(lo_c, lo_o)
    body_sz = body_top - body_bot
    upper_sh = lo_h - body_top; lower_sh = body_bot - lo_l
    is_hammer = 1 if (lower_sh > 2*body_sz if body_sz > 0 else False) else 0
    is_shoot  = 1 if (upper_sh > 2*body_sz if body_sz > 0 else False) else 0
    prev_rng = highs[-2] - lows[-2] if len(highs) >= 2 else 0
    curr_rng = lo_h - lo_l
    inside = 1 if (curr_rng < prev_rng and prev_rng > 0) else 0
    
    # OB proxy
    obi_taker = (last_tb / last_v - 0.5) * 2 if last_v > 0 else 0.0
    
    # Trend
    trend = (ema5 - ema20) / (np.std(closes[-21:]) * 2) if (len(closes)>=21 and ema5 and ema20) else 0.0
    
    feat = {
        'atr14': at14_n, 'atr5': at5_n,
        'bb_position': bb_pos,
        'consec_up': consec_up / 10.0,
        'daily_range_pos': daily_pos,
        'day_of_week': dow / 7.0,
        'ema20_n': ema20_n, 'ema5_n': ema5_n,
        'ema_cross_9_20': max(min(ema_cross_9_20, 0.01), -0.01) / 0.01,
        'ema_cross_raw': max(min(ema_cross_raw, 0.01), -0.01) / 0.01,
        'hour_cos': hour_cos,
        'hour_sin': hour_sin,
        'hour_utc': h_utc / 24.0,
        'inside_bar': inside,
        'is_asia': is_asia,
        'is_hammer': is_hammer,
        'is_shooting_star': is_shoot,
        'is_us': is_us,
        'is_weekend': is_wknd,
        'macd': max(min(macd_n, 0.005), -0.005) / 0.005,
        'mom1': max(min(mom1_v, 0.005), -0.005) / 0.005,
        'mom10': max(min(mom10_v, 0.03), -0.03) / 0.03,
        'mom3': max(min(mom3_v, 0.01), -0.01) / 0.01,
        'mom5': max(min(mom5_v, 0.02), -0.02) / 0.02,
        'obi_taker': obi_taker,
        'price_vs_ema20': max(min(price_vs_ema20, 0.01), -0.01) / 0.01,
        'price_vs_ema5': max(min(price_vs_ema5, 0.005), -0.005) / 0.005,
        'price_vs_ema50': max(min(price_vs_ema50, 0.02), -0.02) / 0.02,
        'price_vs_rolling_high': max(min((price-rh)/price, 0.01), -0.01)/0.01,
        'price_vs_rolling_low': max(min((price-rl)/price, 0.01), -0.01)/0.01,
        'ret_std10': min(ret_std10, 0.01) / 0.01,
        'ret_std5': min(ret_std5, 0.005) / 0.005,
        'rsi14': rsi14_n,
        'rsi5': rsi5_n,
        'rsi5_minus_14': rsi5_m14,
        'stoch_d': stoch_d_n,
        'stoch_k': stoch_k_n,
        'taker_ratio': taker_r,
        'taker_ratio5': taker_r5,
        'trend_strength': max(min(trend, 3.0), -3.0) / 3.0,
        'vol_ratio': min(vol_r, 5.0) / 5.0,
        'vol_ratio20': min(vol_r20, 5.0) / 5.0,
        'vwap_deviation': max(min(vwap_dev, 0.005), -0.005) / 0.005,
    }
    return feat

# ════════════════════════════════════════════════════════════════════════════════
# FLAML v42 預測函數 (主動使用 v39 的 5m klines + 建構邏輯，直接輸出 UP/DOWN)
# ════════════════════════════════════════════════════════════════════════════════
def ml_predict_flaml_v42():
    """FLAML v42 PRIMARY 預測，使用 43 features (same as v39 pipeline).
    Returns: (p_down, p_up, direction, confidence, model_name)
    """
    try:
        # Re-use v39 feature pipeline (same 43 features, same 5m klines)
        klines_5m_dict, times_5m = fetch_btc_klines_5m(limit=150)
        if not klines_5m_dict or len(times_5m) < 30:
            return 0.5, 0.5, "WAIT", 0.5, "v42_no_data"
        klines_5m = [
            [ts, klines_5m_dict[ts]['open'], klines_5m_dict[ts]['high'],
             klines_5m_dict[ts]['low'],  klines_5m_dict[ts]['close'],
             klines_5m_dict[ts]['volume'], klines_5m_dict[ts]['quote_vol'],
             klines_5m_dict[ts]['taker_buy_vol']]
            for ts in times_5m
        ]
        slot_ts = times_5m[-1]
        feat = build_v39_features(klines_5m, slot_ts)
        if feat is None:
            return 0.5, 0.5, "WAIT", 0.5, "v42_feat_err"

        model   = load_v42()
        scaler  = load_scaler_v42()
        feat_names = load_features_v42()
        if model is None or scaler is None or feat_names is None:
            return 0.5, 0.5, "WAIT", 0.5, "v42_no_model"

        X = np.array([[feat.get(fn, 0.0) for fn in feat_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)
        p = model.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = float(max(p[0], p[1]))
        return float(p[0]), float(p[1]), d, c, "v42_flaml"
    except Exception as e:
        print(f"⚠️ ml_predict_flaml_v42 error: {e}")
        return 0.5, 0.5, "WAIT", 0.5, "v42_err"

def ml_predict_v39():
    """v39 prediction using 5m klines — no leakage. Returns 5-tuple always."""
    try:
        klines_5m_dict, times_5m = fetch_btc_klines_5m(limit=150)
        if not klines_5m_dict or len(times_5m) < 30:
            return 0.5, 0.5, "WAIT", 0.5, "v39_no_data"
        # Convert dict {ts: {...}} → list of raw Binance-style lists
        # [open_time, open, high, low, close, volume, quote_vol, taker_buy_vol]
        klines_5m = [
            [ts, klines_5m_dict[ts]['open'], klines_5m_dict[ts]['high'],
             klines_5m_dict[ts]['low'], klines_5m_dict[ts]['close'],
             klines_5m_dict[ts]['volume'], klines_5m_dict[ts]['quote_vol'],
             klines_5m_dict[ts]['taker_buy_vol']]
            for ts in times_5m
        ]
        # FIX: pass single slot_ts (last completed bar), not the full list
        slot_ts = times_5m[-1]
        feat = build_v39_features(klines_5m, slot_ts)
        if feat is None:
            print(f"⚠️ v39 build_features returned None for slot {slot_ts}")
            return 0.5, 0.5, "WAIT", 0.5, "v39_feat_err"
        model = load_v39(); scaler = load_scaler_v39()
        if model is None or scaler is None:
            return 0.5, 0.5, "WAIT", 0.5, "v39_no_model"
        feat_path = os.path.join(BASE, "data/ml_features_v39.json")
        with open(feat_path) as f:
            feat_names = json.load(f)
        X = np.array([[feat.get(fn, 0.0) for fn in feat_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)
        p = model.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = float(max(p[1], p[0]))
        return float(p[0]), float(p[1]), d, c, "v39"
    except Exception as e:
        print(f"⚠️ ml_predict_v39 error: {e}")
        return 0.5, 0.5, "WAIT", 0.5, "v39_feat_err"

# ════════════════════════════════════════════════════════════════════════════════
# ML 預測
# ════════════════════════════════════════════════════════════════════════════════
def ml_predict_v4(rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour):
    m = load_v4(); sc = load_scaler_v4()
    if m is None: return 0.5, 0.5
    X = np.array([[rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour]])
    if sc is not None: X = sc.transform(X)
    p = m.predict_proba(X)[0]
    return float(p[0]), float(p[1])  # p_down, p_up

def ml_predict_v5b(rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour):
    m = load_v5b(); sc = load_scaler_v5b()
    if m is None: return 0.5, 0.5
    X = np.array([[rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour]])
    if sc is not None: X = sc.transform(X)
    p = m.predict_proba(X)[0]
    return float(p[0]), float(p[1])

# ── Beta V1 (RF_d3, AUC=0.6326, 35 features, no hour bias) ──────────────────
def ml_predict_beta_v1(rsi, ls, stoch, willr, bb, atr, mom5, mom1, obi, fng, sr):
    """
    Beta V1: RandomForest depth=3, min_samples_leaf=30, 300 trees, 35 features.
    Holdout AUC=0.6326 | CV acc=51.9% | Holdout acc=57.7%
    Features: RSI_n, LS_n, Stoch_n, WillR_n, BB_n, ATR_n, Mom5_n, OBI_n +
              RSI/L/S/Momentum/Stoch/BB categorical signals.
    """
    try:
        model = load_beta_v1()
        scaler = load_scaler_beta_v1()
        if model is None or scaler is None:
            return 0.5, 0.5, "WAIT", 0.5, "beta_no_model"

        # Build 35 features (matching train_beta_v1.py build_X)
        x = np.array([[
            # Continuous (normalized)
            min(max(rsi/100, 0), 1),                          # 0: rsi_n
            min(max(ls/2.5, 0), 1),                           # 1: ls_n
            min(max(stoch/100, 0), 1),                        # 2: stoch_n
            min(max((willr+100)/100, 0), 1),                  # 3: willr_n (0=oversold,1=overbought)
            min(max(bb, 0), 1),                               # 4: bb_n
            min(max(atr/0.01, 0), 1),                        # 5: atr_n
            min(max(mom5/0.5, -1), 1),                       # 6: mom5_n
            min(max(obi+0.5, 0), 1),                         # 7: obi_n
            # RSI categorical
            1 if rsi < 30 else 0,                           # 8: rsi_os
            1 if rsi < 40 else 0,                           # 9: rsi_weak_os
            1 if 40 <= rsi <= 60 else 0,                    # 10: rsi_neutral
            1 if 60 < rsi <= 70 else 0,                     # 11: rsi_weak_ob
            1 if rsi > 70 else 0,                           # 12: rsi_ob
            1 if rsi < 20 else 0,                           # 13: rsi_extreme_os
            # L/S categorical
            1 if ls < 0.5 else 0,                          # 14: ls_crowded_short
            1 if 0.5 <= ls < 0.9 else 0,                    # 15: ls_moderate_short
            1 if 0.9 <= ls <= 1.1 else 0,                    # 16: ls_neutral
            1 if 1.1 <= ls < 1.3 else 0,                    # 17: ls_moderate_long
            1 if ls >= 1.3 else 0,                           # 18: ls_crowded_long
            # Momentum categorical
            1 if mom5 < -0.2 else 0,                         # 19: mom_strong_neg
            1 if mom5 < 0 else 0,                            # 20: mom_neg
            1 if mom5 > 0 else 0,                           # 21: mom_pos
            # Stochastic categorical
            1 if stoch < 20 else 0,                          # 22: stoch_os
            1 if stoch > 80 else 0,                          # 23: stoch_ob
            # WillR categorical
            1 if willr < -80 else 0,                        # 24: willr_os
            # BB categorical
            1 if bb < 0.2 else 0,                            # 25: bb_lower
            1 if bb > 0.8 else 0,                           # 26: bb_upper
            # Range/Vola
            1 if sr < 20 else 0,                            # 27: range_tight
            1 if sr > 60 else 0,                            # 28: range_wide
            # Composite signals
            (1 if rsi < 30 else 0) * (1 if 1.1 <= ls < 1.3 else 0),  # 29: sig_up_RSI_LS
            (1 if 40 <= rsi <= 60 else 0) * (1 if 0.5 <= ls < 0.9 else 0), # 30: sig_down_RSI_LS
            (1 if rsi < 30 else 0) * (1 if stoch < 20 else 0),             # 31: sig_up_RSI_STOCH
            (1 if stoch < 20 else 0) * (1 if mom5 < 0 else 0),              # 32: sig_up_STOCH_MOM
            (1 if sr < 20 else 0) * (1 if ls >= 1.0 else 0),                # 33: sig_down_RANGE
        ]], dtype=np.float32)

        X_raw = x
        X = scaler.transform(X_raw)
        p = model.predict_proba(X)[0]
        p_down = float(p[0])
        p_up   = float(p[1])
        d = "UP" if p_up >= p_down else "DOWN"
        c = max(p_up, p_down)
        return p_down, p_up, d, c, "beta_v1"

    except Exception as e:
        return 0.5, 0.5, "WAIT", 0.5, f"beta_err:{e}"

def ml_predict_v6_features(rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour,
                           stoch=0.0, bb=0.0, atr=0.1,
                           willr=0.0, rsi_f=0.0, mom1=0.0, mom15=0.0,
                           taker=1.0, taker_avg=1.0,
                           clob_obi=0.0, clob_ibi=0.0,
                           # Beta V1 raw inputs (optional — adds Beta V1 to cascade)
                           beta_ls=None, beta_fng=None, beta_sr=None):
    """
    [BetaV1+v42+v39+v35 Cascade]
      Beta V1 (RF_d3, AUC=0.6326, 35F, no hour bias) PRIMARY
        → v42_flaml (ExtraTrees, 43F) → v39 (LR) → v35 → v17 → v7 → v6
    Returns: (p_down, p_up, direction, confidence, model_name)
    """
    # ── [Beta V1] PRIMARY (RF_d3, AUC=0.6326, no hour bias) ───────────────────
    if beta_ls is not None and beta_fng is not None and beta_sr is not None:
        try:
            p_down, p_up, d, c, tag = ml_predict_beta_v1(
                rsi, beta_ls, stoch, willr, bb, atr,
                momentum_5m, mom1, clob_obi, beta_fng, beta_sr
            )
            if tag == "beta_v1" and d != "WAIT":
                print(f"[ML] p_down={p_down:.3f} p_up={p_up:.3f} dir={d} conf={c:.3f} model=Beta_V1")
                return p_down, p_up, d, c, "beta_v1"
            elif tag != "beta_no_model":
                print(f"[ML] Beta V1 fallback ({tag}): p_down={p_down:.3f} p_up={p_up:.3f}")
        except Exception:
            pass

    # ── [v42 FLAML] Fallback #1 (43 features, ExtraTrees, CV=53.5%) ─────────
    p42 = ml_predict_flaml_v42()
    if p42[2] not in ('WAIT', None) and p42[4] not in ('v42_no_model', 'v42_no_data', 'v42_feat_err', 'v42_err'):
        print(f"[ML] p_down={p42[0]:.3f} p_up={p42[1]:.3f} dir={p42[2]} conf={p42[3]:.3f} model=v42_flaml")
        return p42[0], p42[1], p42[2], p42[3], "v42_flaml"

    # ── [v39] Fallback #2 (43 features, LR C=0.001) ───────────────────────────
    p39 = ml_predict_v39()
    if p39[2] != 'WAIT' and p39[4] not in ('v39_no_model', 'v39_no_data', 'v39_feat_err'):
        print(f"[ML] p_down={p39[0]:.3f} p_up={p39[1]:.3f} dir={p39[2]} conf={p39[3]:.3f} model=v39_fallback")
        return p39[0], p39[1], p39[2], p39[3], "v39_fallback"

    # ── [v35] Fallback #2 (13 features from 1m klines) ───────────────────────
    mv35 = load_v35(); sc35 = load_scaler_v35()
    if mv35 is not None:
        X = np.array([[rsi, macd_h, ema_cross, vol_delta, stoch, bb, atr, willr,
                       mom1, mom15, taker, clob_obi, hour]])
        if sc35 is not None: X = sc35.transform(X)
        p = mv35.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(p[1], p[0])
        print(f"[ML] p_down={p[0]:.3f} p_up={p[1]:.3f} dir={d} conf={c:.3f} model=v35_fallback")
        return float(p[0]), float(p[1]), d, c, "v35_fallback"

    # Fallback v17 (15 features)
    m17 = load_v17(); sc17 = load_scaler_v17()
    if m17 is not None:
        X = np.array([[rsi, macd_h, ema_cross, vol_delta, momentum_5m,
                       mom1, stoch, bb, atr, willr, rsi_f,
                       taker, taker_avg, clob_obi, clob_ibi]])
        if sc17 is not None: X = sc17.transform(X)
        p = m17.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(p[1], p[0])
        return float(p[0]), float(p[1]), d, c, "v17"

    # Fallback v7 (9 features)
    m7 = load_v7(); sc7 = load_scaler_v7()
    if m7 is not None:
        X = np.array([[rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour,
                       stoch, bb, atr]])
        if sc7 is not None: X = sc7.transform(X)
        p = m7.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(p[1], p[0])
        return float(p[0]), float(p[1]), d, c, "v7"

    # Fallback v6 (6 features)
    m6 = load_v6(); sc6 = load_scaler_v6()
    if m6 is not None:
        X = np.array([[rsi, macd_h, ema_cross, vol_delta, momentum_5m, hour]])
        if sc6 is not None: X = sc6.transform(X)
        p = m6.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(p[1], p[0])
        return float(p[0]), float(p[1]), d, c, "v6"

    return 0.5, 0.5, "WAIT", 0.5, "none"

# ════════════════════════════════════════════════════════════════════════════════
# 指標計算
# ════════════════════════════════════════════════════════════════════════════════
def fetch_btc_klines(limit=100):
    """Fetch Binance 1m klines. Returns list of [open, high, low, close, volume]"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol":"BTCUSDT","interval":"1m","limit":limit},
            timeout=8
        )
        data = r.json()
        return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]
        # Binance format: [open_time, open, high, low, close, volume, ...]
    except Exception as e:
        print(f"⚠️ Binance error: {e}")
        return []

def rsi(closes, p=14):
    if len(closes) < p+1: return 50.0
    deltas = np.diff(closes[-p-1:])
    gain = sum(d for d in deltas if d > 0)
    loss = abs(sum(d for d in deltas if d < 0))
    if loss == 0: return 100.0
    rs = gain/loss
    return 100 - 100/(1+rs)

def ema(c, s):
    if len(c) < s: return c[-1] if c else 0
    k = 2/(s+1)
    out = [c[0]]
    for v in c[1:]: out.append(v*k + out[-1]*(1-k))
    return out[-1]

def macd_hist(closes):
    if len(closes) < 26: return 0.0
    e12 = ema(closes, 12); e26 = ema(closes, 26); e9 = ema(closes, 9)
    return float((e12 - e26) - e9)

def calc_vol_delta(klines):
    if len(klines) < 5: return 0.0
    total = 0.0
    for i in range(1, min(6, len(klines))):
        try:
            row = klines[-i]
            if len(row) < 5: return 0.0
            o, h, l, c, vol = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            total += (c - o) * vol
        except (ValueError, TypeError, IndexError):
            return 0.0
    return total / 5

def calc_stoch(closes, highs, lows, p=14):
    if len(closes) < p: return 50.0
    lo = min(lows[-p:]); hi = max(highs[-p:])
    if hi == lo: return 50.0
    return 100 * (closes[-1] - lo) / (hi - lo)

def calc_bb_pos(closes, p=20):
    if len(closes) < p: return 0.0
    win = closes[-p:]
    mu = sum(win)/len(win)
    sigma = (sum((x-mu)**2 for x in win)/len(win))**0.5
    if sigma == 0: return 0.0
    return (closes[-1] - (mu - 2*sigma)) / (4*sigma + 1e-9)

def fetch_taker_ratio(limit=3):
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol":"BTCUSDT","interval":"5m","limit":limit+1},
            timeout=5
        )
        data = r.json()
        buy_vol = sum(float(x[5]) for x in data[-limit:] if float(x[5]) > 0)
        sell_vol = sum(float(x[5]) for x in data[-limit:] if float(x[5]) < 0)
        ratio = abs(buy_vol/sell_vol) if sell_vol else 1.0
        return ratio
    except:
        return 1.0

def fetch_ls_ratio():
    """Binance Futures long/short ratio — returns (ratio, long_pct)"""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol":"BTCUSDT","period":"5m","limit":1},
            timeout=5
        )
        d = r.json()
        if d:
            ratio = float(d[0].get("longShortRatio", 1.0))
            long_pct = float(d[0].get("longAccount", 0.5))
            return ratio, long_pct
    except:
        pass
    return 1.0, 0.5

def fetch_fear_greed():
    """Alternative.me Fear & Greed Index — returns (value, classification)"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        d = r.json()['data'][0]
        return int(d['value']), d['value_classification']
    except:
        return 50, "Neutral"

def fetch_funding_rate():
    """Binance Futures funding rate — returns float"""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"}, timeout=5
        )
        return float(r.json().get("lastFundingRate", 0))
    except:
        return 0.0

# ════════════════════════════════════════════════════════════════════════════════
# v39 Feature Computation (NO LEAKAGE — same logic as build_ml_dataset_v3.py)
# ════════════════════════════════════════════════════════════════════════════════
def fetch_btc_5m_klines(limit=200):
    """Fetch Binance 5m klines. Returns list of [open_time_ms, open, high, low, close, volume, quote_vol, taker_buy_vol]"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit},
            timeout=8
        )
        data = r.json()
        # Returns: [open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_buy_base, taker_buy_quote, ignore]
        return [[int(x[0]), float(x[1]), float(x[2]), float(x[3]),
                  float(x[4]), float(x[5]), float(x[7]),
                  float(x[9])] for x in data]
    except Exception as e:
        print(f"⚠️ Binance 5m klines error: {e}")
        return []

def _ema(prices, period):
    if len(prices) < period: return None
    k = 2.0 / (period + 1)
    ema = float(prices[0])
    for p in prices[1:]: ema = p * k + ema * (1 - k)
    return ema

def _rsi(closes_arr, period=14):
    if len(closes_arr) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes_arr)):
        d = closes_arr[i] - closes_arr[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))

def _atr(highs_arr, lows_arr, closes_arr, period=14):
    if len(highs_arr) < period + 1: return None
    trs = []
    for i in range(1, len(highs_arr)):
        h, l, cp = highs_arr[i], lows_arr[i], closes_arr[i-1]
        trs.append(max(h-l, abs(h-cp), abs(l-cp)))
    if len(trs) < period: return None
    return np.mean(trs[-period:])

def _stoch(highs_arr, lows_arr, closes_arr, k_p=14, d_p=3):
    if len(closes_arr) < k_p: return None, None
    k_vals = []
    for i in range(k_p-1, len(closes_arr)):
        h = max(highs_arr[i-k_p+1:i+1])
        l = min(lows_arr[i-k_p+1:i+1])
        k_vals.append(100.0*(closes_arr[i]-l)/(h-l) if h!=l else 50.0)
    if len(k_vals) < d_p: return k_vals[-1], k_vals[-1]
    return k_vals[-1], np.mean(k_vals[-d_p:])

def _bb(prices_arr, period=20, n_std=2.0):
    if len(prices_arr) < period: return None, None, None
    recent = prices_arr[-period:]
    mu = np.mean(recent); sd = np.std(recent, ddof=0)
    return mu - n_std*sd, mu, mu + n_std*sd

def build_v39_features(klines_5m, slot_ts):
    """
    Compute the 43 v39 features using 5m klines with NO LEAKAGE.
    
    klines_5m: list of [open_time, open, high, low, close, volume, quote_vol, taker_buy_vol]
    slot_ts: prediction slot timestamp (seconds)
    
    IMPORTANT: All features are computed using klines that ENDED BEFORE slot_ts.
    The last bar used is the one indexed at the slot just BEFORE slot_ts.
    closes[-1] = last completed 5m bar's close (NOT the settlement price).
    
    Feature order matches ml_features_v39.json (sorted alphabetically):
    """
    if len(klines_5m) < 30:
        return None
    
    # Find the index of the last kline that ended before slot_ts
    # klines_5m[i]['open_time'] = start of 5m bar; bar ends at open_time + 300s
    # We want the bar where: open_time < slot_ts AND open_time + 300 <= slot_ts
    # i.e., the bar that fully ended before the prediction window
    idx = None
    for i, k in enumerate(klines_5m):
        bar_end = k[0] + 300  # open_time + 300s = bar close time
        if bar_end <= slot_ts:
            idx = i  # last bar that ended before slot_ts
    
    if idx is None or idx < 20:
        return None
    
    # Build lookback from idx-150 to idx (inclusive) — 150 bars ending at the last completed bar
    lookback = klines_5m[max(0, idx-150):idx+1]
    if len(lookback) < 20:
        return None
    
    closes  = [k[4] for k in lookback]  # close prices
    highs   = [k[2] for k in lookback]
    lows    = [k[3] for k in lookback]
    volumes = [k[5] for k in lookback]
    taker_b = [k[7] for k in lookback]
    quotes  = [k[6] for k in lookback]
    
    # closes[-1] = last KNOWN close = bar ending at idx (before prediction window)
    # closes[-2] = last FULLY COMPLETED bar (closes[:-1] = up to idx-1)
    price = closes[-1]  # price at slot_ts - 5min (last known)
    c_feats = closes[:-1]  # up to closes[-2] — NO settlement price leakage
    h_feats = highs[:-1]
    l_feats = lows[:-1]
    v_feats = volumes[:-1]
    tb_feats = taker_b[:-1]
    q_feats = quotes[:-1]
    
    if len(c_feats) < 10:
        return None
    
    # ── EMA features ──────────────────────────────────────────────────────────
    ema5  = _ema(c_feats, 5)
    ema9  = _ema(c_feats, 9)
    ema12 = _ema(c_feats, 12)
    ema20 = _ema(c_feats, 20)
    ema50 = _ema(c_feats, 50)
    
    ema_cross_raw   = (ema5 - ema20) / price if ema20 else 0.0
    ema_cross_9_20  = (ema9  - ema20) / price if ema20 else 0.0
    price_vs_ema5   = (price - ema5)  / price if ema5  else 0.0
    price_vs_ema20  = (price - ema20) / price if ema20 else 0.0
    price_vs_ema50  = (price - ema50) / price if ema50 else 0.0
    ema5_n  = ema5  / price if ema5  else 1.0
    ema20_n = ema20 / price if ema20 else 1.0
    
    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi5  = _rsi(c_feats, 5)
    rsi14 = _rsi(c_feats, 14)
    rsi5_n  = rsi5  / 100.0 if rsi5  else 0.5
    rsi14_n = rsi14 / 100.0 if rsi14 else 0.5
    rsi5_minus_14 = (rsi5 - rsi14) / 100.0 if (rsi5 and rsi14) else 0.0
    
    # ── MACD ───────────────────────────────────────────────────────────────────
    ema12_ = _ema(c_feats, 12)
    ema26_ = _ema(c_feats, 26)
    macd = (ema12_ - ema26_) / price if (ema12_ and ema26_) else 0.0
    
    # ── Stochastic ─────────────────────────────────────────────────────────────
    sk, sd = _stoch(h_feats, l_feats, c_feats)
    stoch_k_n = sk / 100.0 if sk else 0.5
    stoch_d_n = sd / 100.0 if sd else 0.5
    
    # ── ATR ────────────────────────────────────────────────────────────────────
    at5  = _atr(h_feats, l_feats, c_feats, 5)
    at14 = _atr(h_feats, l_feats, c_feats, 14)
    at5_n  = at5  / price if at5  else 0.0
    at14_n = at14 / price if at14 else 0.0
    
    # ── Bollinger ──────────────────────────────────────────────────────────────
    bb_l, bb_m, bb_h = _bb(c_feats)
    bb_pos = (price - bb_l) / (bb_h - bb_l) if (bb_h and bb_h != bb_l) else 0.5
    
    # ── Volume ──────────────────────────────────────────────────────────────────
    last_vol  = volumes[-1]
    avg_vol5  = np.mean(v_feats[-6:-1]) if len(v_feats) >= 6 else np.mean(v_feats)
    avg_vol20 = np.mean(v_feats[-21:-1]) if len(v_feats) >= 21 else np.mean(v_feats)
    vol_r   = last_vol / avg_vol5   if avg_vol5  > 0 else 1.0
    vol_r20 = last_vol / avg_vol20  if avg_vol20 > 0 else 1.0
    
    last_tb  = taker_b[-1]
    last_v   = volumes[-1]
    taker_r  = (last_tb / last_v) if last_v > 0 else 0.5
    avg_tb5  = np.mean(tb_feats[-6:-1]) if len(tb_feats) >= 6 else np.mean(tb_feats)
    avg_v5   = np.mean(v_feats[-6:-1])  if len(v_feats) >= 6 else np.mean(v_feats)
    taker_r5 = (avg_tb5 / avg_v5) if avg_v5 > 0 else 0.5
    
    # ── Momentum (from closes[-2] = last completed bar) ───────────────────────
    mom1  = (c_feats[-1] - c_feats[-2])  / c_feats[-2]  if len(c_feats) >= 2  else 0.0
    mom3  = (c_feats[-1] - c_feats[-4])  / c_feats[-4]  if len(c_feats) >= 4  else 0.0
    mom5  = (c_feats[-1] - c_feats[-6])  / c_feats[-6]  if len(c_feats) >= 6  else 0.0
    mom10 = (c_feats[-1] - c_feats[-11]) / c_feats[-11] if len(c_feats) >= 11 else 0.0
    
    # ── Range ───────────────────────────────────────────────────────────────────
    rh = max(h_feats[-25:])
    rl = min(l_feats[-25:])
    daily_pos = (price - rl) / (rh - rl) if rh > rl else 0.5
    
    # ── Volatility ──────────────────────────────────────────────────────────────
    ret_std5  = np.std([(c_feats[i]-c_feats[i-1])/c_feats[i-1] for i in range(-5,-1)])  if len(c_feats) >= 6 else 0.0
    ret_std10 = np.std([(c_feats[i]-c_feats[i-1])/c_feats[i-1] for i in range(-10,-1)]) if len(c_feats) >= 11 else 0.0
    
    # ── Consecutive up closes ───────────────────────────────────────────────────
    consec_up = 0
    for i in range(len(c_feats)-1, max(0, len(c_feats)-12), -1):
        if c_feats[i] > c_feats[i-1]: consec_up += 1
        else: break
    
    # ── VWAP ───────────────────────────────────────────────────────────────────
    vwap20 = np.sum(q_feats[-21:-1]) / np.sum(v_feats[-21:-1]) if len(v_feats) >= 21 else price
    vwap_dev = (price - vwap20) / price if vwap20 else 0.0
    
    # ── Time features ───────────────────────────────────────────────────────────
    dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    h_utc = dt.hour; dow = dt.weekday()
    hour_sin = np.sin(2*np.pi*h_utc/24)
    hour_cos = np.cos(2*np.pi*h_utc/24)
    is_asia  = 1 if 1 <= h_utc <= 8 else 0
    is_us    = 1 if 13 <= h_utc <= 20 else 0
    is_wknd  = 1 if dow >= 5 else 0
    
    # ── Patterns (last completed bar: klines_5m[idx-1]) ────────────────────────
    lo = klines_5m[idx - 1]  # the bar that ENDED at slot_ts (last completed bar)
    lo_open, lo_high, lo_low, lo_close = lo[1], lo[2], lo[3], lo[4]
    body_top = max(lo_close, lo_open)
    body_bot = min(lo_close, lo_open)
    body_sz  = body_top - body_bot
    upper_sh = lo_high - body_top
    lower_sh = body_bot - lo_low
    
    is_hammer = 1 if (lower_sh > 2*body_sz if body_sz > 0 else False) else 0
    is_shoot  = 1 if (upper_sh > 2*body_sz if body_sz > 0 else False) else 0
    prev_rng  = h_feats[-2] - l_feats[-2] if len(h_feats) >= 2 else 0
    curr_rng  = lo_high - lo_low
    inside    = 1 if (curr_rng < prev_rng and prev_rng > 0) else 0
    
    # ── OB proxy ────────────────────────────────────────────────────────────────
    obi_taker = (last_tb / last_v - 0.5) * 2 if last_v > 0 else 0.0
    
    # ── Trend ───────────────────────────────────────────────────────────────────
    trend = (ema5 - ema20) / (np.std(c_feats[-21:]) * 2) if (len(c_feats)>=21 and ema5 and ema20) else 0.0
    
    # ── Price vs rolling ────────────────────────────────────────────────────────
    price_vs_rh = (price - rh) / price
    price_vs_rl = (price - rl) / price
    
    # Build feature dict matching ml_features_v39.json order
    feat = {
        'atr14':           at14_n,
        'atr5':            at5_n,
        'bb_position':      bb_pos,
        'consec_up':       consec_up / 10.0,
        'daily_range_pos': daily_pos,
        'day_of_week':     dow / 7.0,
        'ema20_n':         ema20_n,
        'ema5_n':          ema5_n,
        'ema_cross_9_20':  max(min(ema_cross_9_20, 0.01), -0.01) / 0.01,
        'ema_cross_raw':   max(min(ema_cross_raw, 0.01), -0.01) / 0.01,
        'hour_cos':        hour_cos,
        'hour_sin':        hour_sin,
        'hour_utc':        h_utc / 24.0,
        'inside_bar':       inside,
        'is_asia':         is_asia,
        'is_hammer':        is_hammer,
        'is_shooting_star': is_shoot,
        'is_us':           is_us,
        'is_weekend':       is_wknd,
        'macd':            max(min(macd, 0.005), -0.005) / 0.005,
        'mom1':            max(min(mom1, 0.005), -0.005) / 0.005,
        'mom10':           max(min(mom10, 0.03), -0.03) / 0.03,
        'mom3':            max(min(mom3, 0.01), -0.01) / 0.01,
        'mom5':            max(min(mom5, 0.02), -0.02) / 0.02,
        'obi_taker':       obi_taker,
        'price_vs_ema20':  max(min(price_vs_ema20, 0.01), -0.01) / 0.01,
        'price_vs_ema5':   max(min(price_vs_ema5, 0.005), -0.005) / 0.005,
        'price_vs_ema50':  max(min(price_vs_ema50, 0.02), -0.02) / 0.02,
        'price_vs_rolling_high': max(min(price_vs_rh, 0.01), -0.01) / 0.01,
        'price_vs_rolling_low':  max(min(price_vs_rl, 0.01), -0.01) / 0.01,
        'ret_std10':       min(ret_std10, 0.01) / 0.01,
        'ret_std5':        min(ret_std5, 0.005) / 0.005,
        'rsi14':           rsi14_n,
        'rsi5':            rsi5_n,
        'rsi5_minus_14':   rsi5_minus_14,
        'stoch_d':         stoch_d_n,
        'stoch_k':         stoch_k_n,
        'taker_ratio':     taker_r,
        'taker_ratio5':   taker_r5,
        'trend_strength':  max(min(trend, 3.0), -3.0) / 3.0,
        'vol_ratio':       min(vol_r, 5.0) / 5.0,
        'vol_ratio20':    min(vol_r20, 5.0) / 5.0,
        'vwap_deviation':  max(min(vwap_dev, 0.005), -0.005) / 0.005,
    }
    
    return feat

# ════════════════════════════════════════════════════════════════════════════════
# CLOB 訂單簿
# ════════════════════════════════════════════════════════════════════════════════
def fetch_clob_obi(client, token_id):
    """Fetch order book imbalance. Returns (obi, fetch_timestamp)"""
    try:
        ob = client.get_order_book(token_id)
        ts = time.time()
        bids = [(float(b.price), float(b.size)) for b in (ob.bids or [])]
        asks = [(float(a.price), float(a.size)) for a in (ob.asks or [])]
        buy_up = sum(s for p,s in asks if 0.40 <= p <= 0.60)
        sell_up = sum(s for p,s in bids if 0.40 <= p <= 0.60)
        total = buy_up + sell_up + 1e-9
        obi = (buy_up - sell_up) / total
        return obi, ts
    except socket.timeout:
        print(f"⚠️ [v5B] CLOB orderbook timeout (15s) → OBI=0")
        return 0.0, 0.0
    except Exception as e:
        print(f"⚠️ [v5B] CLOB OBI error: {e}")
        return 0.0, 0.0

def clob_contrarian(obi):
    """
    CLOB 逆向邏輯：obi > 0 → 群眾偏向 UP → 市場走 DOWN
                       obi < 0 → 群眾偏向 DOWN → 市場走 UP
    Returns: (direction, confidence, strength)
    """
    strength = "none"
    if abs(obi) >= CLOB_STRONG_THRESH:
        strength = "strong"
        conf = min(0.90, 0.50 + abs(obi) * 1.4)
    elif abs(obi) >= CLOB_WEAK_THRESH:
        strength = "medium"
        conf = min(0.70, 0.50 + abs(obi) * 1.4)
    else:
        strength = "weak"
        conf = 0.50 + abs(obi) * 0.5

    if obi > 0:
        return "DOWN", conf, strength
    elif obi < 0:
        return "UP", conf, strength
    return "WEAK", 0.50, "none"

# ════════════════════════════════════════════════════════════════════════════════
# Polymarket Client
# ════════════════════════════════════════════════════════════════════════════════
def init_client():
    try:
        key = open(os.path.expanduser("~/.openclaw/eth_key.txt")).read().strip()
        with open(os.path.expanduser("~/.openclaw/fresh_creds.json")) as f:
            c = json.load(f)
        creds = ApiCreds(c["api_key"], c["api_secret"], c["api_passphrase"])
        return ClobClient(host=HOST, chain_id=CHAIN, key=key, creds=creds,
                          signature_type=2, funder=FUNDER)
    except Exception as e:
        print(f"⚠️ Client init error: {e}")
        return None

def get_balance(client):
    if client is None: return 0.0
    try:
        p = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
        b = client.get_balance_allowance(params=p)
        return int(b.get("balance", "0")) / 1e6
    except socket.timeout:
        print(f"⚠️ [v5B] Balance API timeout → return cached balance")
        return 0.0
    except Exception as e:
        print(f"⚠️ get_balance error: {e}")
        return 0.0

def find_market(last_traded_slot=0):
    """Find active BTC-5m market using Gamma API"""
    now = int(time.time())
    next_slot = (now // 300) * 300 + 300
    cur_slot  = (now // 300) * 300

    for slot in [next_slot, cur_slot]:
        try:
            slot_end = slot + 300
            secs_remaining = slot_end - now
            if secs_remaining < MIN_ENTRY_SECS:
                continue
            if slot == last_traded_slot:
                print(f"🔒 Slot {slot} already traded, skip")
                continue

            r = requests.get("https://gamma-api.polymarket.com/markets",
                             params={"slug": f"btc-updown-5m-{slot}"}, timeout=10)
            if r.status_code != 200 or not r.json():
                continue
            m = r.json()[0]
            if not m.get("acceptingOrders"):
                continue
            ids = json.loads(m.get("clobTokenIds", "[]"))
            if len(ids) < 2:
                continue
            prices = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
            up_px = float(prices[0])

            return {
                "slot": slot,
                "up_token": ids[0], "down_token": ids[1],
                "up_price": up_px,
                "min_size": m.get("orderMinSize", 5),
                "tick_size": str(m.get("orderPriceMinTickSize", "0.01")),
                "question": m.get("question", ""),
                "secs_remaining": secs_remaining,
            }
        except Exception as e:
            print(f"⚠️ find_market error: {e}")
    return None

# ════════════════════════════════════════════════════════════════════════════════
# 狀態管理
# ════════════════════════════════════════════════════════════════════════════════
def load_state():
    try:
        with open(NEWV1_STATE) as f:
            return json.load(f)
    except:
        return {
            "consec_loss": 0, "consec_wins": 0,
            "pause_until": 0, "day_start_bal": 0,
            "daily_stake_total": 0.0, "day": "",
            "last_slot": 0,
        }

def save_state(s):
    os.makedirs(os.path.dirname(NEWV1_STATE) or ".", exist_ok=True)
    with open(NEWV1_STATE, "w") as f:
        json.dump(s, f, indent=2)

def recount_from_log(reset_ts=None):
    """Count consecutive losses from trade log, optionally only after reset_ts (ISO string)"""
    try:
        trades = []
        with open(LOG) as f:
            for line in f:
                try: trades.append(json.loads(line.strip()))
                except: pass
        # [FIX] 只計算 reset_ts 之後的交易，避免舊的連虧一直觸發
        if reset_ts:
            trades = [t for t in trades if str(t.get("ts","")) >= reset_ts]
        recent = [t for t in trades if t.get("status") == "SUCCESS"][-20:]
        loss_streak = 0
        for t in reversed(recent):
            a = t.get("actual")
            if a == "WIN":
                break
            elif a == "LOSS":
                loss_streak += 1
            else:
                continue
        return loss_streak
    except:
        return 0

# ════════════════════════════════════════════════════════════════════════════════
# 日誌
# ════════════════════════════════════════════════════════════════════════════════
def log_trade(direction, conf, status, amount, order_id="", tx_hash="",
              market_name="", odds=0, slot=0,
              indicators=None, ml_signal=None, clob_signal=None,
              bet_pct=0, rsi_val=50, obi=0,
              balance=0, day_pnl=0, note="", token_name="", version=""):
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(),
        "direction": direction,
        "confidence": round(conf, 4) if conf else 0,
        "status": status,
        "amount": round(amount, 4),
        "order_id": order_id,
        "tx_hash": tx_hash,
        "market_name": market_name,
        "odds": round(odds, 4) if odds else 0,
        "slot": slot,
        "token_name": token_name,
        "version": "v5B (RSI Fixed)",
        "indicators": indicators or {},
        "ml_signal": ml_signal or {},
        "clob_signal": clob_signal or {},
        "bet_pct": bet_pct,
        "rsi": rsi_val,
        "obi": round(obi, 4),
        "balance": round(balance, 2),
        "day_pnl_pct": round(day_pnl, 4),
        "note": note,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ════════════════════════════════════════════════════════════════════════════════
# [New V1.4] 結算檢查 — 自動更新所有交易的 WIN/LOSS
# ════════════════════════════════════════════════════════════════════════════════
def check_settlements():
    """檢查所有未結算的 SUCCESS 交易，透過 Gamma API 更新 actual=WIN/LOSS"""
    try:
        entries = []
        if os.path.exists(LOG):
            with open(LOG) as f:
                for line in f:
                    try: entries.append(json.loads(line.strip()))
                    except: pass
        if not entries:
            return 0

        # Find unresolved SUCCESS entries (no 'actual' field)
        unresolved = [e for e in entries if e.get("status") == "SUCCESS" and not e.get("actual")]
        if not unresolved:
            return 0

        print(f"🔍 Checking {len(unresolved)} unresolved trades for settlement...")

        updated = 0
        # Get unique slots to check
        slots_to_check = sorted(set(e.get("slot", 0) for e in unresolved if e.get("slot")))
        checked_markets = {}  # slot -> market_data

        for slot in slots_to_check:
            try:
                r = requests.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"slug": f"btc-updown-5m-{slot}"},
                    timeout=8
                )
                if r.status_code == 200 and r.json():
                    m = r.json()[0]
                    closed = m.get("closed", False)
                    resolved = m.get("resolved", False)
                    if closed or resolved:
                        prices = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
                        checked_markets[slot] = {
                            "closed": m.get("closed", False),
                            "outcomePrices": prices,
                        }
            except:
                pass

        new_entries = []
        for e in entries:
            if e.get("status") == "SUCCESS" and not e.get("actual"):
                slot = e.get("slot", 0)
                market = checked_markets.get(slot)

                if market and market.get("closed"):
                    prices = market.get("outcomePrices", [])
                    # outcomePrices[0]='1' → UP won; [1]='1' → DOWN won
                    actual = ""
                    if len(prices) >= 2:
                        up_won = prices[0] == '1'
                        trade_dir = e.get("direction", "")
                        actual = "WIN" if (trade_dir == "UP" and up_won) or (trade_dir == "DOWN" and not up_won) else "LOSS"
                        if actual:
                            e["actual"] = actual
                            marker = "✅" if actual == "WIN" else "❌"
                            print(f"  {marker} Slot {slot} {trade_dir} → {actual}")
                            updated += 1
            new_entries.append(e)

        if updated > 0:
            with open(LOG, "w") as f:
                for e in new_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"✅ Updated {updated} settlement records")

        return updated
    except Exception as e:
        print(f"⚠️ check_settlements error: {e}")
        return 0

# ════════════════════════════════════════════════════════════════════════════════
# [New V1.4] 核心交易邏輯
# ════════════════════════════════════════════════════════════════════════════════
def run_trade():
    mode_tag = "🔍 PREDICT ONLY" if PREDICT_ONLY else "💰 LIVE TRADING"
    print(f"\n{'='*60}")
    print(f"  🦁 BTC-5m v5B — Beta V1 ML + RSI Rules + UTC04 Alpha")
    print(f"  [{mode_tag}]")
    print(f"{'='*60}")

    client = init_client()
    if client is None:
        print("❌ Client init failed"); return

    # ── [New V1.4] 結算檢查 — 更新 WIN/LOSS ──────────────────────────────
    check_settlements()

    balance = get_balance(client)
    print(f"💰 Balance: ${balance:.2f}")

    if balance < 5:
        print("⚠️ Balance < $5, stop"); return

    state = load_state()
    now = time.time()

    # ── Day change: reset daily stake ──
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("day", "") != today:
        state["day"] = today
        state["daily_stake_total"] = 0.0
        state["day_start_bal"] = balance
        print(f"📅 New day {today}, reset daily tracking")

    day_pnl_pct = ((balance - state["day_start_bal"]) / state["day_start_bal"]) if state["day_start_bal"] > 0 else 0

    # ── Hard stops ──
    # 1. Daily loss hard stop
    if state["day_start_bal"] > 0 and day_pnl_pct <= -HARD_STOP_PCT:
        print(f"🚨 [HARD STOP] 日虧 {-day_pnl_pct:.1%} ≥ {HARD_STOP_PCT:.0%}，今日停止");
        log_trade("WAIT", 0, "HARD_STOP_DAY", 0, note=f"日虧{-day_pnl_pct:.1%}")
        return

    # 2. Consecutive loss pause
    real_consec_loss = recount_from_log(reset_ts=state.get("reset_ts"))
    if real_consec_loss != state.get("consec_loss", 0):
        print(f"🔄 連虧更新: {state.get('consec_loss',0)} → {real_consec_loss}")
        state["consec_loss"] = real_consec_loss

    # [New V1.5] 暫停時改為 SIM 模式：繼續預測 + 記錄，但不做真實交易
    # [v5B] 支援環境變數 SIM_MODE=1 強制開啟 SIM（方便 cron 調試）
    SIM_MODE = os.environ.get("SIM_MODE", "") == "1"
    # 也支援 state['sim_mode'] 手動開啟 SIM（用戶主動要求觀望模式）
    if not SIM_MODE and state.get("sim_mode", False) and state.get("pause_until", 0) > now:
        remain = (state["pause_until"] - now) / 60
        print(f"⏸️ SIM模式（手動）中，剩{remain:.0f}分鐘 → 僅預測，不下單")
        SIM_MODE = True
    elif state["consec_loss"] >= MAX_CONSEC_LOSS:
        # [FIX v5A-SIM] 若 pause_until==0，代表已手動重置，不重新啟動暫停
        if state.get("pause_until", 0) == 0:
            state["pause_until"] = now + PAUSE_HOURS * 3600
            save_state(state)
            print(f"🚨 [PAUSE] 連虧 {state['consec_loss']}次，暫停{PAUSE_HOURS}小時")
        if now < state["pause_until"]:
            remain = (state["pause_until"] - now) / 60
            print(f"⏸️  SIM模式中，剩{remain:.0f}分鐘 → 預測追蹤")
            SIM_MODE = True
        else:
            state["pause_until"] = 0
            state["consec_loss"] = 0
            state["sim_mode"] = False
            state["reset_ts"] = datetime.now().isoformat()
            save_state(state)
    # [FIX v5A-SIM] 手動重置後（pause_until==0），即使 consec_loss>=3 也直接跑真實交易
    if state.get("pause_until", 0) == 0:
        SIM_MODE = False

    # [FIX v5A-SIM] 方向翻轉自動解鎖：追蹤 SIM 期間方向，方向翻轉時解除 SIM
    # (moved to after final_dir is computed — see below)

    # 3. [v1.2] 移除每日下注上限（已停用）
    # MAX_DAILY_STAKE = None means no daily cap

    # ── Market data ──
    market = find_market(last_traded_slot=state.get("last_slot", 0))
    if not market:
        print("❌ No active BTC-5m market"); return

    slot = market.get("slot", 0)
    if slot == state.get("last_slot", 0):
        print("⏭️ Same slot as last trade, skip (PREDICT_ONLY 模式)")
        log_trade("WAIT", 0, "SAME_SLOT", 0, note="同slot跳過")
        return

    up_price = float(market.get("up_price", 0.5))
    # [FIX v5A] slot = start of 5-min window. secs until market closes = slot + 300 - now
    remaining_secs = max(0, (slot + 300) - time.time())
    if remaining_secs < MIN_ENTRY_SECS:
        print(f"⏭️ 距離收市僅{remaining_secs:.0f}秒 < {MIN_ENTRY_SECS}秒，跳過")
        return

    print(f"📋 Market: {market.get('question','')[:50]}")
    print(f"   UP={up_price:.3f} | 剩餘{secs_fmt(remaining_secs)}")

    # ── Fetch BTC data ──
    klines = fetch_btc_klines(limit=100)
    if len(klines) < 34:
        print("❌ Klines不足"); return

    closes = [k[3] for k in klines]   # close prices
    highs  = [k[1] for k in klines]
    lows   = [k[2] for k in klines]
    btc_price = closes[-1]

    # Indicators
    rsi_val  = rsi(closes)
    macd_val  = macd_hist(closes)
    e12 = ema(closes, 12); e26 = ema(closes, 26)
    ema_cross_val = (e12 - e26) / e26 * 100 if e26 != 0 else 0
    mom_5m = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
    mom_1m = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
    mom_15m = (closes[-1] - closes[-15]) / closes[-15] * 100 if len(closes) >= 15 else mom_5m
    vol_delta = calc_vol_delta(klines[-6:])
    stoch = calc_stoch(closes, highs, lows)
    stoch_n = (stoch - 50) / 50
    bb_pos = calc_bb_pos(closes)
    atr_val = _atr(highs, lows, closes, 14)
    atr_val = atr_val / btc_price if atr_val else 0.0
    # ── [Beta V1] slot_range (high-low of last 5 candles, in $) ────────────
    recent_high = max(highs[-5:])
    recent_low = min(lows[-5:])
    slot_range = recent_high - recent_low   # raw $ range for Beta V1 sr feature
    highest14 = max(highs[-14:]) if len(highs) >= 14 else max(highs)
    lowest14 = min(lows[-14:]) if len(lows) >= 14 else min(lows)
    willr_val = -100.0 * (highest14 - closes[-1]) / (highest14 - lowest14) if highest14 != lowest14 else -50.0
    taker = fetch_taker_ratio(3)
    hour_utc = datetime.now(timezone.utc).hour

    fng_val, fng_cls = fetch_fear_greed()
    ls_ratio, long_pct = fetch_ls_ratio()

    print(f"📊 BTC: ${btc_price:,.0f} | RSI:{rsi_val:.1f} | MACD:{macd_val:.2f} | Stoch:{stoch:.0f}")
    print(f"   EMA-cross: {ema_cross_val:+.3f}% | VolDelta:{vol_delta:+.2f} | Taker:{taker:.3f}")

    # ── [v5B-RSI] 修正後的 RSI 方向邏輯（2026-04-12）──────────────────────────
    # 市場數據根源（Binance 300根5m K線 + 280筆結算數據）：
    #
    #   RSI < 30  → 市場實際 DOWN 60-100%  🔴 應做空
    #   RSI 30-40 → 市場實際 DOWN 72%     🔴 應順勢做空
    #   RSI 40-60 → 市場中性（52% DOWN / 48% UP）⚠️ 觀望
    #   RSI 60-70 → 市場實際 UP 67%       🟢 應做多
    #   RSI > 70  → 市場實際 UP 71-100%   🟢 應做多
    #
    # 核心問題：舊 v5A 的「RSI<30→UP」完全顛倒
    #           ML 在 RSI>60 預測 DOWN 也是錯誤（市場67%往上）
    #
    # 修復后的矩陣：
    #   RSI < 25  → UP 信號（小倉位逆勢抄底）
    #   RSI 25-35 → DOWN_PREF（市場慣性順勢，倉位正常）
    #   RSI 35-65 → 全域 WAIT（中性區，RSI 失效）
    #   RSI 65-75 → UP_PREF（市場動能向上）
    #   RSI > 75  → DOWN 信號（極端超買反轉，小倉位）
    # ─────────────────────────────────────────────────────────────────────────────
    if rsi_val < 25:
        rsi_rule_dir = "UP_ONLY"
        rsi_tag = "超賣"
        print(f"✅ [v5B-RSI] RSI={rsi_val:.0f}<25 → UP 逆勢抄底")
    elif rsi_val < 40:
        rsi_rule_dir = "DOWN_PREF"
        print(f"🔴 [v5B-RSI] RSI={rsi_val:.0f}<40 → DOWN 市場慣性偏空")
    elif rsi_val <= 60:
        rsi_rule_dir = "WAIT"
        print(f"⏸️  [v5B-RSI] RSI={rsi_val:.0f} 中性區（25-60）→ 全域 WAIT")
        log_trade("WAIT", 0, "RSI_NEUTRAL", 0,
                   market_name=market.get("question",""),
                   indicators={"rsi":round(rsi_val,1),"macd":round(macd_val,3)},
                   balance=balance, day_pnl=day_pnl_pct,
                   note=f"[v5B-RSI] RSI={rsi_val:.0f} 中性區（25-60）")
        return
    elif rsi_val < 75:
        rsi_rule_dir = "UP_PREF"
        rsi_tag = "偏強"
        print(f"🟢 [v5B-RSI] RSI={rsi_val:.0f}>60 → UP 市場動能向上")
    else:
        rsi_rule_dir = "DOWN_ONLY"
        rsi_tag = "超買"
        print(f"⚠️  [v5B-RSI] RSI={rsi_val:.0f}>75 → DOWN 極端超買反轉")

    # ── ML Signal: Beta V1 PRIMARY → v42_flaml → v39 → v35 → v17 → v7 → v6 ──
    p_down, p_up, ml_dir, ml_conf, model_used = ml_predict_v6_features(
        rsi_val, macd_val, ema_cross_val, vol_delta, mom_5m, hour_utc,
        stoch=stoch, bb=bb_pos, atr=0.1, willr=willr_val,
        mom1=mom_1m, mom15=mom_15m, taker=taker, clob_obi=0.0,
        # Beta V1 raw inputs (no hour bias, AUC=0.6326)
        beta_ls=ls_ratio, beta_fng=fng_val, beta_sr=slot_range
    )
    print(f"🤖 ML: {ml_dir} @ {ml_conf:.1%} ({model_used})")

    # Also get v4 and v5b for agreement check
    pd_v4, pu_v4 = ml_predict_v4(rsi_val, macd_val, ema_cross_val, vol_delta, mom_5m, hour_utc)
    pd_v5b, pu_v5b = ml_predict_v5b(rsi_val, macd_val, ema_cross_val, vol_delta, mom_5m, hour_utc)
    d_v4 = "UP" if pu_v4 >= pd_v4 else "DOWN"
    d_v5b = "UP" if pu_v5b >= pd_v5b else "DOWN"
    c_v4 = max(pu_v4, pd_v4); c_v5b = max(pu_v5b, pd_v5b)
    print(f"   v4={d_v4}({c_v4:.0%}) v5b={d_v5b}({c_v5b:.0%}) model={model_used}")

    # ── [v35 High-Confidence Override] ──────────────────────────────────────
    # ⚠️ DISABLED 2026-04-02：v35 conf>55% 直接信任 → 42% WR，虧損加速器
    directions = [ml_dir, d_v4, d_v5b]
    # ── Model agreement check ──
    unique_dirs = set(directions)
    if len(unique_dirs) == 3:
        # All 3 different → skip
        print(f"🚫 [New V1.4] 三模型分歧 ({ml_dir}/{d_v4}/{d_v5b})，觀望")
        log_trade("WAIT", 0, "MODEL_DISAGREEMENT", 0,
                   market_name=market.get("question",""),
                   indicators={"rsi":rsi_val,"macd":macd_val,"ema_cross":ema_cross_val},
                   ml_signal={"v35":ml_dir,"v4":d_v4,"v5b":d_v5b,"model":model_used},
                   balance=balance, day_pnl=day_pnl_pct,
                   note="三方向分歧")
        return

    # Majority direction (2/3 agree)
    up_count = directions.count("UP")
    down_count = directions.count("DOWN")
    primary_dir = "UP" if up_count >= 2 else "DOWN"
    agreement = max(up_count, down_count)  # 2 or 3

    print(f"   Agreement: {agreement}/3 → {primary_dir}")

    # ── [v5B-RSI] RSI 方向強制執行 ───────────────────────────────────────
    # 市場數據說：
    #   RSI<35 → 市場偏向 DOWN（慣性+均值回歸）
    #   RSI>65 → 市場偏向 UP（動能+趨勢）
    # 如果 ML 方向和 RSI 市場傾向矛盾，以 RSI 為準
    # 35-65 → 已經在上面 return 了
    # ─────────────────────────────────────────────────────────────────────────────
    rsi_ok = True
    if rsi_rule_dir == "DOWN_PREF" and primary_dir == "UP":
        # RSI<35 但 ML 說 UP → 忽略 ML，順 RSI 做 DOWN
        print(f"🚨 [v5B-RSI] RSI={rsi_val:.0f}<40 → 無視ML={primary_dir}，強制順勢做空")
        primary_dir = "DOWN"
    elif rsi_rule_dir == "UP_PREF" and primary_dir == "DOWN":
        # RSI>65 但 ML 說 DOWN → 忽略 ML，順 RSI 做 UP
        print(f"🚨 [v5B-RSI] RSI={rsi_val:.0f}>60 → 無視ML={primary_dir}，強制順勢做多")
        primary_dir = "UP"
    elif rsi_rule_dir == "UP_ONLY":
        if primary_dir != "UP":
            print(f"⏸️  [v5B-RSI] RSI={rsi_val:.0f}<25 → ML說{primary_dir}，RSI強制UP")
            primary_dir = "UP"
    elif rsi_rule_dir == "DOWN_ONLY":
        if primary_dir != "DOWN":
            print(f"⏸️  [v5B-RSI] RSI={rsi_val:.0f}>75 → ML說{primary_dir}，RSI極端超買反轉做空")
            primary_dir = "DOWN"

    # ── CLOB Contrarian Signal ──
    try:
        up_token = market.get("up_token", "")
        if up_token:
            obi, clob_ts = fetch_clob_obi(client, up_token)
            if clob_ts > 0 and (time.time() - clob_ts) > 30:
                print(f"   ⚠️ CLOB數據過期 ({(time.time()-clob_ts):.0f}s)，忽略")
                obi = 0.0
            else:
                print(f"   📊 CLOB OBI: {obi:+.3f} (age={(time.time()-clob_ts):.0f}s)")
        else:
            obi = 0.0
    except Exception as e:
        print(f"   ⚠️ CLOB fetch error: {e}")
        obi = 0.0

    clob_dir, clob_conf, clob_str = clob_contrarian(obi)
    print(f"   📊 CLOB: {clob_dir} @ {clob_conf:.1%} [{clob_str}]")

    # ── [New V1.5 FIX] UTC 07/11 ML系統性偏差修復 ──
    # 數據根源：
    #   UTC 07: ML 100% UP預測，但準確率只有36%；CLOB 73%準確率
    #   UTC 11: ML 90% UP預測，但準確率只有43%；市場實際52% UP（接近硬幣）
    # 問題：ML在這兩個時段有嚴重UP偏見，override邏輯讓錯誤信號通過
    # 修復：UTC 07/11時，CLOB冲突立大於ML，不等"strong"等級
    HOUR_ML_BAD = {7, 11}   # ML在這兩個小時系統性偏差

    final_dir = primary_dir
    clob_flip = False

    if clob_dir != primary_dir and hour_utc in HOUR_ML_BAD:
        # [FIX] UTC 07/11：任何CLOB冲突都跟隨CLOB
        flip_tag = "🚨 [NewV1.5 FIX] UTC07/11 CLOBOverride"
        print(f"{flip_tag}: {primary_dir} → {clob_dir} (ML={ml_dir} 系統偏差，CLOB準確率73%/52%)")
        final_dir = clob_dir
        clob_flip = True
    elif clob_str == "strong" and clob_dir != primary_dir:
        # Strong CLOB conflict → flip (其餘時段保持原有邏輯)
        print(f"🚨 [New V1.4] CLOB強衝突翻轉: {primary_dir} → {clob_dir}")
        final_dir = clob_dir
        clob_flip = True
    elif clob_str == "medium" and clob_dir != primary_dir:
        print(f"⚠️ [New V1.4] CLOB中等衝突，方向維持{primary_dir}，倉位減半")
        # Direction stays, bet will be halved by clob_weak flag
    else:
        print(f"✅ [New V1.4] CLOB確認或無信號，方向={primary_dir}")

    # ── [v33] 時段阻擋檢查 ──
    # [v39] Blocked 時段也跑 SIM（預測日誌追蹤，不下真單）
    BLOCKED_HOUR_SIM = False
    if hour_utc in HOUR_BLOCK:
        print(f"⛔ UTC {hour_utc:02d} BLOCKED (WR<40%) → 跑 SIM 模式")
        BLOCKED_HOUR_SIM = True
        bet_pct = BASE_BET_PCT  # 計算 bet_pct 供日誌使用
        bet_usdc = balance * bet_pct
        bet_usdc = max(MIN_BET_USDC, min(bet_usdc, MAX_BET_USDC))

    # ── [v33] OBI 過濾：避開 +0.1~+1.0 區間（WR 44.4%）──
    obi_ok = True
    obi_blocked = False
    if abs(obi) > 0:
        clob_age = time.time() - clob_ts
        if OBI_BAD_RANGE[0] < obi < OBI_BAD_RANGE[1] and clob_age < 60:
            # OBI 在惡劣區間，減半倉位
            print(f"⚠️ [v33] OBI {obi:+.2f} 在 bad range {OBI_BAD_RANGE}，bet × 0.5")
            obi_ok = False
            obi_blocked = True

    # ── [v39] 新信號：Fear & Greed + L/S Ratio + Funding Rate ──
    fng_val, fng_cls = fetch_fear_greed()
    ls_ratio, long_pct = fetch_ls_ratio()
    funding = fetch_funding_rate()
    print(f"   📡 F&G={fng_val}({fng_cls}) | L/S={ls_ratio:.2f}(多={long_pct:.0%}) | FR={funding*100:+.4f}%")

    # F&G 方向偏置（極端恐慌 <20 偏 UP，極端貪婪 >80 偏 DOWN）
    fng_bias = None
    if fng_val < 20:
        fng_bias = "UP"
        print(f"   🔴 F&G={fng_val} 極端恐慌 → 偏向 UP")
    elif fng_val > 80:
        fng_bias = "DOWN"
        print(f"   🟢 F&G={fng_val} 極端貪婪 → 偏向 DOWN")

    # L/S Ratio 方向偏置（多方>70% = 過熱 → 偏 DOWN；空方>70% = 過空 → 偏 UP）
    ls_bias = None
    if long_pct > 0.70:
        ls_bias = "DOWN"
        print(f"   ⚠️ 多方={long_pct:.0%} 過擠 → 偏向 DOWN（逆勢）")
    elif long_pct < 0.35:
        ls_bias = "UP"
        print(f"   ⚠️ 空方={1-long_pct:.0%} 過擠 → 偏向 UP（逆勢）")

    # 新信號強化/減弱倉位
    signal_mult = 1.0
    # F&G 與方向一致 → +20%；相反 → -30%
    if fng_bias == final_dir:
        signal_mult *= 1.20
        print(f"   ✅ F&G確認方向 → bet × 1.20")
    elif fng_bias and fng_bias != final_dir:
        signal_mult *= 0.70
        print(f"   ⚠️ F&G逆向 → bet × 0.70")
    # L/S 確認 → +10%；相反 → -20%
    if ls_bias == final_dir:
        signal_mult *= 1.10
        print(f"   ✅ L/S確認方向 → bet × 1.10")
    elif ls_bias and ls_bias != final_dir:
        signal_mult *= 0.80
        print(f"   ⚠️ L/S逆向 → bet × 0.80")

    # ── [FIX v5A-SIM] 方向翻轉自動解鎖（需在 final_dir 確定後）──
    sim_dir_history = state.get("sim_dir_history", [])
    prev_sim_dir = sim_dir_history[-1] if sim_dir_history else None
    if SIM_MODE and prev_sim_dir and prev_sim_dir != final_dir:
        # 方向翻轉了 → 可以解除
        print(f"🔄 [v5A-SIM] 方向翻轉 {prev_sim_dir}→{final_dir}，解除SIM")
        SIM_MODE = False
        state["pause_until"] = 0
        state["consec_loss"] = 0
        state["sim_dir_history"] = []
        state["sim_mode"] = False
        state["reset_ts"] = datetime.now().isoformat()
        save_state(state)

    # 記錄本次方向（最多保留5筆）
    if SIM_MODE:
        sim_dir_history = (sim_dir_history or [])[-4:]
        sim_dir_history.append(final_dir)
        state["sim_dir_history"] = sim_dir_history

    # ── [New V1.4] Position Sizing ──
    hour_mult = HOUR_BET_MULT.get(hour_utc, 0.5)
    rsi_mult = rsi_bet_mult(rsi_val)

    # Base bet
    bet_pct = BASE_BET_PCT

    # Reduce if model agreement is only 2/3
    if agreement == 2:
        bet_pct *= 0.75
        print(f"   📉 2/3模型共識，bet × 0.75")

    # CLOB medium conflict → halve bet
    if clob_str == "medium" and not clob_flip:
        bet_pct *= 0.50
        print(f"   📉 CLOB中等衝突，bet × 0.50")

    # [v33] OBI bad range → halve bet
    if not obi_ok:
        bet_pct *= 0.50
        print(f"   📉 [v33] OBI bad range，bet × 0.50")

    # Apply hour multiplier
    bet_pct *= hour_mult
    print(f"   📊 Hour × {hour_mult:.2f} (UTC {hour_utc:02d})")

    # [v39] UTC 04 額外加倉 1.5x（歷史 74% WR #1 Alpha）
    if hour_utc == 4:
        bet_pct *= 1.50
        print(f"   🚀 UTC 04 Alpha × 1.50 (歷史WR=74%)")

    # Apply RSI multiplier
    bet_pct *= rsi_mult
    print(f"   📊 RSI × {rsi_mult:.2f} (RSI={rsi_val:.0f})")

    # [v39] Apply signal multiplier (F&G + L/S)
    bet_pct *= signal_mult
    if signal_mult != 1.0:
        print(f"   📡 Signal × {signal_mult:.2f} (F&G+L/S)")

    # Final bet amount
    bet_usdc = balance * bet_pct
    bet_usdc = max(MIN_BET_USDC, min(bet_usdc, MAX_BET_USDC))

    # [v39 風控] 新模型上線 6 小時內，強制 cap 在 10% balance
    if time.time() < V39_SAFETY_END:
        v39_cap = balance * V39_MAX_BET_PCT
        if bet_usdc > v39_cap:
            print(f"🛡️  [v39 Safety] 限制 ${bet_usdc:.2f} → ${v39_cap:.2f} (10% balance, 6h 風控)")
            bet_usdc = v39_cap

    # Daily stake remaining check
    # [v1.2] 每日下注上限已移除

    print(f"💰 Bet: ${bet_usdc:.2f} ({bet_pct:.3%}) | Today: ${state.get('daily_stake_total',0):.2f}")

    # ── Execute ──
    if PREDICT_ONLY or SIM_MODE or BLOCKED_HOUR_SIM:
        pred_token = market.get("up_token") if final_dir == "UP" else market.get("down_token")
        if BLOCKED_HOUR_SIM:
            mode_tag = "SIM_HOUR_BLOCKED"
            mode_lbl = "⛔ Blocked-SIM"
        elif SIM_MODE:
            mode_tag = "SIM_PAUSED"
            mode_lbl = "⏸️ 暫停SIM"
        else:
            mode_tag = "PREDICT_ONLY"
            mode_lbl = "🔍 預測模式"
        print(f"\n{'='*60}")
        print(f"  {mode_lbl} — 只收集數據，不交易")
        print(f"  📌 方向: {final_dir} | 信心: {ml_conf:.1%}")
        print(f"  📊 預測達成（{mode_lbl}），跳過真實下單")
        print(f"{'='*60}")
        log_trade(final_dir, ml_conf, mode_tag, 0,
                  market_name=market.get("question",""), odds=up_price, slot=slot,
                  indicators={"rsi":round(rsi_val,1),"macd":round(macd_val,3)},
                  ml_signal={"direction":ml_dir,"confidence":ml_conf,"model":model_used,
                              "v4":d_v4,"v5b":d_v5b,"agreement":agreement,"primary":primary_dir},
                  clob_signal={"direction":clob_dir,"confidence":clob_conf,"obi":round(obi,4),
                               "flip":clob_flip,"strength":clob_str},
                  bet_pct=bet_pct, rsi_val=rsi_val, obi=obi,
                  balance=balance, day_pnl=day_pnl_pct,
                  token_name=pred_token)
        return

    token = market.get("up_token") if final_dir == "UP" else market.get("down_token")
    if not token:
        print("❌ No token found"); return

    # Get price from order book
    try:
        ob = client.get_order_book(token)
        best_px = float(ob.asks[0].price) if ob.asks else (up_price if final_dir == "UP" else 1-up_price)
    except socket.timeout:
        print(f"⚠️ [v5B] Orderbook timeout → use API price")
        best_px = up_price if final_dir == "UP" else 1-up_price
    except Exception as e:
        print(f"⚠️ [v5B] Orderbook error: {e}")
        best_px = up_price if final_dir == "UP" else 1-up_price
    best_px = min(max(best_px, 0.01), 0.99)

    size = max(market.get("min_size", 1), round(bet_usdc / best_px, 2))
    cost = size * best_px

    print(f"📌 {final_dir} @ {best_px:.3f} | size={size} | cost=${cost:.2f}")

    order = OrderArgs(token_id=token, price=best_px, size=size, side=BUY)
    opts  = PartialCreateOrderOptions(tick_size=market.get("tick_size", 0.01))

    result = None
    for attempt in range(3):
        try:
            signed = client.create_order(order, opts)
            result = client.post_order(signed)
            break
        except Exception as e:
            print(f"⚠️ Order retry {attempt+1}/3: {e}")
            time.sleep(2)

    if result is None:
        print("❌ Order failed after 3 retries")
        log_trade(final_dir, ml_conf, "FAILED", cost,
                  market_name=market.get("question",""), odds=best_px, slot=slot,
                  indicators={"rsi":rsi_val,"macd":macd_val},
                  ml_signal={"direction":ml_dir,"confidence":ml_conf,"model":model_used,
                              "v4":d_v4,"v5b":d_v5b,"agreement":agreement},
                  clob_signal={"direction":clob_dir,"confidence":clob_conf,"obi":obi,"flip":clob_flip},
                  bet_pct=bet_pct, rsi_val=rsi_val, obi=obi,
                  balance=balance, day_pnl=day_pnl_pct)
        return

    order_id = result.get("orderID", "")
    tx_hash  = (result.get("transactionsHashes") or [""])[0]

    if result.get("success"):
        comments = {
            "UP":   ["🚀 多頭爆發！", "📈 反彈要嚟！", "💪 準備起飛！"],
            "DOWN": ["🔻 空頭來襲！", "💨 向下走！", "⚡ 順勢而為！"],
        }
        print(f"\n{'='*60}")
        print(f"  ✅ BTC-5m New V1.4 — {final_dir}")
        print(f"  📊 {btc_price:,.0f} | RSI:{rsi_val:.0f} | OBI:{obi:+.2f}")
        print(f"  💬 {random.choice(comments[final_dir])}")
        print(f"  🔗 Order: {order_id}")
        print(f"  🔗 Tx Hash: {tx_hash}")
        print(f"{'='*60}")

        # Update state
        state["consec_wins"] = state.get("consec_wins", 0) + 1
        state["consec_loss"] = 0
        state["daily_stake_total"] = state.get("daily_stake_total", 0) + cost
        state["last_slot"] = slot
        save_state(state)

        log_trade(final_dir, ml_conf, "SUCCESS", cost, order_id, tx_hash,
                  market_name=market.get("question",""), odds=best_px, slot=slot,
                  indicators={"rsi":rsi_val,"macd":macd_val,"ema_cross":ema_cross_val,
                             "vol_delta":round(vol_delta,3),"stoch":round(stoch_n,3)},
                  ml_signal={"direction":ml_dir,"confidence":ml_conf,"model":model_used,
                              "v4":d_v4,"v5b":d_v5b,"agreement":agreement,"primary":primary_dir},
                  clob_signal={"direction":clob_dir,"confidence":clob_conf,"obi":round(obi,4),
                               "flip":clob_flip,"strength":clob_str},
                  bet_pct=bet_pct, rsi_val=rsi_val, obi=obi,
                  balance=balance, day_pnl=day_pnl_pct,
                  token_name=token)
    else:
        print(f"❌ Order not successful: {result}")
        log_trade(final_dir, ml_conf, "ERROR", cost, order_id,
                  market_name=market.get("question",""), odds=best_px, slot=slot,
                  balance=balance, day_pnl=day_pnl_pct)

def secs_fmt(s):
    m = int(s)//60; sec = int(s)%60
    return f"{m}m{sec}s"

# ── CLI Entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_trade()
