#!/usr/bin/env python3
"""
BTC-5m ConservativeV1 - 統計紀律版 + 橫向研究優化
============================================================
設計哲學:
  「1369筆教訓:只有 v17 UP 信號有統計顯著性 (WR=62.5%, p=0.0175)。
   所有其他策略版本均不顯著。紀律就是只在有意義的信號上下注。」

核心邏輯:
  1. V4 信號評分(傳統技術指標)
  2. v17 ML 模型確認(UP 方向唯一顯著信號, 15 features)
  3. v35 ML (13 features) - DOWN → +0.20 分(統計反轉信號)
  4. 雙重確認:V4評分方向 == v17方向,且 v17信心 >= 0.52
  5. 方向多樣化:3 個外部獨立信號(F&G、L/S、MomDiv)≥4 個(即全票)一致反對 ML 共識 → 翻轉方向(門檻從 ≥2 升至 ≥4,修復 Apr 5 方向翻轉災難)

  6. 【時段白名單】(2026-04-05 新增)
     只在 WR≥65% 的好時段交易:UTC [00,01,02,03,07,12,13,15,18]
     避開 WR<45% 的死亡時段:UTC [04,05,06,08,09,10,11,14,16,17,19,22,23]
     歷史數據:好時段 82筆 WR=66% P&L=+$170,壞時段 108筆 WR=38% P&L=-$234

  6. 【橫向研究新增規則】
  6a. Polymarket方向衝突:ML vs PM方向衝突 → 跟市場,放棄ML(研究:衝突時ML 100%錯誤)
  6b. Polymarket方向一致:ML = PM → 強力信號
  6c. 信心 > 56% → 倉位 ×1.2
  6d. 波幅 > $100 → 倉位 ×0.8
  6e. 橫行市 < $30 → 倉位 ×0.5

  7. 倉位 = BASE(5%) × 波幅倍數 × 信心倍數(min $3, max $15)
  8. 熔斷:連3虧→暫停2h / 日虧20%→停止 / 餘額<$20→停止
"""

import os, json, time, requests, pickle, numpy as np, importlib.util
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (ApiCreds, OrderArgs, PartialCreateOrderOptions,
                                         BalanceAllowanceParams, AssetType)
from py_clob_client.order_builder.constants import BUY

# ── 配置 ─────────────────────────────────────────────────────────────────────
HOST   = "https://clob.polymarket.com"
CHAIN  = 137
FUNDER = "0x8d8BA13d2c3D1935bF0b8BD2052AC73e8E329376"
BASE   = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader")
LOG    = os.path.join(BASE, "logs/real_trades_log.jsonl")
STATE  = os.path.join(BASE, "data/trade_state_conservative.json")

# ── 模型路徑（只用 v42 + v46）────────────────────────────────────────────────
# v42: FLAML LRL1, 43F, 4185 samples, CV=55.3%
MODEL_PATH_V42   = os.path.join(BASE, "data/ml_model_v42_flaml.pkl")
SCALER_PATH_V42  = os.path.join(BASE, "data/scaler_v42_flaml.pkl")
FEAT_PATH_V42    = os.path.join(BASE, "data/ml_features_v42.json")
# v46: GB_d3, 67F, 1449 samples, CV=52.17%, Stability=0.0267
MODEL_PATH_V46   = os.path.join(BASE, "data/ml_model_v46.pkl")
SCALER_PATH_V46  = os.path.join(BASE, "data/scaler_v46.pkl")
FEAT_PATH_V46    = os.path.join(BASE, "data/ml_features_v46.json")

# ── 風控參數(固定,無動態調整)──────────────────────────────────────────────
BASE_BET_PCT  = 0.10    # 10% 固定倉位
MIN_BET_USDC  = 3.0     # 最小 $3
MAX_BET_USDC  = 50.0    # 最大 $50
MIN_BAL_FLOOR = 20.0    # 餘額低於 $20 停止
HARD_STOP_PCT = 0.20    # 日虧 20% 停止
PAUSE_HRS     = 2      # SIM 模式持續(小時) — 正常 2小時，24小時版本已移除
MAX_CONSEC_LOSS = 3    # 連虧閾值
SIM_MODE_DURATION = PAUSE_HRS * 3600  # SIM 模式持續秒數
MIN_ENTRY_SECS  = 120   # 距收市最少秒數

# ── 預測模式 ─────────────────────────────────────────────────────────────────
PREDICT_ONLY = os.environ.get("PREDICT_ONLY", "0") == "1"

# ════════════════════════════════════════════════════════════════════════════════
# 模型加載(懶加載)
# ════════════════════════════════════════════════════════════════════════════════
# ── 模型 cache ────────────────────────────────────────────────────────────────
_mv42 = _sv42 = _feat42 = None
_mv46 = _sv46 = _feat46 = None

def load_v42():
    global _mv42
    if _mv42 is None and os.path.exists(MODEL_PATH_V42):
        with open(MODEL_PATH_V42, "rb") as f:
            _mv42 = pickle.load(f)
        print("🤖 v42 loaded (FLAML LRL1, 43F, CV=55.3%)")
    return _mv42

def load_scaler_v42():
    global _sv42
    if _sv42 is None and os.path.exists(SCALER_PATH_V42):
        with open(SCALER_PATH_V42, "rb") as f:
            _sv42 = pickle.load(f)
    return _sv42

def load_features_v42():
    global _feat42
    if _feat42 is None and os.path.exists(FEAT_PATH_V42):
        with open(FEAT_PATH_V42) as f:
            _feat42 = json.load(f)
    return _feat42

def load_v46():
    global _mv46
    if _mv46 is None and os.path.exists(MODEL_PATH_V46):
        with open(MODEL_PATH_V46, "rb") as f:
            _mv46 = pickle.load(f)
        print("🤖 v46 loaded (GB_d3, 67F, CV=52.17%)")
    return _mv46

def load_scaler_v46():
    global _sv46
    if _sv46 is None and os.path.exists(SCALER_PATH_V46):
        with open(SCALER_PATH_V46, "rb") as f:
            _sv46 = pickle.load(f)
    return _sv46

def load_features_v46():
    global _feat46
    if _feat46 is None and os.path.exists(FEAT_PATH_V46):
        with open(FEAT_PATH_V46) as f:
            _feat46 = json.load(f)
    return _feat46

# ── v43_flaml loaders (保留供 v43 stacking 用) ──────────────────────────────
_mv43 = _sv43 = _feat43 = None
def load_v43():
    global _mv43
    if _mv43 is None and os.path.exists(os.path.join(BASE, "data/ml_model_v43_flaml.pkl")):
        with open(os.path.join(BASE, "data/ml_model_v43_flaml.pkl"), "rb") as f:
            _mv43 = pickle.load(f)
        print("🤖 v43_flaml loaded (LGBM, 23F, CV=63.84%)")
    return _mv43

def load_scaler_v43():
    global _sv43
    if _sv43 is None and os.path.exists(SCALER_PATH_V43_FLAML):
        with open(SCALER_PATH_V43_FLAML, "rb") as f:
            _sv43 = pickle.load(f)
    return _sv43

def load_features_v43():
    global _feat43
    if _feat43 is None and os.path.exists(FEAT_PATH_V43_FLAML):
        with open(FEAT_PATH_V43_FLAML) as f:
            _feat43 = json.load(f)
    return _feat43

# ════════════════════════════════════════════════════════════════════════════════
# 數據獲取
# ════════════════════════════════════════════════════════════════════════════════
def fetch_btc_klines_1m(limit=100):
    """Fetch Binance 1m klines. Returns list of [open, high, low, close, volume]"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            timeout=8
        )
        data = r.json()
        return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]
    except Exception as e:
        print(f"⚠️ Binance 1m error: {e}")
        return []

def fetch_btc_klines_5m(limit=150):
    """Fetch Binance 5m klines - for v35 ML model."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit},
            timeout=8
        )
        data = r.json()
        klines = {}
        for x in data:
            ts = int(x[0]) // 1000
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

# ── 指標 helper functions(來自 real_trader_newv1.py)───────────────────────
def ema_calc(prices, period):
    if len(prices) < period: return None
    k = 2.0 / (period + 1)
    e = float(prices[0])
    for p in prices[1:]: e = p * k + e * (1 - k)
    return e

def rsi_calc(closes, p=14):
    if len(closes) < p + 1: return 50.0
    deltas = np.diff(closes[-p-1:])
    gain = sum(d for d in deltas if d > 0)
    loss = abs(sum(d for d in deltas if d < 0))
    if loss == 0: return 100.0
    return 100 - 100 / (1 + gain / loss)

def macd_hist(closes):
    if len(closes) < 26: return 0.0
    e12 = ema_calc(closes, 12)
    e26 = ema_calc(closes, 26)
    e9  = ema_calc(closes, 9)
    if e12 is None or e26 is None or e9 is None: return 0.0
    return float((e12 - e26) - e9)

def calc_vol_delta(klines):
    """Volume delta from last 5 1m klines"""
    if len(klines) < 5: return 0.0
    total = 0.0
    for i in range(1, min(6, len(klines))):
        try:
            row = klines[-i]
            if len(row) < 5: return 0.0
            o, c, vol = float(row[0]), float(row[3]), float(row[4])
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
    mu = sum(win) / len(win)
    sigma = (sum((x - mu) ** 2 for x in win) / len(win)) ** 0.5
    if sigma == 0: return 0.0
    return (closes[-1] - (mu - 2 * sigma)) / (4 * sigma + 1e-9)

def _atr(highs_arr, lows_arr, closes_arr, period=14):
    if len(closes_arr) < period + 1: return 0.0
    trs = []
    for i in range(1, len(highs_arr)):
        h, l, c = highs_arr[i], lows_arr[i], closes_arr[i]
        prev_c = closes_arr[i-1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if len(trs) < period: return 0.0
    return float(np.mean(trs[-period:]))

def fetch_taker_ratio_5m(limit=3):
    """Taker buy ratio from 5m klines"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit + 1},
            timeout=5
        )
        data = r.json()
        buy_vol  = sum(float(x[5]) for x in data[-limit:] if float(x[5]) > 0)
        sell_vol = sum(float(x[5]) for x in data[-limit:] if float(x[5]) < 0)
        ratio = abs(buy_vol / sell_vol) if sell_vol else 1.0
        return ratio
    except:
        return 1.0

# ════════════════════════════════════════════════════════════════════════════════
# V4 風格信號評分(來自 trader_v4.py)
# ════════════════════════════════════════════════════════════════════════════════
def fetch_fear_greed():
    """
    Alternative.me Fear & Greed Index
    返回: (value_0_100, direction_str)
    極端恐慌 (<20) → 長期反轉向上(傾向 UP)
    極端貪婪 (>80) → 長期反轉向下(傾向 DOWN)
    """
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = r.json()
        val = int(data["data"][0]["value"])
        if val < 20:   return val, "UP"
        elif val > 80: return val, "DOWN"
        else:          return val, "NEUTRAL"
    except:
        return None, "UNAVAILABLE"

def fetch_ls_ratio():
    """
    Binance Futures Long/Short Ratio
    返回: (ratio_float, direction_str)
    多方比例 >2.0x → 多方槓桿過度擁擠 → 傾向 DOWN(反轉)
    空方比例 >2.0x → 空方槓桿過度擁擠 → 傾向 UP(反轉)
    """
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "5m", "limit": 5},
            timeout=5
        )
        data = r.json()
        if not data: return None, "UNAVAILABLE"
        ls = sum(float(x["longShortRatio"]) for x in data[-5:]) / 5
        if ls > 2.0:   return ls, "DOWN"
        elif ls < 0.5:  return ls, "UP"
        else:           return ls, "NEUTRAL"
    except:
        return None, "UNAVAILABLE"

def calc_momentum_divergence(closes_1m, stoch, willr_val):
    """
    動量分歧檢測
    價格仍在創新低/高,但指標停滯 → 分歧 → 傾向反轉
    """
    if len(closes_1m) < 20: return "NEUTRAL"
    recent_ret = [(closes_1m[i] - closes_1m[i-1]) / closes_1m[i-1] * 100
                  for i in range(-10, -1)]
    avg_ret = sum(recent_ret) / len(recent_ret)
    price_trend = "UP" if avg_ret > 0 else "DOWN"
    stoch_ok = stoch > 50
    willr_ok = willr_val > -50
    if price_trend == "UP" and not (stoch_ok and willr_ok): return "DOWN"
    elif price_trend == "DOWN" and (stoch_ok and willr_ok):   return "UP"
    return "NEUTRAL"

def diversification_signals(closes_1m, stoch, willr_val):
    """
    整合 3 個獨立於 ML 的外部信號源,用於方向多樣化
    返回: (up_count, down_count, up_signals_list, down_signals_list, fg_val, fg_dir, ls_val)
    """
    fg_val, fg_dir = fetch_fear_greed()
    ls_val, ls_dir = fetch_ls_ratio()
    mom_dir = calc_momentum_divergence(closes_1m, stoch, willr_val)

    print(f"   🌡️ F&G: {fg_val or '?'} → {fg_dir} | L/S: {f'{ls_val:.2f}' if ls_val else '?'} → {ls_dir} | MomDiv: {mom_dir}")

    up_signals, down_signals = [], []
    if fg_dir  == "UP":   up_signals.append("F&G")
    elif fg_dir  == "DOWN": down_signals.append("F&G")
    if ls_dir  == "UP":   up_signals.append("L/S")
    elif ls_dir  == "DOWN": down_signals.append("L/S")
    if mom_dir == "UP":   up_signals.append("MomDiv")
    elif mom_dir == "DOWN": down_signals.append("MomDiv")

    print(f"   🔀 多樣化: UP信號={len(up_signals)} {up_signals} | DOWN信號={len(down_signals)} {down_signals}")
    return len(up_signals), len(down_signals), up_signals, down_signals, fg_val, fg_dir, ls_val

def v4_signal_score(closes_1m, trend2h):
    """
    V4 風格評分:RSI + MACD + EMA + 動量 + 市場趨勢
    Returns: (score, rsi_val, macd_val, ema9, ema21, mom5)
    """
    if len(closes_1m) < 30:
        return 0.0, 50.0, 0.0, 0.0, 0.0, 0.0

    rsi_val  = rsi_calc(closes_1m, 14)
    macd_val = macd_hist(closes_1m)
    ema9     = ema_calc(closes_1m, 9)
    ema21    = ema_calc(closes_1m, 21)
    mom5     = (closes_1m[-1] - closes_1m[-6]) / closes_1m[-6] * 100 if len(closes_1m) >= 6 else 0

    score = 0.0

    # RSI
    if rsi_val < 30:
        score += 0.25
    elif rsi_val > 70:
        score -= 0.25
    elif rsi_val < 40:
        score += 0.12
    elif rsi_val > 60:
        score -= 0.12

    # MACD
    score += 0.20 if macd_val > 0 else -0.20

    # EMA cross
    score += 0.20 if ema9 > ema21 else -0.20

    # Momentum
    if mom5 > 0.1:
        score += 0.15
    elif mom5 < -0.1:
        score -= 0.15

    # Market trend (2h)
    if trend2h > 0.60:
        score += 0.20
    elif trend2h < 0.40:
        score -= 0.20

    return score, rsi_val, macd_val, ema9, ema21, mom5

# ════════════════════════════════════════════════════════════════════════════════
# v35 ML 推理(13 features,與 real_trader_newv1.py ml_predict_v6_features 一致)
# ════════════════════════════════════════════════════════════════════════════════
def predict_v35(rsi_val, macd_val, ema_cross_val, vol_delta, stoch, bb_pos,
               atr_val, willr_val, mom_1m, mom_15m, taker, hour_utc):
    """
    v35: 13 features
    [rsi, macd_h, ema_cross, vol_delta, stoch, bb, atr, willr, mom1, mom15, taker, clob_obi, hour]
    """
    mv35 = load_v35()
    sc35 = load_scaler_v35()
    if mv35 is None or sc35 is None:
        return "N/A", 0.0

    X = np.array([[
        rsi_val, macd_val, ema_cross_val, vol_delta,
        stoch, bb_pos, atr_val, willr_val,
        mom_1m, mom_15m, taker, 0.0, hour_utc  # clob_obi=0.0
    ]], dtype=np.float32)

    try:
        X_sc = sc35.transform(X)
        p = mv35.predict_proba(X_sc)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(float(p[1]), float(p[0]))
        return d, c
    except Exception as e:
        print(f"⚠️ v35 predict error: {e}")
        return "N/A", 0.0

# ════════════════════════════════════════════════════════════════════════════════
# v17 ML 推理(15 features)
# ════════════════════════════════════════════════════════════════════════════════
def predict_v17(rsi_val, macd_val, ema_cross_val, vol_delta, mom_5m,
                mom_1m, stoch, bb_pos, atr_val, willr_val,
                rsi_fast, taker, taker_avg):
    """
    v17: 15 features
    [rsi, macd_h, ema_cross, vol_delta, momentum_5m, mom1, stoch, bb, atr, willr,
     rsi_f, taker, taker_avg, clob_obi, clob_ibi]
    """
    mv17 = load_v17()
    sc17 = load_scaler_v17()
    if mv17 is None or sc17 is None:
        return "N/A", 0.0

    X = np.array([[
        rsi_val, macd_val, ema_cross_val, vol_delta, mom_5m,
        mom_1m, stoch, bb_pos, atr_val, willr_val,
        rsi_fast, taker, taker_avg, 0.0, 0.0  # clob_obi=0, clob_ibi=0
    ]], dtype=np.float32)

    try:
        X_sc = sc17.transform(X)
        p = mv17.predict_proba(X_sc)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = max(float(p[1]), float(p[0]))
        return d, c
    except Exception as e:
        print(f"⚠️ v17 predict error: {e}")
        return "N/A", 0.0

# ════════════════════════════════════════════════════════════════════════════════
# v41 Neural Network 預測(83 features, MLP 128x64)
# ════════════════════════════════════════════════════════════════════════════════
def _ema_c(data, n):
    if len(data) < n: return None
    k = 2.0/(n+1); ema = sum(data[:n])/n
    for v in data[n:]: ema = v*k + ema*(1-k)
    return ema

def _rsi_c(data, n):
    if len(data) < n+1: return 50.0
    deltas = [data[i]-data[i-1] for i in range(1,len(data))]
    gain = sum(d for d in deltas[-n:] if d>0)/n
    loss = sum(-d for d in deltas[-n:] if d<0)/n
    if loss == 0: return 100.0
    return 100-(100/(1+gain/loss))

def _stoch_c(highs, lows, closes, n=14):
    if len(closes) < n: return 50.0
    lo = min(lows[-n:]); hi = max(highs[-n:])
    if hi == lo: return 50.0
    return (closes[-1]-lo)/(hi-lo)*100

def build_v41_features(klines_5m_list, slot_ts):
    """Build 83 v41 features from 5m kline list [[ts,O,H,L,C,V,...],...]"""
    slot_ts_ms = slot_ts * 1000
    idx = None
    for i, k in enumerate(klines_5m_list):
        if k[0]+300000 <= slot_ts_ms:
            idx = i
    if idx is None or idx < 30: return None
    lookback = klines_5m_list[max(0,idx-200):idx+1]
    if len(lookback) < 30: return None

    o = [k[1] for k in lookback]
    c = [k[4] for k in lookback]
    h = [k[2] for k in lookback]
    l = [k[3] for k in lookback]
    v = [k[5] for k in lookback]
    tb = [k[7] for k in lookback] if len(k) > 7 else [0]*len(lookback)
    q  = [k[6] for k in lookback] if len(k) > 6 else [0]*len(lookback)

    c_p = c[:-1]; h_p = h[:-1]; l_p = l[:-1]
    v_p = v[:-1]; tb_p = tb[:-1]; q_p = q[:-1]; o_p = o[:-1]
    price = c[-1]

    if len(c_p) < 30: return None

    # EMA
    ema5  = _ema_c(c_p, 5); ema9  = _ema_c(c_p, 9)
    ema12 = _ema_c(c_p, 12); ema20 = _ema_c(c_p, 20)
    ema50 = _ema_c(c_p, 50); ema200 = _ema_c(c_p, 200)
    ema26 = _ema_c(c_p, 26)

    # RSI
    rsi5_v  = _rsi_c(c_p, 5); rsi14_v = _rsi_c(c_p, 14); rsi28_v = _rsi_c(c_p, 28)
    rsi_div = (rsi5_v - rsi14_v)/100.0
    rsi_ext = (1 if rsi14_v < 25 else 0) + (1 if rsi14_v > 75 else 0)
    rsi_prev5 = _rsi_c(c_p[-6:-1], 5) if len(c_p) >= 6 else 50.0
    rsi_trend_v = (rsi5_v - rsi_prev5)/100.0

    # MACD
    macd_v = (ema12 - ema26)/price if (ema12 and ema26) else 0.0
    signal_v = _ema_c(list(c_p[-27:]), 9) if len(c_p) >= 27 else macd_v
    macd_hist = (macd_v - signal_v)/price if price else 0.0

    # Bollinger
    m20 = sum(c_p[-20:])/20; s20 = (sum((x-m20)**2 for x in c_p[-20:])/20)**0.5
    bb_up = m20+2*s20; bb_lo = m20-2*s20
    bb_pos = max(0.0, min(1.0, (price-bb_lo)/(bb_up-bb_lo+1e-9)))
    bb_w = (bb_up-bb_lo)/m20 if m20 else 0.0
    bb_sq = 1 if bb_w < 0.05 else 0

    # ATR
    trs = [max(h_p[i]-l_p[i], abs(h_p[i]-c_p[i-1]), abs(l_p[i]-c_p[i-1])) for i in range(1,len(c_p))]
    atr14_v = sum(trs[-14:])/14 if len(trs)>=14 else (sum(trs)/len(trs) if trs else 0)
    atr5_v  = sum(trs[-5:])/5 if len(trs)>=5 else (sum(trs)/len(trs) if trs else 0)
    atr_n = atr14_v/price if price else 0.0
    atr_ratio = atr5_v/atr14_v if atr14_v > 0 else 1.0

    # Stochastic
    sk = _stoch_c(h_p[-14:], l_p[-14:], c_p[-14:])
    sd = _stoch_c(h_p[-15:-1], l_p[-15:-1], c_p[-15:-1])
    sk_n = sk/100.0; sd_n = sd/100.0; sk_m_sd = (sk-sd)/100.0
    sk_diff = sk-sd
    sk_cross_up = 1 if (sk > sd) and (c_p[-1] > c_p[-2]) else 0
    sk_cross_dn = 1 if (sk < sd) and (c_p[-1] < c_p[-2]) else 0

    # Momentum
    mom1_v  = (c_p[-1]-c_p[-2])/c_p[-2]*100 if len(c_p)>=2 else 0.0
    mom3_v  = (c_p[-1]-c_p[-4])/c_p[-4]*100 if len(c_p)>=4 else 0.0
    mom5_v  = (c_p[-1]-c_p[-6])/c_p[-6]*100 if len(c_p)>=6 else 0.0
    mom10_v = (c_p[-1]-c_p[-11])/c_p[-11]*100 if len(c_p)>=11 else 0.0
    mom15_v = (c_p[-1]-c_p[-16])/c_p[-16]*100 if len(c_p)>=16 else 0.0
    mom30_v = (c_p[-1]-c_p[-31])/c_p[-31]*100 if len(c_p)>=31 else 0.0
    mom_accel = mom5_v - mom10_v
    mom_div = mom5_v + mom30_v

    # Volume
    vs20 = sum(v_p[-20:])/20 if len(v_p)>=20 else sum(v_p)/max(len(v_p),1)
    vd = (v_p[-1]-vs20)/vs20 if vs20 > 0 else 0.0
    vs5  = sum(v_p[-5:])/5 if len(v_p)>=5 else sum(v_p)/max(len(v_p),1)
    vr = v_p[-1]/vs20 if vs20 > 0 else 1.0
    vr5 = vs5/vs20 if vs20 > 0 else 1.0
    _rng = range(max(0, len(v_p)-3), len(v_p))
    _rng2 = range(max(0, len(v_p)-3), len(v_p))
    vol_f = 1 if all(v_p[i] < v_p[i-1] for i in _rng2) else 0
    vol_r = 1 if all(v_p[i] > v_p[i-1] for i in _rng) else 0
    _rng3 = range(max(0, len(v_p)-3), len(v_p))
    vol_f = 1 if all(v_p[i] < v_p[i-1] for i in _rng3) else 0
    vol_sg = 1 if vr > 2.0 else 0

    # Taker
    ltb = tb_p[-1] if tb_p else 0; lv = v_p[-1] if v_p else 1
    tr = (ltb/lv-0.5)*2 if lv > 0 else 0.0
    tb20 = sum(tb_p[-20:])/20 if len(tb_p)>=20 else sum(tb_p)/max(len(tb_p),1)
    v20  = sum(v_p[-20:])/20 if len(v_p)>=20 else sum(v_p)/max(len(v_p),1)
    ta = (tb20/(v20+1e-9)-0.5)*2
    tk_su = 1 if tr > 0.3 else 0
    tk_sd = 1 if tr < -0.3 else 0
    consec_up_n = 0
    consec_dn_n = 0
    _rng_c = range(1, min(6, len(c_p)))
    for i in _rng_c:
        if len(c_p) >= 2:
            if c_p[-i] > c_p[-i-1]:
                consec_up_n += 1
            if c_p[-i] < c_p[-i-1]:
                consec_dn_n += 1
    c5u = 1 if consec_up_n >= 4 else 0
    c5d = 1 if consec_dn_n >= 4 else 0

    # Candlestick
    body = abs(c_p[-1]-o_p[-1]); rng = h_p[-1]-l_p[-1]
    us = h_p[-1]-max(c_p[-1],o_p[-1])
    ls = min(c_p[-1],o_p[-1])-l_p[-1]
    hm = 1 if (ls>2*body and us<body*0.3 and rng>0) else 0
    ss = 1 if (us>2*body and ls<body*0.3 and rng>0) else 0
    dj = 1 if body<rng*0.1 else 0
    ib = 1 if (h_p[-1]<h_p[-2] and l_p[-1]>l_p[-2]) else 0
    ob = 1 if (h_p[-1]>h_p[-2] and l_p[-1]<l_p[-2]) else 0
    bue = 1 if (c_p[-1]>o_p[-1] and c_p[-2]<o_p[-2] and c_p[-1]>o_p[-2] and c_p[-2]<o_p[-1]) else 0
    bee = 1 if (c_p[-1]<o_p[-1] and c_p[-2]>o_p[-2] and c_p[-1]<o_p[-2] and c_p[-2]>o_p[-1]) else 0

    # VWAP & Range
    cum_q = sum(q_p); cum_vv = sum(v_p)
    vwap = cum_q/cum_vv if cum_vv > 0 else price
    vw_d = (price-vwap)/price if price else 0.0
    d_rng = h_p[-1]-l_p[-1]
    rp = (price-l_p[-1])/d_rng if d_rng>0 else 0.5
    rth = 1 if rp>0.9 else 0; rtl = 1 if rp<0.1 else 0

    # Rolling high/low
    rhi20 = max(h_p[-20:]) if len(h_p)>=20 else max(h_p)
    rlo20 = min(l_p[-20:]) if len(l_p)>=20 else min(l_p)
    rhi50 = max(h_p[-50:]) if len(h_p)>=50 else max(h_p)
    rlo50 = min(l_p[-50:]) if len(l_p)>=50 else min(l_p)
    pvrhi = (price-rhi20)/rhi20 if rhi20 else 0.0
    pvrlo = (price-rlo20)/rlo20 if rlo20 else 0.0
    dfh = (rhi50-price)/rhi50 if rhi50 else 0.0
    dfl = (price-rlo50)/rlo50 if rlo50 else 0.0

    # Trend
    t5_20 = (ema5-ema20)/ema20 if ema20 else 0.0
    t20_50 = (ema20-ema50)/ema50 if ema50 else 0.0
    ts = abs(t5_20)
    adx_p = atr14_v/(price+1e-9)*100

    # Return distribution
    rets = [(c_p[i]-c_p[i-1])/c_p[i-1]*100 for i in range(1,len(c_p))]
    rs5  = np.std(c_p[-6:]) if len(c_p)>=6 else 0.0
    rs10 = np.std(c_p[-11:]) if len(c_p)>=11 else 0.0
    rs20 = np.std(c_p[-21:]) if len(c_p)>=21 else 0.0
    rm = np.mean(rets[-20:]) if len(rets)>=20 else (np.mean(rets) if rets else 0.0)
    rsk = np.mean([(r-rm)**3 for r in rets[-20:]])/(rs20**3+1e-9) if rs20>0 else 0.0

    # Session
    hr = datetime.fromtimestamp(slot_ts, tz=timezone.utc).hour
    dow = datetime.fromtimestamp(slot_ts, tz=timezone.utc).weekday()
    ias = 1 if 0<=hr<8 else 0
    ius = 1 if 13<=hr<21 else 0
    iwe = 1 if dow>=5 else 0
    hp = 1 if hr in (4,14,15) else 0

    # Ichimoku
    tk = (max(h_p[-9:])+min(l_p[-9:]))/2 if len(h_p)>=9 else price
    kj = (max(h_p[-26:])+min(l_p[-26:]))/2 if len(h_p)>=26 else price
    sa = (tk+kj)/2
    sb = (max(h_p[-52:])+min(l_p[-52:]))/2 if len(h_p)>=52 else price
    ct = (sb-sa)/price if price else 0.0
    pvc = (price-sa)/price if price else 0.0
    abc = 1 if price>sa and price>sb else 0
    blc = 1 if price<sa and price<sb else 0

    return np.array([[
        # EMA & Trend
        (ema5-ema20)/price/10.0 if ema20 else 0.0,
        (ema9-ema20)/price/10.0 if ema20 else 0.0,
        (ema20-ema50)/price/10.0 if ema50 else 0.0,
        (price-ema5)/price/10.0 if ema5 else 0.0,
        (price-ema20)/price/10.0 if ema20 else 0.0,
        (price-ema50)/price/10.0 if ema50 else 0.0,
        (price-ema200)/price/10.0 if ema200 else 0.0,
        # RSI
        rsi5_v/100.0, rsi14_v/100.0, rsi28_v/100.0, rsi_div, float(rsi_ext), rsi_trend_v,
        # MACD
        macd_v/10.0, macd_hist/10.0,
        # Bollinger
        bb_pos, bb_w/10.0, float(bb_sq),
        # Stochastic
        sk_n, sd_n, sk_m_sd, sk_diff/100.0, float(sk_cross_up), float(sk_cross_dn),
        # ATR
        atr_ratio, atr_n/10.0,
        # Momentum
        mom1_v/10.0, mom3_v/10.0, mom5_v/10.0, mom10_v/10.0, mom15_v/10.0, mom30_v/10.0, mom_accel/10.0, mom_div/10.0,
        # Volume
        vd, vr/5.0, vr5/5.0, float(vol_r), float(vol_f), float(vol_sg),
        # Taker
        tr, ta/10.0, tr, float(tk_su), float(tk_sd),
        # Consecutive
        consec_up_n/5.0, consec_dn_n/5.0, float(c5u), float(c5d),
        # Candlestick
        float(hm), float(ss), float(dj), float(ib), float(ob), float(bue), float(bee),
        # VWAP & Range
        vw_d/10.0, rp, float(rth), float(rtl),
        # Rolling
        pvrhi/10.0, pvrlo/10.0, dfh/10.0, dfl/10.0,
        # Trend
        ts/10.0, t5_20/10.0, adx_p/10.0,
        # Return dist
        rs5/10.0, rs10/10.0, rs20/10.0, rsk/10.0,
        # Session
        np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24), hr/24.0,
        float(ias), float(ius), float(iwe), dow/6.0, float(hp),
        # Ichimoku
        ct/10.0, pvc/10.0, float(abc), float(blc),
    ]], dtype=np.float32)

# ── v39 Feature Builder (copied from real_trader_newv1.py) ──────────────────
# Required by predict_flaml_v42
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


def predict_v46(klines_5m=None, slot_ts=0, fng=50, ls_ratio=1.0,
                 btc_price=70000, slot_range=50, div_up=0, div_down=0):
    """V46 GB_d3 — GradientBoosting, 67 features, 1449 labeled samples.
    CV=52.17% ± 2.54% | Stability ±2.67% | No saturation.

    Computes all 67 features from klines + external signals, then
    uses the GB model to predict direction + confidence.

    Returns: (direction, confidence)
    """
    try:
        import math  # local import inside function
        model  = load_v46()
        feat_meta = load_features_v46()
        if model is None or feat_meta is None:
            return "N/A", 0.0
        feat_names = feat_meta if isinstance(feat_meta, list) else feat_meta.get('features', [])

        # ── Compute all 67 features from klines ──────────────────────────────
        f = {}

        if klines_5m and len(klines_5m) >= 30:
            closes = np.array([float(k[4]) for k in klines_5m])
            highs  = np.array([float(k[2]) for k in klines_5m])
            lows   = np.array([float(k[3]) for k in klines_5m])
            vols   = np.array([float(k[5]) for k in klines_5m])
            taker_buy = np.array([float(k[7]) for k in klines_5m]) if len(klines_5m[0]) > 7 else np.ones_like(closes) * 0.5

            c = closes[-1]
            p = c  # current price

            def ema_arr(arr, n):
                k = 2/(n+1)
                out = float(arr[0])
                for v in arr[1:]:
                    out = float(v)*k + out*(1-k)
                return out

            # ── Raw indicators ──────────────────────────────────────────────
            # EMA
            ema5  = ema_arr(closes, 5)
            ema9  = ema_arr(closes, 9)
            ema12 = ema_arr(closes, 12)
            ema20 = ema_arr(closes, 20)
            ema50 = ema_arr(closes, 50)

            # EMA cross
            f['ema_cross_raw']   = (ema5  - ema20) / p * 100 if p else 0.0
            f['ema_cross_9_20']  = (ema9  - ema20) / p * 100 if p else 0.0
            f['ema5_n']          = ema5  / p if p else 1.0
            f['ema20_n']         = ema20 / p if p else 1.0

            # Price vs EMA
            f['price_vs_ema5_n']  = (p - ema5)  / p * 100 if p else 0.0
            f['price_vs_ema20_n'] = (p - ema20) / p * 100 if p else 0.0
            f['price_vs_ema50_n'] = (p - ema50) / p * 100 if p else 0.0

            # MACD
            ema26 = ema_arr(closes, 26)
            macd_val = (ema12 - ema26) / p * 100 if p else 0.0
            f['macd_n'] = macd_val

            # RSI
            deltas = np.diff(closes)
            gains  = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg14_g = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
            avg14_l = max(float(np.mean(losses[-14:])) if len(losses) >= 14 else float(np.mean(losses)), 1e-8)
            rs14 = avg14_g / avg14_l
            rsi14 = 100 - (100 / (1 + rs14))
            avg5_g = float(np.mean(gains[-5:])) if len(gains) >= 5 else avg14_g
            avg5_l = max(float(np.mean(losses[-5:])) if len(losses) >= 5 else avg14_l, 1e-8)
            rsi5  = 100 - (100 / (1 + avg5_g/avg5_l))
            f['rsi14_n']    = rsi14 / 100.0
            f['rsi5_n']     = rsi5  / 100.0
            f['rsi5_m14']   = (rsi5 - rsi14) / 100.0

            # Stochastic
            period = min(14, len(closes))
            cmin = float(closes[-period:].min())
            cmax = float(closes[-period:].max())
            stoch_k = 100 * (c - cmin) / (cmax - cmin + 1e-8) if cmax != cmin else 50.0
            stoch_d = stoch_k * 0.7 + 50 * 0.3
            f['stoch_k_n'] = stoch_k / 100.0
            f['stoch_d_n'] = stoch_d / 100.0

            # Williams %R
            hn = float(highs[-period:].max())
            ln = float(lows[-period:].min())
            willr = -100 * (hn - c) / (hn - ln + 1e-8) if hn != ln else -50.0
            f['willr_n'] = willr / 100.0

            # ATR
            trs = [max(float(highs[i])-float(lows[i]),
                        abs(float(highs[i])-closes[i-1]),
                        abs(float(lows[i])-closes[i-1]))
                   for i in range(1, len(closes))]
            atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
            f['atr_n'] = atr / p * 100 if p else 0.0

            # Bollinger position
            ma20 = float(np.mean(closes[-20:]))
            sd20 = max(float(np.std(closes[-20:])), 1e-8)
            bb_l = ma20 - 2*sd20
            bb_u = ma20 + 2*sd20
            f['bb_pos_n'] = (c - bb_l) / (bb_u - bb_l + 1e-8)

            # Momentum
            mom1  = (c - closes[-2])  / closes[-2]  * 100 if closes[-2]  != 0 else 0.0
            mom5  = (c - closes[-6])  / closes[-6]  * 100 if len(closes) >= 6 and closes[-6]  != 0 else 0.0
            mom10 = (c - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 and closes[-10] != 0 else mom5
            mom15 = (c - closes[-15]) / closes[-15] * 100 if len(closes) >= 15 and closes[-15] != 0 else mom5
            f['mom1_n']  = mom1  / 100.0
            f['mom5_n']  = mom5  / 100.0
            f['mom10_n'] = mom10 / 100.0
            f['mom15_n'] = mom15 / 100.0

            # Consecutive
            consec_up = consec_down = 0
            for i in range(len(closes)-1, -1, -1):
                if closes[i] > closes[i-1]: consec_up += 1
                else: break
            for i in range(len(closes)-1, -1, -1):
                if closes[i] < closes[i-1]: consec_down += 1
                else: break
            f['consec_up']   = min(consec_up / 10.0, 1.0)
            f['consec_down'] = min(consec_down / 10.0, 1.0)

            # Volume
            vol5  = float(np.mean(vols[-5:]))  if len(vols) >= 5 else float(np.mean(vols))
            vol20 = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
            vol20 = max(vol20, 1e-8)
            vol_delta = (vol5 / vol20 - 1)
            f['vol_delta_n'] = vol_delta

            # Volume drifting
            ma5v = float(np.mean(closes[-5:]))
            f['vol_drifting_up']   = 1.0 if c > ma5v else 0.0
            f['vol_drifting_down'] = 1.0 if c < ma5v else 0.0

            # Taker buy ratio
            tb5  = float(np.mean(taker_buy[-5:]))  if len(taker_buy) >= 5 else float(np.mean(taker_buy))
            tb20 = float(np.mean(taker_buy[-20:])) if len(taker_buy) >= 20 else float(np.mean(taker_buy))
            taker_val = tb5 / max(tb20, 1e-8) if tb20 > 0 else 1.0
            f['taker_n']     = taker_val
            f['taker_avg_n'] = tb20

            # OBI (taker buy pressure)
            obi_taker = tb5 - 0.5  # centered at 0
            f['obi_taker'] = obi_taker

            # Range position
            rmin = float(lows[-5:].min())
            rmax = float(highs[-5:].max())
            f['range_pos_n'] = (c - rmin) / (rmax - rmin + 1e-8) if rmax != rmin else 0.5

            # Price slope 20
            f['price_slope20_n'] = mom15 / 100.0

            # Skewness of returns
            returns = np.diff(closes) / closes[:-1]
            m_r = float(np.mean(returns))
            s_r = max(float(np.std(returns)), 1e-8)
            sk = float(np.mean(((returns - m_r)/s_r)**3)) if s_r > 1e-8 else 0.0
            f['sk_m_sd_n'] = sk * s_r

        else:
            # Default values when no klines
            for fn in feat_names:
                if fn not in ('hour_utc', 'hour_cos', 'hour_sin', 'hour_good', 'hour_bad',
                               'fng', 'fng_extreme_fear', 'fng_fear', 'ls_ratio', 'ls_crowded',
                               'ls_bullish', 'f', 'ls_bearish', 'fear_crowded_contrarian',
                               'rsi_mom_bullish', 'rsi_mom_bearish', 'vol_delta_n',
                               'vol_drifting_up', 'vol_drifting_down', 'div_up', 'div_down',
                               'consec_up', 'consec_down', 'consec_up_only', 'consec_down_only'):
                    f[fn] = 0.0

        # ── External signals ────────────────────────────────────────────────────
        hour_utc = float((slot_ts % 86400) // 3600) if slot_ts > 0 else 12.0
        f['hour_utc']   = hour_utc
        f['hour_cos']   = math.cos(2 * math.pi * hour_utc / 24)
        f['hour_sin']   = math.sin(2 * math.pi * hour_utc / 24)
        f['hour_good']  = 1.0 if hour_utc in (2, 8, 12) else 0.0
        f['hour_bad']   = 1.0 if hour_utc in (0, 1, 6, 14, 15, 18, 22, 23) else 0.0
        f['fng']         = float(fng) if fng else 50.0
        f['fng_extreme_fear'] = 1.0 if f['fng'] < 20 else 0.0
        f['fng_fear']    = 1.0 if f['fng'] < 30 else 0.0
        f['ls_ratio']    = float(ls_ratio) if ls_ratio else 1.0
        f['ls_crowded']  = 1.0 if f['ls_ratio'] >= 1.3 else 0.0
        f['ls_bullish']  = 1.0 if 1.1 <= f['ls_ratio'] < 1.3 else 0.0

        # ── Engineered bins ─────────────────────────────────────────────────────
        rsi14 = (f.get('rsi14_n', 0.5)) * 100
        mom1  = f.get('mom1_n', 0.0) * 100
        mom5  = f.get('mom5_n', 0.0) * 100
        willr = (f.get('willr_n', -0.5)) * 100
        stoch_k = (f.get('stoch_k_n', 0.5)) * 100
        ema_cross = f.get('ema_cross_raw', 0.0)
        rp = f.get('range_pos_n', 0.5)
        fng_v = f.get('fng', 50)
        ls_v  = f.get('ls_ratio', 1.0)

        f['rsi_30_40']      = 1.0 if 30 <= rsi14 < 40 else 0.0
        f['rsi_50_60']      = 1.0 if 50 <= rsi14 < 60 else 0.0
        f['rsi_60_70']      = 1.0 if 60 <= rsi14 < 70 else 0.0
        f['rsi_70_100']     = 1.0 if rsi14 >= 70 else 0.0
        f['rsi_extreme']    = 1.0 if rsi14 < 30 or rsi14 > 70 else 0.0
        f['rsi_oversold']   = 1.0 if rsi14 < 40 else 0.0

        f['ema_positive']   = 1.0 if ema_cross > 0 else 0.0
        f['ema_negative']   = 1.0 if ema_cross < 0 else 0.0

        f['mom_flat']       = 1.0 if abs(mom1) < 10 else 0.0
        f['mom_positive']  = 1.0 if mom5 > 0 else 0.0
        f['mom_negative']  = 1.0 if mom5 < 0 else 0.0
        f['mom_mixed']    = 1.0 if mom1 * mom5 < 0 else 0.0

        f['stoch_overbought'] = 1.0 if stoch_k > 80 else 0.0
        f['stoch_oversold']   = 1.0 if stoch_k < 20 else 0.0

        f['range_upper']   = 1.0 if rp > 0.8 else 0.0
        f['range_lower']   = 1.0 if rp < 0.2 else 0.0
        f['range_middle'] = 1.0 if 0.3 <= rp <= 0.7 else 0.0

        f['vol_delta_pos'] = 1.0 if f.get('vol_delta_n', 0) > 0 else 0.0
        f['vol_delta_neg'] = 1.0 if f.get('vol_delta_n', 0) < 0 else 0.0
        f['obi_positive']  = 1.0 if f.get('obi_taker', 0) > 0 else 0.0

        f['consec_up_only']   = f.get('consec_up', 0.0)
        f['consec_down_only'] = f.get('consec_down', 0.0)

        # Composite
        f['rsi_mom_bullish']      = f['rsi_50_60'] * f['mom_positive']
        f['rsi_mom_bearish']      = (1.0 if rsi14 > 60 else 0.0) * f['mom_negative']
        f['fear_crowded_contrarian'] = f['fng_extreme_fear'] * f['ls_crowded']

        # ── Build feature vector ─────────────────────────────────────────────
        feat_arr = np.array([[f.get(fn, 0.0) for fn in feat_names]], dtype=float)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # GB model prediction
        p = model.predict_proba(feat_arr)[0]
        p_up = float(p[1])
        d = "UP" if p_up >= 0.5 else "DOWN"
        c = float(max(p_up, 1.0 - p_up))
        return d, c

    except Exception as e:
        print(f"⚠️ predict_v46 error: {e}")
        return "N/A", 0.0
def predict_flaml_v42(klines_5m_list, slot_ts):
    """FLAML v42 PRIMARY 預測，使用 v39 相同的 43 features。
    Returns: (direction, confidence, model_or_None)"""
    try:
        model  = load_v42()
        scaler = load_scaler_v42()
        feat_names = load_features_v42()
        if model is None or scaler is None or feat_names is None:
            return "N/A", 0.0, None
        feat = build_v39_features(klines_5m_list, slot_ts)
        if feat is None:
            return "N/A", 0.0, None
        X = np.array([[feat.get(fn, 0.0) for fn in feat_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)
        p = model.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = float(max(p[0], p[1]))
        return d, c, model
    except Exception as e:
        print(f"⚠️ predict_flaml_v42 error: {e}")
        return "N/A", 0.0, None

def predict_v43(klines_5m_list, slot_ts):
    """v43_flaml PRIMARY — LGBM 23F, CV=63.84%, trained 2026-04-07
    Stacking: uses v42 prediction probabilities as 3 input features.
    Returns: (direction, confidence, model)"""
    try:
        model  = load_v43()
        scaler = load_scaler_v43()
        feat_meta = load_features_v43()
        if model is None or scaler is None or feat_meta is None:
            return "N/A", 0.0, None
        feat = build_v39_features(klines_5m_list, slot_ts)
        if feat is None:
            return "N/A", 0.0, None
        # Get v42 predictions for stacking features
        try:
            mv42 = load_v42(); sv42 = load_scaler_v42(); fn42 = load_features_v42()
            if mv42 and sv42 and fn42:
                X42 = np.array([[feat.get(fn, 0.0) for fn in fn42]])
                X42 = np.nan_to_num(X42, nan=0.0, posinf=0.0, neginf=0.0)
                X42 = sv42.transform(X42)
                p42 = mv42.predict_proba(X42)[0]
                feat['v39_p_up'] = float(p42[1])
                feat['v39_p_down'] = float(p42[0])
                feat['v39_conf'] = float(max(p42))
        except Exception as e42:
            feat['v39_p_up'] = 0.5; feat['v39_p_down'] = 0.5; feat['v39_conf'] = 0.5
        # Build feature vector from 23 feature names
        feat_names = feat_meta if isinstance(feat_meta, list) else feat_meta.get('feature_names', FEAT_V43_FLAML)
        X = np.array([[feat.get(fn, 0.0) for fn in feat_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)
        p = model.predict_proba(X)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = float(max(p[0], p[1]))
        return d, c, model
    except Exception as e:
        print(f"⚠️ predict_v43 error: {e}")
        return "N/A", 0.0, None

def predict_v41(klines_5m_list, slot_ts):
    """Predict using v41 MLP Neural Network (83 features)"""
    mv41 = load_v41(); sc41 = load_scaler_v41(); feat41 = load_features_v41()
    if mv41 is None or sc41 is None or feat41 is None:
        return "N/A", 0.0, None
    X = build_v41_features(klines_5m_list, slot_ts)
    if X is None:
        return "N/A", 0.0, None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        X_sc = sc41.transform(X)
        p = mv41.predict_proba(X_sc)[0]
        d = "UP" if p[1] >= p[0] else "DOWN"
        c = float(max(p[1], p[0]))
        return d, c, mv41
    except Exception as e:
        print(f"⚠️ v41 predict error: {e}")
        return "N/A", 0.0, None

# ════════════════════════════════════════════════════════════════════════════════
# 市場查詢
# ════════════════════════════════════════════════════════════════════════════════
def find_market(last_traded_slot=0):
    """找下一個活躍的 BTC-5m 市場（2026-04-09 Fix: gamma-api 失效，改用 web scraping）"""
    now = int(time.time())
    next_slot = (now // 300) * 300 + 300
    cur_slot  = (now // 300) * 300

    for slot in [next_slot, cur_slot]:
        slot_end = slot + 300
        secs_remaining = slot_end - now
        if secs_remaining < MIN_ENTRY_SECS:
            continue
        if slot == last_traded_slot:
            continue
        try:
            url = f'https://polymarket.com/event/btc-updown-5m-{slot}'
            r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                continue
            text = r.text
            # 從 HTML 提取 clobTokenIds（格式: clobTokenIds":["id1","id2"]）
            idx = text.find('clobTokenIds')
            if idx < 0:
                continue
            # 找到 [ 和 matching ]
            start = text.index('[', idx)
            depth, end = 1, start + 1
            while depth > 0 and end < len(text):
                c = text[end]
                if c == '[': depth += 1
                elif c == ']': depth -= 1
                end += 1
            ids = json.loads(text[start:end])
            if len(ids) < 2:
                continue
            # 提取 outcomePrices
            px_idx = text.find('outcomePrices', idx)
            if px_idx >= 0:
                px_start = text.index('[', px_idx)
                depth, px_end = 1, px_start + 1
                while depth > 0 and px_end < len(text):
                    c = text[px_end]
                    if c == '[': depth += 1
                    elif c == ']': depth -= 1
                    px_end += 1
                prices = json.loads(text[px_start:px_end])
            else:
                prices = [0.5, 0.5]
            # 提取 orderMinSize
            ms_idx = text.find('orderMinSize', idx)
            if ms_idx >= 0:
                ms_match = re.search(r'orderMinSize[^0-9]*([0-9.]+)', text[ms_idx:ms_idx+30])
                min_size = float(ms_match.group(1)) if ms_match else 5.0
            else:
                min_size = 5.0
            return {
                "up_token":    str(ids[0]),
                "down_token":  str(ids[1]),
                "up_price":    float(prices[0]),
                "min_size":    min_size,
                "tick_size":   "0.01",
                "question":    f"BTC-5m slot {slot}",
                "slot":        slot,
                "secs_remaining": secs_remaining,
            }
        except Exception as e:
            pass
    return None
def get_market_trend():
    """過去2小時結算市場的 UP 勝率"""
    now = int(time.time())
    up_w, dn_w = 0, 0
    for offset in range(1, 25):
        slot = (now // 300) * 300 - offset * 300
        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": f"btc-updown-5m-{slot}"},
                timeout=8
            )
            if r.status_code == 200 and r.json():
                m = r.json()[0]
                if m.get("closed"):
                    p = json.loads(m["outcomePrices"])
                    if float(p[0]) >= 0.5:
                        up_w += 1
                    else:
                        dn_w += 1
        except:
            pass
    total = up_w + dn_w
    return up_w / total if total > 0 else 0.5

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
    if client is None:
        return 0.0
    try:
        p = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
        b = client.get_balance_allowance(params=p)
        return int(b.get("balance", "0")) / 1e6
    except Exception as e:
        print(f"⚠️ get_balance error: {e}")
        return 0.0

# ════════════════════════════════════════════════════════════════════════════════
# 結算檢查
# ════════════════════════════════════════════════════════════════════════════════
def check_settlements():
    """檢查並更新未結算交易的 WIN/LOSS"""
    try:
        entries = []
        if os.path.exists(LOG):
            with open(LOG) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except:
                        pass

        cutoff_ts = int(time.time()) - 48 * 3600  # Only check trades from last 48h
        unresolved = [e for e in entries
                      if e.get("status") == "SUCCESS"
                      and not e.get("actual")
                      and e.get("version") == "ConservativeV1"
                      and e.get("slot", 0) > cutoff_ts]  # ← 只檢查48小時內的單
        if not unresolved:
            return 0

        print(f"🔍 Checking {len(unresolved)} unresolved ConservativeV1 trades...")

        slots_to_check = sorted(set(e.get("slot", 0) for e in unresolved if e.get("slot")))
        checked = {}

        # ── Web Scraping 方式（2026-04-09：gamma-api 失效，改用 Polymarket 網頁抓取） ──
        for slot in slots_to_check:
            try:
                url = f'https://polymarket.com/event/btc-updown-5m-{slot}'
                r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    text = soup.get_text().lower()
                    if 'final outcome was' in text:
                        outcome = text.split('final outcome was')[1].split('.')[0].strip().strip('"').strip()
                        if 'up' in outcome:
                            checked[slot] = ['1', '0']  # UP won
                        elif 'down' in outcome:
                            checked[slot] = ['0', '1']  # DOWN won
            except:
                pass

        updated = 0
        # 載入狀態用於更新連續虧損計數
        state = load_state()
        now = int(time.time())
        counted_slots = set()  # ← Bug Fix: 每個 slot 只計一次 consec_loss

        for e in entries:
            if e.get("status") == "SUCCESS" and not e.get("actual") and e.get("version") == "ConservativeV1":
                slot = e.get("slot", 0)
                prices = checked.get(slot)
                if prices:
                    up_won = prices[0] == '1'
                    trade_dir = e.get("direction", "")
                    actual = "WIN" if (trade_dir == "UP" and up_won) or (trade_dir == "DOWN" and not up_won) else "LOSS"
                    if actual:
                        e["actual"] = actual
                        print(f"  {'✅' if actual == 'WIN' else '❌'} Slot {slot} {trade_dir} → {actual}")
                        updated += 1
                        # ── 更新連續虧損計數（每個 slot 只計一次）──
                        if slot in counted_slots:
                            continue  # 同 slot 多筆單，跳過重複計數
                        counted_slots.add(slot)
                        if actual == "LOSS":
                            state["consec_loss"] += 1
                            print(f"  🔻 連虧計數: {state['consec_loss']}/{MAX_CONSEC_LOSS}")
                            if state["consec_loss"] >= MAX_CONSEC_LOSS:
                                state["pause_until"] = now + SIM_MODE_DURATION
                                print(f"  ⛔ 連虧觸發!進入 SIM 模式 {SIM_MODE_DURATION//60}min")
                                save_state(state)
                                return 0  # ← Bug Fix 2: 同一次 run 觸發熔斷後立即 return，不繼續下單
                        else:  # WIN
                            if state["consec_loss"] > 0:
                                print(f"  ✅ 連虧歸零 (前值={state['consec_loss']})")
                            state["consec_loss"] = 0
                            # ← Bug Fix: WIN 時不清除 pause_until，讓熔斷時間自然到期

        if updated > 0:
            with open(LOG, "w") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"✅ Updated {updated} settlement records")

            # ── 同時更新 decisions 表的 actual 欄位 ──
            try:
                import sqlite3
                db_path = os.path.join(BASE, "data/btc5m_research.db")
                con = sqlite3.connect(db_path)
                for slot, prices in checked.items():
                    up_won = prices[0] == '1'
                    # 更新所有未結算且狀態為 SUCCESS 的決策
                    cur = con.execute(
                        "SELECT id, final_dir FROM decisions WHERE slot=? AND actual IS NULL AND status='SUCCESS'",
                        (slot,)
                    )
                    rows = cur.fetchall()
                    for row_id, fdir in rows:
                        actual = 'WIN' if (fdir == 'UP' and up_won) or (fdir == 'DOWN' and not up_won) else 'LOSS'
                        con.execute("UPDATE decisions SET actual=? WHERE id=?", (actual, row_id))
                con.commit()
                con.close()
            except Exception as e2:
                print(f"⚠️ decisions DB update error: {e2}")

            save_state(state)

        return updated
    except Exception as e:
        print(f"⚠️ check_settlements error: {e}")
        return 0

# ════════════════════════════════════════════════════════════════════════════════
# 狀態管理
# ════════════════════════════════════════════════════════════════════════════════
def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {
        "balance": 0, "consec_loss": 0, "pause_until": 0,
        "daily_pnl": 0, "daily_date": "", "last_slot": 0,
        "version": "ConservativeV1"
    }

def save_state(state):
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)

# ════════════════════════════════════════════════════════════════════════════════
# 日誌
# ════════════════════════════════════════════════════════════════════════════════
import random
_JOKES = [
    "ML 模型說 DOWN,但我的直覺說 UP--我不是在交易,我在和模型吵架。",
    "這個市場今天UP了3次,我全押DOWN。職業病,無藥可救。",
    "我的止損紀律比程序員的代碼註釋還嚴格--兩者都不存在。",
    "昨晚的WR=37.5%?我不生氣,我只是在用數學計算復仇。",
    "機器學習模型告訴我明天會下雨,結果今天就在下雨--95%置信區間的教訓。",
    "我對這個市場的信仰是統計顯著的--但P-value不這麼認為。",
    "比特幣5分鐘後走向何方?這個問題連中本聰都回答不了,更何況是我。",
    "今天第三次連虧了,我冷靜地打開SOUL.md深呼吸--裡面寫著「不假裝自信」。",
    "我的止損點和行情圖的交集面積,比我的交易帳戶還大。",
    "聽說樂透的中獎率是1/45,000,000--我的WR=50%簡直就是奇蹟。",
    "機器學習模型:WR=52%。我:今天吃魚。",
    "我的交易策略基於1369筆數據,但錢包說:數據不足。",
    "市場說 UP,我說 DOWN,錢包說:你們都不對。",
    "今天最後一次機會全押 UP--不是因為有信心,是因為沒選擇。",
    "這個系統最穩定的部分是列印出來的 P&L--因為總是壞的。",
    "我的槓桿已經加滿,我的樂觀也是。兩者都危險。",
    "方向多樣化是保險,但我們的保險條款是啞巴英文。",
    "昨晚教訓:100% DOWN 不是策略,是實驗。",
    "F&G=11 是恐慌,但我的帳戶比我更恐慌。",
    "我這個交易機器人最像人的部分?就是這堆冷笑話。",
]

def tell_joke():
    joke = random.choice(_JOKES)
    print(f"  🦁 {joke}")
    return joke

def log_trade(price, direction, conf, status, amount, order_id="", tx_hash="", slot=0,
              v4_score=0.0, v17_dir="N/A", v17_conf=0.0, v35_dir="N/A", note=""):
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    rec = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "version":    "ConservativeV1",
        "price":      round(price, 4),
        "direction":  direction,
        "confidence": round(conf, 4),
        "status":     status,
        "amount":     round(amount, 4),
        "order_id":   order_id,
        "tx_hash":    tx_hash,
        "slot":       slot,
        "v4_score":   round(v4_score, 4),
        "v17_dir":    "N/A",
        "v17_conf":   0.0,
        "v35_dir":    "N/A",
        "note":       note,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def log_decision(d):
    """
    完整決策記錄 → SQLite decisions 表
    d = dict 包含所有决策因子
    可 JOIN btc5m_research.db 的 slots 表拿到實際結果
    """
    import sqlite3
    db_path = os.path.join(BASE, "data/btc5m_research.db")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT,
            ts_unix       INTEGER,
            slot          INTEGER,
            hour_utc      INTEGER,
            status        TEXT,
            sim_mode      INTEGER DEFAULT 0,
            btc_price     REAL,
            final_dir     TEXT,
            bet_amount    REAL,
            entry_price   REAL,
            -- ML Models
            v43_dir       TEXT,
            v43_conf      REAL,
            v42_dir       TEXT,
            v42_conf      REAL,
            v17_dir       TEXT,
            v17_conf      REAL,
            v35_dir       TEXT,
            v35_conf      REAL,
            v41_dir       TEXT,
            v41_conf      REAL,
            v4_score      REAL,
            -- Technical Indicators
            rsi           REAL,
            macd          REAL,
            ema_cross     REAL,
            stoch         REAL,
            willr         REAL,
            bb_pos        REAL,
            atr           REAL,
            mom_5m        REAL,
            mom_1m        REAL,
            mom_15m       REAL,
            taker         REAL,
            -- External Signals
            fng_value     INTEGER,
            fng_class     TEXT,
            ls_ratio      REAL,
            pm_up_price   REAL,
            pm_dir        TEXT,
            -- Volatility & Position
            slot_range    REAL,
            vol_class     TEXT,
            vol_mult      REAL,
            conf_mult     REAL,
            combined_mult REAL,
            -- Diversification
            div_up        INTEGER,
            div_down      INTEGER,
            div_note      TEXT,
            -- Decision Factors
            clob_override INTEGER DEFAULT 0,
            note          TEXT,
            -- Result (filled later by settlement)
            actual        TEXT DEFAULT NULL,
            pnl           REAL DEFAULT NULL
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_decisions_slot ON decisions(slot);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts_unix);
    """)

    now_dt = datetime.now(timezone.utc)
    cur.execute("""
        INSERT INTO decisions (
            ts, ts_unix, slot, hour_utc, status, sim_mode,
            btc_price, final_dir, bet_amount, entry_price,
            v43_dir, v43_conf, v42_dir, v42_conf,
            v46_dir, v46_conf,
            v17_dir, v17_conf, v35_dir, v35_conf, v41_dir, v41_conf,
            v4_score, rsi, macd, ema_cross, stoch, willr, bb_pos, atr,
            mom_5m, mom_1m, mom_15m, taker,
            fng_value, fng_class, ls_ratio,
            pm_up_price, pm_dir,
            slot_range, vol_class, vol_mult, conf_mult, combined_mult,
            div_up, div_down, div_note,
            clob_override, note, v45_dir, v45_conf
        ) VALUES (
            ?,?,?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,
            ?,?,?,?,?,
            ?,?,?,
            ?,?,?,?
        )
    """, (
        now_dt.isoformat(), int(now_dt.timestamp()),
        d.get("slot", 0), d.get("hour_utc", -1),
        d.get("status", ""), int(d.get("sim_mode", False)),
        d.get("btc_price"), d.get("final_dir", "WAIT"),
        d.get("bet_amount", 0), d.get("entry_price", 0),
        d.get("v43_dir", "N/A"), d.get("v43_conf", 0),
        d.get("v42_dir", "N/A"), d.get("v42_conf", 0),
        d.get("v46_dir", "N/A"), d.get("v46_conf", 0),
        d.get("v17_dir", "N/A"), d.get("v17_conf", 0),
        d.get("v35_dir", "N/A"), d.get("v35_conf", 0),
        d.get("v41_dir", "N/A"), d.get("v41_conf", 0),
        d.get("v4_score", 0),
        d.get("rsi"), d.get("macd"), d.get("ema_cross"),
        d.get("stoch"), d.get("willr"), d.get("bb_pos"), d.get("atr"),
        d.get("mom_5m"), d.get("mom_1m"), d.get("mom_15m"), d.get("taker"),
        d.get("fng_value"), d.get("fng_class", ""), d.get("ls_ratio"),
        d.get("pm_up_price"), d.get("pm_dir", "NEUTRAL"),
        d.get("slot_range"), d.get("vol_class", ""),
        d.get("vol_mult", 1.0), d.get("conf_mult", 1.0), d.get("combined_mult", 1.0),
        d.get("div_up", 0), d.get("div_down", 0), d.get("div_note", ""),
        int(d.get("clob_override", False)), d.get("note", ""),
        d.get("v45_dir", "N/A"), d.get("v45_conf", 0)
    ))
    con.commit()
    con.close()

# ════════════════════════════════════════════════════════════════════════════════
# 主執行
# ════════════════════════════════════════════════════════════════════════════════
def run():
    mode_tag = "🔍 PREDICT ONLY" if PREDICT_ONLY else "💰 LIVE TRADING"
    print(f"\n{'='*60}")
    print(f"  🦁 BTC-5m ConservativeV1 - 統計紀律版")
    print(f"  [{mode_tag}]")
    print(f"{'='*60}")

    # ── 結算檢查 ───────────────────────────────────────────────────────────
    check_settlements()

    # ── 初始化 Client ───────────────────────────────────────────────────────
    client = init_client()
    if client is None:
        print("❌ Client init failed"); return

    balance = get_balance(client)
    print(f"💰 餘額: ${balance:.2f}")

    # ── 同步 State File ─────────────────────────────────────────────────────
    # 確保 state file 與實際餘額和日 P&L 同步
    state = load_state()
    if abs(state.get("balance", 0) - balance) > 0.01:
        print(f"   📝 State sync: balance {state.get('balance',0):.2f} → {balance:.2f}")
        state["balance"] = round(balance, 2)

    now = int(time.time())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reset_ts = state.get("manual_reset_ts")
    # New UTC day → automatic full reset
    if state.get("daily_date", "") != today:
        state["daily_pnl"] = 0.0
        state["daily_date"] = today
        state["manual_reset_ts"] = None  # clear manual reset on new day
        print(f"   📝 New day UTC, reset daily_pnl")

    # Calculate P&L from log
    # Rule: only count trades SETTLED AFTER manual_reset_ts (Leo wants to discard pre-reset history)
    # Accumulation: only add trades not yet counted (prevents double-counting)
    try:
        import os
        LOG = os.path.join(BASE, "logs/real_trades_log.jsonl")
        last_counted_ts = state.get("last_counted_ts", "")  # last trade ts we counted
        
        if os.path.exists(LOG):
            with open(LOG) as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line.strip())
                    if e.get("actual") not in ("WIN", "LOSS"):
                        continue
                    trade_ts = e.get("ts", "")
                    today_str = today  # defined in outer scope
                    
                    # Filter: must be today
                    if not trade_ts.startswith(today_str):
                        continue
                    
                    # Filter: must be after manual_reset_ts (if set)
                    if reset_ts:
                        reset_dt = datetime.fromisoformat(reset_ts.replace("Z", "+00:00"))
                        trade_dt_str = trade_ts.replace("Z", "+00:00")
                        trade_dt = datetime.fromisoformat(trade_dt_str)
                        if trade_dt < reset_dt:
                            continue  # skip pre-reset trades
                    
                    # Skip if already counted (by comparing ts)
                    if trade_ts <= last_counted_ts:
                        continue
                    
                    # Count this trade
                    amt = float(e.get("amount", 10))
                    pnl_contrib = amt * (0.99 if e.get("actual") == "WIN" else -1.0)
                    state["daily_pnl"] = round(state.get("daily_pnl", 0.0) + pnl_contrib, 2)
                    state["last_counted_ts"] = trade_ts  # remember last counted
        
        if reset_ts:
            reset_dt_human = datetime.fromisoformat(reset_ts.replace("Z", "+00:00")).strftime("%H:%M UTC")
            print(f"   📝 Manual reset at {reset_dt_human}, counting post-reset trades only")
    except Exception as e:
        print(f"   ⚠️ daily_pnl calc error: {e}")

    save_state(state)
    print(f"   📊 daily_pnl=${state['daily_pnl']:.2f} consec_loss={state.get('consec_loss',0)}")

    # ── 日虧停止 ─────────────────────────────────────────────────────────────
    if state["daily_pnl"] <= -balance * HARD_STOP_PCT:
        print(f"⛔ 日虧觸發 ${state['daily_pnl']:.2f} > {HARD_STOP_PCT*100:.0f}% of ${balance:.2f},停止")
        return

    # ── 連虧熔斷 ─────────────────────────────────────────────────────────────
    sim_mode = False
    if state["pause_until"] > now:
        remaining = (state["pause_until"] - now) // 60
        print(f"⏸️ SIM 模式中，暫停 {remaining}min，繼續預測但不下實盤")
        sim_mode = True  # 繼續執行預測邏輯，最後下單步驟跳過
    elif state["consec_loss"] >= MAX_CONSEC_LOSS:
        # pause_until 已過期但 consec_loss 未清零 → 清理 stale state
        print(f"⚠️ 連虧{state['consec_loss']}次 且 pause_until 已過期，清理 stale state")
        state["consec_loss"] = 0
        save_state(state)
    # ── 市場數據 ─────────────────────────────────────────────────────────────
    market = find_market(state.get("last_slot", 0))
    if not market:
        print("⚠️ 無活躍市場"); return

    slot     = market["slot"]
    secs_rem = market["secs_remaining"]
    print(f"🎯 Slot {slot} | 剩餘 {secs_rem}s | Up:{market['up_price']*100:.1f}%")

    # ── 重複下單防護：檢查 log 裡是否已有同 slot SUCCESS ──────────────────────
    try:
        with open(LOG) as _lf:
            _existing = [json.loads(l) for l in _lf if l.strip()]
        _slot_success = [e for e in _existing
                         if e.get("slot") == slot
                         and e.get("status") == "SUCCESS"
                         and e.get("version") == "ConservativeV1"]
        if _slot_success:
            print(f"⏭️ Slot {slot} 已有 {len(_slot_success)} 筆 SUCCESS，跳過重複下單")
            return
    except Exception:
        pass

    # ── 1m K線 → 指標計算 ───────────────────────────────────────────────────
    klines_1m = fetch_btc_klines_1m(limit=100)
    if len(klines_1m) < 34:
        print("❌ K線不足"); return

    closes_1m = [k[3] for k in klines_1m]
    highs_1m  = [k[1] for k in klines_1m]
    lows_1m   = [k[2] for k in klines_1m]

    btc_price  = closes_1m[-1]
    rsi_val    = rsi_calc(closes_1m, 14)
    macd_val   = macd_hist(closes_1m)
    ema9       = ema_calc(closes_1m, 9)
    ema21      = ema_calc(closes_1m, 21)
    mom_5m     = (closes_1m[-1] - closes_1m[-6]) / closes_1m[-6] * 100 if len(closes_1m) >= 6 else 0
    mom_1m     = (closes_1m[-1] - closes_1m[-2]) / closes_1m[-2] * 100 if len(closes_1m) >= 2 else 0
    mom_15m    = (closes_1m[-1] - closes_1m[-16]) / closes_1m[-16] * 100 if len(closes_1m) >= 16 else mom_5m
    vol_delta  = calc_vol_delta(klines_1m)
    stoch      = calc_stoch(closes_1m, highs_1m, lows_1m, 14)
    bb_pos     = calc_bb_pos(closes_1m, 20)
    atr_val    = _atr(highs_1m, lows_1m, closes_1m, 14)
    atr_val    = atr_val / btc_price if btc_price else 0.0
    highest14  = max(highs_1m[-14:]) if len(highs_1m) >= 14 else max(highs_1m)
    lowest14   = min(lows_1m[-14:])  if len(lows_1m)  >= 14 else min(lows_1m)
    willr_val  = -100.0 * (highest14 - closes_1m[-1]) / (highest14 - lowest14) if highest14 != lowest14 else -50.0
    rsi_fast   = rsi_calc(closes_1m, 5)
    ema_cross_val = (ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0

    taker      = fetch_taker_ratio_5m(3)
    taker_avg   = fetch_taker_ratio_5m(5)
    trend2h     = get_market_trend()
    hour_utc    = datetime.now(timezone.utc).hour

    # ── 時段白名單過濾（2026-04-05 新增，根據 Apr 3-5 數據）────────────────
    # 好時段 WR≥65%: UTC 00,01,02,03,07,12,13,15,18
    # 壞時段 WR<45% (禁):    UTC 04,05,06,08,09,10,11,14,16,17,19,22,23
    # 時段 filter 已移除（2026-04-07）— 全天候交易，依賴 ML 信心過濾

    # ── V4 信號評分 ──────────────────────────────────────────────────────────
    v4_score, _, _, _, _, _ = v4_signal_score(closes_1m, trend2h)

    print(f"📊 BTC:${btc_price:,.0f} | RSI:{rsi_val:.1f} | MACD:{'▲' if macd_val>0 else '▼'} | EMA:{'▲' if ema9>ema21 else '▼'}")
    print(f"   V4 Score:{v4_score:+.2f} | Trend2h:{trend2h:.0%}")

    # ── 5m K線 → v41 Neural Network ──────────────────────────────────────────
    klines_5m_list = None
    try:
        kl5m_dict, kl5m_times = fetch_btc_klines_5m(limit=200)
        if kl5m_times:
            # Convert dict to sorted list
            klines_5m_list = []
            for t in sorted(kl5m_dict.keys()):
                k = kl5m_dict[t]
                klines_5m_list.append([
                    t,              # 0: ts
                    k['open'],       # 1: O
                    k['high'],       # 2: H
                    k['low'],        # 3: L
                    k['close'],      # 4: C
                    k['volume'],     # 5: V
                    k['quote_vol'], # 6: Q
                    k['taker_buy_vol'],  # 7: TB
                ])
    except: pass

    # ── ML 推理：只用 v42（PRIMARY）+ v46（CONFIRM）──────────────────────────
    # v42: FLAML LRL1, 43F, 4185 samples, CV=55.3%
    v42_dir, v42_conf = "N/A", 0.0
    if klines_5m_list:
        v42_dir, v42_conf, _ = predict_flaml_v42(klines_5m_list, slot)
        print(f"   ⭐ v42 PRIMARY: {v42_dir} {v42_conf:.2%}" if v42_dir != "N/A" else "   ⭐ v42 PRIMARY: unavailable")
    else:
        print("   ⭐ v42 PRIMARY: klines unavailable")

    # ── F&G + L/S fetch (needed for v46 + diversification) ─────────────────
    fng_v46 = 50.0
    ls_v46  = 1.0
    try:
        rf = requests.get("https://api.alternative.me/fng", timeout=5)
        fng_v46 = float(rf.json().get('data', [{}])[0].get('value', 50))
    except: pass
    try:
        rls = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                          params={"symbol":"BTCUSDT","period":"1h","limit":3}, timeout=5)
        ls_v46 = float(rls.json()[0].get('longShortRatio', 1.0))
    except: pass

    # ── Slot range from Polymarket prices ──────────────────────────────────
    slot_range_v46 = 0.0
    if market and market.get('up_price') and market.get('down_price'):
        slot_range_v46 = abs(market['up_price'] - market['down_price']) * 100  # cents

    # v46: GB_d3, 67F, 1449 samples, CV=52.17%
    v46_dir, v46_conf = "N/A", 0.0
    if klines_5m_list:
        v46_dir, v46_conf = predict_v46(
            klines_5m=klines_5m_list,
            slot_ts=slot,
            fng=fng_v46,
            ls_ratio=ls_v46,
            btc_price=btc_price,
            slot_range=slot_range_v46,
            div_up=0,
            div_down=0
        )
        print(f"   🔶 v46 CONFIRM: {v46_dir} {v46_conf:.2%}" if v46_dir != "N/A" else "   🔶 v46 CONFIRM: unavailable")
    else:
        print("   🔶 v46 CONFIRM: klines unavailable")

    # Fetch ls_val for diversification_signals
    ls_val = None
    try:
        r_ls = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                            params={"symbol":"BTCUSDT","period":"1h","limit":3}, timeout=5)
        rd_ls = r_ls.json()
        if rd_ls: ls_val = float(rd_ls[0].get('longShortRatio', 1.0))
    except: pass

    # ── Cascade 決策邏輯（V46 PRIMARY + V42 CONFIRM）──────────────────────────
    # 2026-04-10: V46 GB_d3 1,449樣本 CV=52.17% 升為 PRIMARY
    effective_score = v4_score
    final_dir = None
    note = ""

    if v46_dir != "N/A" and v46_conf >= 0.55:
        # V46 高信心 → 直接下注
        final_dir = v46_dir
        note = f"V46 PRIMARY HIGH {v46_dir}@{v46_conf:.2%}"
        print(f"   ✅ V46 PRIMARY 高信心({v46_conf:.2%}) → {v46_dir}")

    elif v46_dir != "N/A" and v46_conf >= 0.52:
        # V46 弱信號 → 需要 V42 同向確認
        if v42_dir == v46_dir:
            final_dir = v46_dir
            note = f"V46({v46_conf:.2%})+V42({v42_conf:.2%})確認 {v46_dir}"
            print(f"   ✅ V46+V42 同向確認 → {v46_dir}")
        else:
            print(f"   ⏸️ V46({v46_dir} {v46_conf:.2%}) vs V42({v42_dir} {v42_conf:.2%}) 衝突，觀望")

    elif v42_dir != "N/A" and v42_conf >= 0.52:
        # V46 無信心但 V42 有 → V42 單獨下注
        final_dir = v42_dir
        note = f"V42單獨 {v42_dir}@{v42_conf:.2%}"
        print(f"   ✅ V42 fallback ({v42_conf:.2%}) → {v42_dir}")

    else:
        print(f"   ⏸️ V46+V42 均無足夠信心，觀望")


    # ══ CLOB 方向衝突檢查（必須在震盪過濾之前）═══════════════════════════
    # 當 CLOB 方向與 ML 衝突且市場明確時，優先跟 CLOB
    pm_up_price = market.get("up_price", 0.5)
    pm_dir = "NEUTRAL"
    if pm_up_price >= 0.52:
        pm_dir = "UP"
    elif pm_up_price <= 0.48:
        pm_dir = "DOWN"
    clob_override = False
    if final_dir is not None and pm_dir in ("UP", "DOWN") and final_dir != pm_dir:
        # CLOB 與 ML 衝突，計算衝突強度
        pm_strength = abs(pm_up_price - 0.5)  # 0=中立, 0.5=極端
        if pm_strength >= 0.03:  # CLOB 方向明確（>3%偏離中立）
            old_dir = final_dir
            final_dir = pm_dir
            clob_override = True
            note = note + f" | CLOB override {old_dir}→{pm_dir}"
            print(f"   🔄 CLOB override: ML={old_dir} vs CLOB={pm_dir}({pm_up_price:.2f}) → FOLLOW CLOB")
    elif final_dir is None and pm_dir in ("UP", "DOWN"):
        # ML 無信號但 CLOB 有信號，用 CLOB
        pm_strength = abs(pm_up_price - 0.5)
        if pm_strength >= 0.05:  # CLOB 方向非常明確（>5%偏離中立）
            final_dir = pm_dir
            clob_override = True
            note = note + f" | CLOB-only {pm_dir}@{pm_up_price:.2f}"
            print(f"   🔄 CLOB-only signal: {pm_dir}@{pm_up_price:.2f} → TRADE")

    # ══ 震盪市場過濾(已移除舊邏輯 — 用交易結果推算市場方向容易自我鎖死) ══════
    # 直接用 ML 共識信心判斷:信心 >= 0.52 才交易，不依賴歷史交易結果
    if final_dir is not None:
        print(f"   ✅ 震盪過濾已移除，依賴 ML 信心閾值")
    # (舊代碼備註: recent_mkt 邏輯在沒有新交易時會永久鎖死)

    # ── 方向多樣化審查(獨立於 ML 的外部信號)───────────────────────────────
    div_up, div_down, div_up_signals, div_down_signals, fng_val, fng_class, ls_val = diversification_signals(closes_1m, stoch, willr_val)

    div_note = ""
    if final_dir is not None:
        if final_dir == "DOWN" and div_up >= 4:
            final_dir = "UP"
            div_note = f"🔄 多樣化翻轉: DOWN→UP ({'/'.join(div_up_signals)} 所有3票支持)"
            print(f"   {div_note}")
            tell_joke()
        elif final_dir == "UP" and div_down >= 4:
            final_dir = "DOWN"
            div_note = f"🔄 多樣化翻轉: UP→DOWN ({'/'.join(div_down_signals)} 所有3票支持)"
            print(f"   {div_note}")
            tell_joke()
        elif div_up >= 4 or div_down >= 4:
            div_note = "(外部信號一致但 ML 未達共識,維持觀望)"
            print(f"   ⏸️ {div_note}")
            tell_joke()
            final_dir = None
        else:
            div_note = "✅ 多樣化確認"
            tell_joke()

    # ══ NEW: Polymarket 方向衝突檢查 + 波幅 + 倉位倍數 ══════════════════════
    # 橫向研究結論:ML與Polymarket方向衝突時,ML 100%錯誤
    # ── Step 1: 波幅計算 ───────────────────────────────────────────────────
    slot_range_usd = None
    try:
        klines_5m_dict, klines_5m_times = fetch_btc_klines_5m(limit=50)
        if slot in klines_5m_dict:
            k5 = klines_5m_dict[slot]
            rng_high = k5.get("high", btc_price)
            rng_low  = k5.get("low", btc_price)
            slot_range_usd = rng_high - rng_low
        else:
            # 估算:從當前5m bar估算
            if klines_5m_times:
                latest_5m_ts = klines_5m_times[-1]
                latest_5m = klines_5m_dict.get(latest_5m_ts, {})
                if latest_5m:
                    slot_range_usd = latest_5m.get("high", btc_price) - latest_5m.get("low", btc_price)
    except Exception as e:
        print(f"   ⚠️ 波幅計算失敗: {e}")

    # 波幅分級
    vol_mult = 1.0
    vol_class = "NORMAL"
    if slot_range_usd is not None:
        if slot_range_usd < 30:
            vol_mult = 0.5   # 橫行市:×0.5
            vol_class = "SIDEWAYS"
        elif slot_range_usd > 100:
            vol_mult = 0.8   # 大波動:×0.8
            vol_class = "BIG"
        else:
            vol_mult = 1.0
            vol_class = "NORMAL"
        print(f"   📊 波幅: ${slot_range_usd:.0f} ({vol_class}) → 倉位×{vol_mult}")

    # ── Step 2: CLOB 衝突已在上方處理 ──────────────────────────────────────
    # (CLOB override logic moved before volatility filter)

    # ── Step 3: 信心倍數 + 倉位計算 ────────────────────────────────────────
    conf_mult = 1.0
    # PRIMARY 信心來源: v46 (v46 is PRIMARY cascade since 2026-04-10)
    # Use v46_conf if available, else v42_conf
    primary_conf = v46_conf if v46_dir != "N/A" else (v42_conf if v42_dir != "N/A" else 0.5)
    primary_src = "v46" if v46_dir != "N/A" else "v42"

    if primary_conf >= 0.56:
        conf_mult = 1.2
        print(f"   ⭐ 高信心 {primary_src}={primary_conf:.2%} → 倉位×1.2")
    elif primary_conf < 0.52:
        conf_mult = 0.8   # 低信心 → ×0.8
        print(f"   ⚠️ 低信心 {primary_src}={primary_conf:.2%} → 倉位×0.8")
    else:
        print(f"   📊 信心 {primary_src}={primary_conf:.2%} → 倉位×1.0")

    # 組合倍數
    combined_mult = vol_mult * conf_mult
    print(f"   📐 倉位倍數: 波幅({vol_mult}) × 信心({conf_mult}) = {combined_mult:.2f}")

    # ── 決策記錄 dict（所有分支共用）──────────────────────────────────────────
    _dec_base = {
        "slot":         slot,
        "hour_utc":     hour_utc,
        "sim_mode":     sim_mode,
        "btc_price":    round(btc_price, 2),
        "final_dir":    final_dir or "WAIT",
        # ML 模型 (v46 PRIMARY + v42 CONFIRM)
        "v42_dir":      v42_dir,   "v42_conf":  round(v42_conf, 4),
        "v46_dir":      v46_dir,   "v46_conf":  round(v46_conf, 4),
        # legacy fields (set to N/A for DB compatibility)
        "v43_dir":      "N/A",     "v43_conf":  0.0,
        "v17_dir":      "N/A",     "v17_conf":  0.0,
        "v35_dir":      "N/A",     "v35_conf":  0.0,
        "v41_dir":      "N/A",     "v41_conf":  0.0,
        "v4_score":     round(v4_score, 4),
        "v45_dir":      "N/A",     "v45_conf":  0.0,
        # 技術指標
        "rsi":          round(rsi_val, 2),
        "macd":         round(macd_val, 6),
        "ema_cross":    round(ema_cross_val, 4),
        "stoch":        round(stoch, 2),
        "willr":        round(willr_val, 2),
        "bb_pos":       round(bb_pos, 4),
        "atr":          round(atr_val, 6),
        "mom_5m":       round(mom_5m, 4),
        "mom_1m":       round(mom_1m, 4),
        "mom_15m":      round(mom_15m, 4),
        "taker":        round(taker, 4) if taker else None,
        # 外部信號
        "fng_value":    fng_val,
        "fng_class":    fng_class,
        "ls_ratio":     round(ls_val, 4) if ls_val else None,
        "pm_up_price":  round(pm_up_price, 4),
        "pm_dir":       pm_dir,
        # 波幅 & 倉位
        "slot_range":   round(slot_range_usd, 2) if slot_range_usd else None,
        "vol_class":    vol_class,
        "vol_mult":     vol_mult,
        "conf_mult":    conf_mult,
        "combined_mult":combined_mult,
        # 多樣化
        "div_up":       div_up,
        "div_down":     div_down,
        "div_note":     div_note,
        # 其他
        "clob_override":clob_override,
        "note":         note,
    }

    # ── 交易執行 ─────────────────────────────────────────────────────────────
    if final_dir is None:
        log_trade(btc_price, "WAIT", abs(effective_score), "WAIT", 0,
                  slot=slot, v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note=note)
        try:
            log_decision({**_dec_base, "status": "WAIT", "bet_amount": 0, "entry_price": 0})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")
        tell_joke()
        return

    # 倉位計算(應用組合倍數)
    base_bet = max(MIN_BET_USDC, min(balance * BASE_BET_PCT, MAX_BET_USDC))
    bet_amount = round(base_bet * combined_mult, 2)
    bet_amount = max(MIN_BET_USDC, min(bet_amount, MAX_BET_USDC))  # 仍受MAX cap
    token  = market["up_token"] if final_dir == "UP" else market["down_token"]
    mkt_px = market["up_price"] if final_dir == "UP" else (1 - market["up_price"])

    # 嘗試从 order book 獲取更好價格
    try:
        ob = client.get_order_book(token)
        best_px = float(ob.asks[0].price) if (ob.asks and len(ob.asks) > 0) else mkt_px
    except:
        best_px = mkt_px

    best_px = min(max(best_px, 0.01), 0.99)
    cost    = bet_amount / best_px

    print(f"\n{'='*50}")
    print(f"   🎯 Direction: {final_dir}")
    print(f"   💰 Bet: ${bet_amount:.2f} (×{combined_mult:.2f} from base ${base_bet:.2f})")
    print(f"   📍 Price: {best_px:.4f}")
    print(f"{'='*50}")

    if PREDICT_ONLY:
        log_trade(btc_price, final_dir, v42_conf, "PREDICT", bet_amount,
                  slot=slot, v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note=note)
        try:
            log_decision({**_dec_base, "status": "PREDICT", "bet_amount": bet_amount, "entry_price": best_px})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")
        print(f"🔍 PREDICT ONLY - 未下單")
        return

    if sim_mode:
        log_trade(btc_price, final_dir, v42_conf, "SIM", bet_amount,
                  slot=slot, v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note=f"[SIM] {note}")
        try:
            log_decision({**_dec_base, "status": "SIM", "bet_amount": bet_amount, "entry_price": best_px,
                          "note": f"[SIM] {note}"})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")
        print(f"🔵 SIM 模式 - 預測: {final_dir} | 倉位: ${bet_amount:.2f} | 未下實盤")
        save_state(state)
        return

    # ── 真實下單 ─────────────────────────────────────────────────────────────
    order = OrderArgs(token_id=token, price=best_px, size=cost, side=BUY)
    opts  = PartialCreateOrderOptions(tick_size=market.get("tick_size", "0.01"))

    result = None
    for attempt in range(3):
        try:
            signed = client.create_order(order, opts)
            result = client.post_order(signed)
            break
        except Exception as e:
            print(f"⚠️ 下單重試 {attempt+1}/3: {e}")
            time.sleep(2)

    if result is None:
        print("❌ 下單失敗(3次重試)")
        log_trade(btc_price, final_dir, v42_conf, "FAIL", bet_amount,
                  slot=slot, v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note="ERROR: 3 retries failed")
        try:
            log_decision({**_dec_base, "status": "FAIL", "bet_amount": bet_amount,
                          "entry_price": best_px, "note": "ERROR: 3 retries failed"})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")
        return

    order_id = result.get("orderID", "")
    tx_hash  = (result.get("transactionsHashes") or [""])[0]

    if result.get("success"):
        print(f"✅ 下單成功! Order: {order_id}")
        print(f"   Tx: {tx_hash}")
        log_trade(btc_price, final_dir, v42_conf, "SUCCESS", bet_amount,
                  order_id=order_id, tx_hash=tx_hash, slot=slot,
                  v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note=note)
        try:
            log_decision({**_dec_base, "status": "SUCCESS", "bet_amount": bet_amount, "entry_price": best_px})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")
        state["last_slot"] = slot
        save_state(state)
        tell_joke()
    else:
        print(f"❌ 下單未成功: {result}")
        log_trade(btc_price, final_dir, v42_conf, "FAIL", bet_amount,
                  slot=slot, v4_score=v4_score, v17_dir="N/A",
                  v17_conf=v42_conf, v35_dir="N/A", note=f"ORDER_FAILED: {result}")
        try:
            log_decision({**_dec_base, "status": "FAIL", "bet_amount": bet_amount,
                          "entry_price": best_px, "note": f"ORDER_FAILED: {result}"})
        except Exception as e:
            print(f"   ⚠️ log_decision failed: {e}")

if __name__ == "__main__":
    run()
