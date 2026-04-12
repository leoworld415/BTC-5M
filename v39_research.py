#!/usr/bin/env python3
"""
v39 Research Collector — v39專屬預測數據收集
每5分鐘預測並記錄，不下單，純收集訓練數據

用法: .venv/bin/python v39_research.py
"""
import os, sys, json, time, sqlite3, requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "btc5m_research.db"

# ── 安靜模式（cron時抑制輸出） ──────────────────────────────────────────
if os.getenv("V39_QUIET"):
    def print(*a, **kw): pass

# ── 市場數據 ───────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
HEADERS = {"Content-Type": "application/json"}

def fetch_current_slot_from_binance():
    """從 Binance 5m klines 取得當前 slot_ts（上一根完成的 5m bar，轉換成秒）"""
    klines_dict, times = fetch_btc_klines_5m(limit=10)
    if not times:
        return None
    # Binance 返回毫秒，轉換為秒
    return times[-1] // 1000

# ── v39 特徵構建（從 build_v39_features） ──────────────────────────────
def _ema_calc(data, n):
    if len(data) < n: return None
    k = 2.0 / (n + 1)
    ema = sum(data[:n]) / n
    for v in data[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _rsi_calc(data, n):
    if len(data) < n + 1: return 50.0
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gain = sum(d for d in deltas[-n:] if d > 0) / n
    loss = sum(-d for d in deltas[-n:] if d < 0) / n
    if loss == 0: return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _stoch(highs, lows, closes, n=14):
    if len(closes) < n: return 50.0
    lowest = min(lows[-n:])
    highest = max(highs[-n:])
    if highest == lowest: return 50.0
    return (closes[-1] - lowest) / (highest - lowest) * 100

def fetch_btc_klines_5m(limit=150):
    """Fetch Binance BTCUSDT 5m klines"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=6)
        if r.status_code != 200: return None, []
        raw = r.json()
        klines_dict = {}
        times = []
        for k in raw:
            ts = int(k[0])
            times.append(ts)
            klines_dict[ts] = {
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]), "quote_vol": float(k[7]),
                "taker_buy_vol": float(k[9])
            }
        return klines_dict, sorted(times)
    except Exception as e:
        print(f"⚠️ Binance fetch error: {e}")
        return None, []

def build_v39_features(klines_5m, slot_ts):
    """
    Compute 43 v39 features from 5m klines. NO LEAKAGE.
    Same logic as real_trader_newv1.py line 794+.
    """
    if len(klines_5m) < 30: return None
    # Find bar indexed at slot_ts
    # k[0] is in milliseconds (Binance), slot_ts is in seconds
    slot_ts_ms = slot_ts * 1000
    idx = None
    for i, k in enumerate(klines_5m):
        bar_end = k[0] + 300000  # bar end time in ms
        if bar_end <= slot_ts_ms:
            idx = i
    if idx is None or idx < 20: return None
    lookback = klines_5m[max(0, idx-150):idx+1]
    if len(lookback) < 20: return None
    closes  = [k[4] for k in lookback]
    highs   = [k[2] for k in lookback]
    lows    = [k[3] for k in lookback]
    volumes = [k[5] for k in lookback]
    taker_b = [k[7] for k in lookback]
    quotes  = [k[6] for k in lookback]
    price = closes[-1]
    c_feats = closes[:-1]
    h_feats = highs[:-1]
    l_feats = lows[:-1]
    v_feats = volumes[:-1]
    tb_feats = taker_b[:-1]
    q_feats = quotes[:-1]
    if len(c_feats) < 20: return None

    # EMA features
    ema5  = _ema_calc(c_feats, 5)
    ema9  = _ema_calc(c_feats, 9)
    ema12 = _ema_calc(c_feats, 12)
    ema20 = _ema_calc(c_feats, 20)
    ema50 = _ema_calc(c_feats, 50)
    ema_cross_raw   = (ema5 - ema20) / price if ema20 else 0.0
    ema_cross_9_20 = (ema9  - ema20) / price if ema20 else 0.0
    price_vs_ema5   = (price - ema5)  / price if ema5  else 0.0
    price_vs_ema20  = (price - ema20) / price if ema20 else 0.0
    price_vs_ema50  = (price - ema50) / price if ema50 else 0.0
    ema5_n  = ema5  / price if ema5  else 1.0
    ema20_n = ema20 / price if ema20 else 1.0

    # RSI
    rsi5_v  = _rsi_calc(c_feats, 5)
    rsi14_v = _rsi_calc(c_feats, 14)
    rsi5_n  = rsi5_v  / 100.0 if rsi5_v  else 0.5
    rsi14_n = rsi14_v / 100.0 if rsi14_v else 0.5
    rsi5_m14 = (rsi5_v - rsi14_v) / 100.0 if (rsi5_v and rsi14_v) else 0.0
    rsi_extreme = (1 if rsi14_v < 30 else 0) + (1 if rsi14_v > 70 else 0)

    # MACD
    ema26_v = _ema_calc(c_feats, 26)
    macd_n = (ema12 - ema26_v) / price if (ema12 and ema26_v) else 0.0

    # Bollinger Bands
    mean_p = sum(c_feats[-20:]) / 20
    std_p = (sum((v - mean_p)**2 for v in c_feats[-20:]) / 20) ** 0.5
    bb_pos = (price - (mean_p - 2*std_p)) / (4*std_p + 1e-9) if std_p > 0 else 0.5
    bb_pos = max(0, min(1, bb_pos))

    # ATR
    trs = [max(h_feats[i]-l_feats[i], abs(h_feats[i]-c_feats[i-1]), abs(l_feats[i]-c_feats[i-1])) for i in range(1,len(c_feats))]
    atr = sum(trs[-14:])/14 if len(trs) >= 14 else sum(trs)/len(trs) if trs else 0
    atr_n = atr / price if price else 0.0

    # Stochastic
    sk = _stoch(h_feats[-14:], l_feats[-14:], c_feats[-14:])
    sk_n = sk / 100.0
    sd = _stoch(h_feats[-15:-1], l_feats[-15:-1], c_feats[-15:-1])
    sd_n = sd / 100.0
    sk_m_sd = (sk - sd) / 100.0

    # Williams %R
    highest14 = max(h_feats[-14:])
    lowest14  = min(l_feats[-14:])
    willr = -100.0*(highest14 - price)/(highest14 - lowest14 + 1e-9) if highest14 != lowest14 else -50.0
    willr_n = willr / 100.0 + 0.5

    # Momentum
    mom1  = (c_feats[-1]  - c_feats[-2])  / c_feats[-2]  * 100 if len(c_feats) >= 2  else 0.0
    mom5  = (c_feats[-1]  - c_feats[-5])  / c_feats[-5]  * 100 if len(c_feats) >= 5  else 0.0
    mom10 = (c_feats[-1]  - c_feats[-10]) / c_feats[-10] * 100 if len(c_feats) >= 10 else 0.0
    mom15 = (c_feats[-1]  - c_feats[-15]) / c_feats[-15] * 100 if len(c_feats) >= 15 else 0.0
    mom1_n  = mom1  / 10.0
    mom5_n  = mom5  / 10.0
    mom10_n = mom10 / 10.0

    # Volume
    vol_sma20 = sum(v_feats[-20:])/20 if len(v_feats) >= 20 else sum(v_feats)/max(len(v_feats),1)
    vol_delta  = (v_feats[-1] - vol_sma20) / vol_sma20 if vol_sma20 > 0 else 0.0
    vol_drifting_up   = 1 if mom5 > 0 and vol_delta > 0 else 0
    vol_drifting_down = 1 if mom5 < 0 and vol_delta < 0 else 0

    # Taker buy ratio
    last_tb = tb_feats[-1] if tb_feats else 0
    last_v  = v_feats[-1]  if v_feats  else 1
    taker_ratio = (last_tb / last_v - 0.5) * 2 if last_v > 0 else 0.0
    tb_sma20 = sum(tb_feats[-20:])/20 if len(tb_feats) >= 20 else sum(tb_feats)/max(len(tb_feats),1)
    taker_avg = (tb_sma20 / (sum(v_feats[-20:])/20 + 1e-9) - 0.5) * 2 if len(v_feats) >= 20 else 0.0
    taker_n   = taker_ratio / 2.0 + 0.5
    taker_avg_n = taker_avg / 2.0 + 0.5
    obi_taker  = taker_ratio  # same signal
    obi_ibi = 0.0  # IFI not available without order book

    # VWAP
    cum_q = sum(q_feats)
    cum_vv = sum(v_feats)
    vwap60_n = (cum_q / cum_vv if cum_vv > 0 else price) / price - 1.0 if price else 0.0

    # Range position
    daily_range = highs[-1] - lows[-1]
    range_pos = (price - lows[-1]) / daily_range if daily_range > 0 else 0.5
    range_pos_n = range_pos

    # Consecutive bars
    consec_up   = sum(1 for i in range(1, min(6,len(c_feats))) if c_feats[-i] > c_feats[-i-1])
    consec_down = sum(1 for i in range(1, min(6,len(c_feats))) if c_feats[-i] < c_feats[-i-1])

    # Hour (from slot_ts)
    from datetime import datetime as dt
    hour = dt.fromtimestamp(slot_ts, tz=timezone.utc).hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    features = {
        "hour_cos": hour_cos,
        "hour_sin": hour_sin,
        "rsi14_n": rsi14_n,
        "rsi5_n": rsi5_n,
        "rsi5_m14": rsi5_m14,
        "rsi_extreme": rsi_extreme,
        "macd_n": macd_n,
        "ema_cross_raw": ema_cross_raw,
        "ema_cross_9_20": ema_cross_9_20,
        "price_vs_ema5_n": price_vs_ema5,
        "price_vs_ema20_n": price_vs_ema20,
        "price_vs_ema50_n": price_vs_ema50,
        "bb_pos_n": bb_pos,
        "atr_n": atr_n,
        "stoch_k_n": sk_n,
        "stoch_d_n": sd_n,
        "sk_m_sd_n": sk_m_sd,
        "willr_n": willr_n,
        "mom1_n": mom1_n,
        "mom5_n": mom5_n,
        "mom10_n": mom10_n,
        "mom15_n": mom15 / 10.0,
        "vol_delta_n": vol_delta,
        "vol_drifting_up": vol_drifting_up,
        "vol_drifting_down": vol_drifting_down,
        "taker_n": taker_n,
        "taker_avg_n": taker_avg_n,
        "obi_taker": obi_taker,
        "range_pos_n": range_pos_n,
        "consec_up": consec_up / 5.0,
        "consec_down": consec_down / 5.0,
        "price_slope20_n": (c_feats[-1] - c_feats[-20]) / c_feats[-20] / 20.0 if len(c_feats) >= 20 else 0.0,
        "ema5_n": ema5_n,
        "ema20_n": ema20_n,
    }
    return features

# ── v39 模型載入與預測 ──────────────────────────────────────────────────
import pickle
MODEL_PATH_V39 = DATA_DIR / "ml_model_v39.pkl"
SCALER_PATH_V39 = DATA_DIR / "scaler_v39.pkl"
FEAT_PATH_V39 = DATA_DIR / "ml_features_v39.json"

_mv39 = _sv39 = None
def load_v39():
    global _mv39
    if _mv39 is None and MODEL_PATH_V39.exists():
        with open(MODEL_PATH_V39, "rb") as f:
            _mv39 = pickle.load(f)
    return _mv39
def load_scaler_v39():
    global _sv39
    if _sv39 is None and SCALER_PATH_V39.exists():
        with open(SCALER_PATH_V39, "rb") as f:
            _sv39 = pickle.load(f)
    return _sv39

def v39_predict(features):
    """Return (p_down, p_up, direction, confidence, model_name)"""
    mv39 = load_v39(); sc = load_scaler_v39()
    if mv39 is None or sc is None:
        return 0.5, 0.5, "WAIT", 0.5, "v39_no_model"
    if not FEAT_PATH_V39.exists():
        return 0.5, 0.5, "WAIT", 0.5, "v39_no_feats"
    with open(FEAT_PATH_V39) as f:
        feat_names = json.load(f)
    X = np.array([[features.get(fn, 0.0) for fn in feat_names]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = sc.transform(X)
    p = mv39.predict_proba(X)[0]
    d = "UP" if p[1] >= p[0] else "DOWN"
    c = float(max(p[1], p[0]))
    return float(p[0]), float(p[1]), d, c, "v39"

# ── Fear & Greed + L/S Ratio ──────────────────────────────────────────
def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=4)
        if r.status_code == 200:
            d = r.json().get("data", [{}])[0]
            return int(d.get("value", 50)), d.get("value_classification","")
    except: pass
    return None, None

def fetch_ls_ratio():
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "5m", "limit": 5},
            timeout=4
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                last = float(data[-1].get("longShortRatio", 1.0))
                return last
    except: pass
    return None

# ── SQLite ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 10000")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS v39_research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            slot_ts INTEGER,
            hour_utc INTEGER,
            rsi REAL,
            macd_n REAL,
            ema_cross_raw REAL,
            vol_delta REAL,
            stoch_k REAL,
            willr REAL,
            obi_taker REAL,
            taker_ratio REAL,
            range_pos REAL,
            consec_up INTEGER,
            consec_down INTEGER,
            fng_value INTEGER,
            fng_class TEXT,
            ls_ratio REAL,
            hour_mult REAL,
            v39_p_down REAL,
            v39_p_up REAL,
            v39_dir TEXT,
            v39_conf REAL,
            v39_model TEXT,
            -- 43 raw features JSON
            features_json TEXT,
            -- outcome filled later
            actual_outcome TEXT,
            settled_at TEXT
        )
    """)
    conn.commit()
    return conn

# HOUR_MULT lookup（從 real_trader_newv1.py）
HOUR_BET_MULT = {
    0:0.50, 1:1.00, 2:0.50, 3:0.50, 4:1.00,
    5:0.50, 6:0.00, 7:0.50, 8:0.00, 9:0.00,
    10:1.00, 11:0.75, 12:0.50, 13:0.50, 14:0.75,
    15:0.75, 16:0.75, 17:0.00, 18:0.50, 19:0.50,
    20:0.50, 21:0.50, 22:0.50, 23:0.50
}

def main():
    print(f"[v39 Research] {datetime.now().strftime('%H:%M:%S')} 開始收集...")
    conn = init_db()
    cur = conn.cursor()

    # slot_ts = 上一根完成的 Binance 5m bar
    slot_ts = fetch_current_slot_from_binance()
    if not slot_ts:
        print("⚠️  無法取得 Binance slot")
        conn.close()
        return

    hour_utc = datetime.fromtimestamp(slot_ts, tz=timezone.utc).hour

    # Already collected?
    cur.execute("SELECT id FROM v39_research WHERE slot_ts=? LIMIT 1", (slot_ts,))
    if cur.fetchone():
        print(f"⏭️  slot {slot_ts} 已收集，跳過")
        conn.close()
        return

    # Build features
    klines_dict, times = fetch_btc_klines_5m(limit=150)
    if not klines_dict or len(times) < 30:
        print("⚠️  Binance klines 不足")
        conn.close()
        return

    klines_5m = [[t, klines_dict[t]['open'], klines_dict[t]['high'],
                   klines_dict[t]['low'], klines_dict[t]['close'],
                   klines_dict[t]['volume'], klines_dict[t]['quote_vol'],
                   klines_dict[t]['taker_buy_vol']] for t in times]

    features = build_v39_features(klines_5m, slot_ts)
    if features is None:
        print("⚠️  v39 features 計算失敗")
        conn.close()
        return

    # Market conditions
    closes = [klines_dict[t]['close'] for t in times]
    price = closes[-1]
    rsi_v  = _rsi_calc(closes[:-1], 14)
    ema12  = _ema_calc(closes[:-1], 12)
    ema26  = _ema_calc(closes[:-1], 26)
    macd_n = (ema12 - ema26) / price if (ema12 and ema26) else 0.0
    highs  = [klines_dict[t]['high']  for t in times]
    lows   = [klines_dict[t]['low']   for t in times]
    vols   = [klines_dict[t]['volume'] for t in times]
    sk     = _stoch(highs[-15:-1], lows[-15:-1], closes[-15:-1])
    highest14 = max(highs[-14:])
    lowest14  = min(lows[-14:])
    willr = -100.0*(highest14 - price)/(highest14 - lowest14 + 1e-9) if highest14 != lowest14 else -50.0
    tb = [klines_dict[t]['taker_buy_vol'] for t in times]
    taker_ratio = (tb[-1] / vols[-1] - 0.5) * 2 if vols[-1] > 0 else 0.0
    daily_range = highs[-1] - lows[-1]
    range_pos = (price - lows[-1]) / daily_range if daily_range > 0 else 0.5
    vol_sma20 = sum(vols[-20:])/20
    vol_delta = (vols[-1] - vol_sma20) / vol_sma20 if vol_sma20 > 0 else 0.0
    consec_up = sum(1 for i in range(1,6) if len(closes) > i and closes[-i] > closes[-i-1])
    consec_down = sum(1 for i in range(1,6) if len(closes) > i and closes[-i] < closes[-i-1])

    fng_val, fng_class = fetch_fng()
    ls_ratio = fetch_ls_ratio()
    hour_mult = HOUR_BET_MULT.get(hour_utc, 0.5)

    # v39 prediction
    pd, pu, d, c, model = v39_predict(features)

    # Insert
    cur.execute("""
        INSERT INTO v39_research
        (collected_at, slot_ts, hour_utc, rsi, macd_n, ema_cross_raw,
         vol_delta, stoch_k, willr, obi_taker, taker_ratio, range_pos,
         consec_up, consec_down, fng_value, fng_class, ls_ratio, hour_mult,
         v39_p_down, v39_p_up, v39_dir, v39_conf, v39_model, features_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        slot_ts, hour_utc,
        rsi_v, macd_n, features.get('ema_cross_raw', 0),
        vol_delta, sk/100.0, willr/100.0 + 0.5, features.get('obi_taker',0),
        taker_ratio, range_pos,
        consec_up, consec_down,
        fng_val, fng_class, ls_ratio, hour_mult,
        pd, pu, d, c, model,
        json.dumps(features)
    ))
    conn.commit()

    # ── 實時更新 actual_outcome（修覆DB落後問題）────────────
    cur.execute("""
        UPDATE v39_research
        SET actual_outcome = (
            SELECT s.winner FROM slots s
            WHERE s.slot_ts = v39_research.slot_ts AND s.winner IS NOT NULL
        )
        WHERE actual_outcome IS NULL
        AND EXISTS (
            SELECT 1 FROM slots s2
            WHERE s2.slot_ts = v39_research.slot_ts AND s2.winner IS NOT NULL
        )
    """)
    updated = cur.rowcount
    conn.commit()

    conn.close()
    print(f"✅ slot={slot_ts} [{datetime.now(timezone.utc).strftime('%H:%M')}] v39={d} @{c:.1%} fng={fng_val} ls={ls_ratio} h_utc={hour_utc}(×{hour_mult}){f' | matched {updated} settlements' if updated else ''}")

if __name__ == "__main__":
    main()
