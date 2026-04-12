#!/usr/bin/env python3
"""
BTC-5m 模擬交易系統 v1.1 - 修復版
"""
import requests
import json
import os
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 20.0
MAX_BET_PCT = 0.05
REVERSE_THRESHOLD = 0.65
MIN_CONFIDENCE = 0.50

BASE_DIR = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader")
WALLET_FILE = os.path.join(BASE_DIR, "wallet.json")
os.makedirs(os.path.dirname(WALLET_FILE), exist_ok=True)

class VirtualWallet:
    def __init__(self, initial=INITIAL_CAPITAL):
        self.initial = initial
        self.load()
    
    def load(self):
        if os.path.exists(WALLET_FILE):
            with open(WALLET_FILE, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', self.initial)
                self.history = data.get('history', [])
        else:
            self.balance = self.initial
            self.history = []
    
    def save(self):
        with open(WALLET_FILE, 'w') as f:
            json.dump({'balance': self.balance, 'initial': self.initial, 'history': self.history[-100:]}, f, indent=2)
    
    def get_max_bet(self):
        return self.balance * MAX_BET_PCT
    
    def can_bet(self, amount):
        return amount > 0 and amount <= self.get_max_bet() + 0.01  # 允許小誤差
    
    def place_bet(self, amount, direction, confidence, result):
        if not self.can_bet(amount):
            return False, "餘額不足"
        
        self.balance -= amount
        
        if result == 1:
            profit = amount
            self.balance += amount * 2
            result_desc = "贏"
        else:
            profit = -amount
            result_desc = "輸"
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'amount': amount,
            'confidence': confidence,
            'result': result_desc,
            'profit': profit,
            'balance': self.balance
        })
        self.save()
        return True, f"{result_desc} ${abs(profit):.2f}"
    
    def get_stats(self):
        if not self.history:
            return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit': 0, 'balance': self.balance}
        wins = sum(1 for r in self.history if r['profit'] > 0)
        losses = sum(1 for r in self.history if r['profit'] < 0)
        profit = sum(r['profit'] for r in self.history)
        return {'trades': len(self.history), 'wins': wins, 'losses': losses, 'win_rate': wins/len(self.history)*100, 'profit': profit, 'balance': self.balance}

def fetch_klines(limit=100):
    try:
        res = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": SYMBOL, "interval": "1m", "limit": limit}, timeout=10).json()
        return [float(k[4]) for k in res]
    except:
        return []

def fetch_orderbook():
    try:
        res = requests.get("https://api.binance.com/api/v3/depth", params={"symbol": SYMBOL, "limit": 20}).json()
        bids = sum(float(p)*float(q) for p, q in res['bids'])
        asks = sum(float(p)*float(q) for p, q in res['asks'])
        return (bids-asks)/(bids+asks) if (bids+asks) > 0 else 0
    except:
        return 0

def calc_rsi(closes, period=14):
    if len(closes) < period+1: return 50
    deltas = [closes[i]-closes[i-1] for i in range(1, len(closes))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_g = sum(gains[:period])/period
    avg_l = sum(losses[:period])/period
    for i in range(period, len(gains)):
        avg_g = (avg_g*(period-1)+gains[i])/period
        avg_l = (avg_l*(period-1)+losses[i])/period
    return 100 - (100/(1+avg_g/avg_l)) if avg_l>0 else 50

def calc_ema(closes, span):
    if len(closes) < span: return closes[-1] if closes else 0
    k = 2/(span+1)
    ema = closes[0]
    for p in closes[1:]: ema = p*k + ema*(1-k)
    return ema

def calc_macd(closes):
    if len(closes) < 26: return 0
    return calc_ema(closes, 12) - calc_ema(closes, 26) - calc_ema([calc_ema(closes, 12)-calc_ema(closes, 26)], 9)

def calc_ema_cross(closes):
    e9 = calc_ema(closes[-30:] if len(closes)>=30 else closes, 9)
    e21 = calc_ema(closes[-30:] if len(closes)>=30 else closes, 21)
    return 1 if e9 > e21 else -1

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
def init_model():
    np.random.seed(42)
    X = np.random.rand(200, 3)
    y = np.random.randint(0, 2, 200)
    model.fit(X, y)
init_model()

def ml_predict(rsi, macd_hist, ema_cross):
    return model.predict_proba(np.array([[rsi/100, macd_hist/10, ema_cross]]))[0][1]

def signal_filter(prob_up):
    if prob_up >= REVERSE_THRESHOLD:
        return "DOWN", 1-prob_up, True
    elif prob_up >= MIN_CONFIDENCE:
        return "UP", prob_up, False
    elif prob_up <= (1-MIN_CONFIDENCE):
        return "DOWN", 1-prob_up, False
    return "WAIT", 0.5, False

def run_simulation():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{now} HKT] BTC-5m 模擬交易 v1.1")
    
    wallet = VirtualWallet()
    stats = wallet.get_stats()
    
    print(f"\n💰 餘額: ${stats['balance']:.2f}")
    print(f"📊 交易: {stats['trades']} | 贏: {stats['wins']} | 輸: {stats['losses']}")
    print(f"📈 勝率: {stats['win_rate']:.1f}% | 損益: ${stats['profit']:.2f}")
    
    closes = fetch_klines()
    if not closes:
        print("❌ 無數據")
        return
    
    price = closes[-1]
    ob = fetch_orderbook()
    rsi = calc_rsi(closes)
    macd = calc_macd(closes)
    ema_cross = calc_ema_cross(closes)
    prob = ml_predict(rsi, macd, ema_cross)
    direction, confidence, reversed = signal_filter(prob)
    
    print(f"\n📊 市場:")
    print(f"   價格: ${price:.2f}")
    print(f"   RSI: {rsi:.1f} | MACD: {macd:.2f} | EMA: {'GOLDEN' if ema_cross>0 else 'DEATH'}")
    print(f"   OB: {ob:.3f} | ML: {prob:.1%}")
    
    # 下注
    if direction != "WAIT" and confidence >= MIN_CONFIDENCE:
        bet_size = round(wallet.get_max_bet(), 2)
        
        # 模擬結果
        result = 1 if np.random.rand() < confidence else 0
        
        success, msg = wallet.place_bet(bet_size, direction, confidence, result)
        
        if success:
            print(f"\n🎯 下注: {direction}" + (" (反轉)" if reversed else ""))
            print(f"   金額: ${bet_size:.2f}")
            print(f"   信心: {confidence:.1%}")
            print(f"   結果: {msg}")
        else:
            print(f"\n⚠️ 失敗: {msg}")
    else:
        print(f"\n⏭️ 跳過: {direction}, 信心 {confidence:.1%}")
    
    stats = wallet.get_stats()
    print(f"\n💰 餘額: ${stats['balance']:.2f} | 損益: ${stats['profit']:.2f}")

if __name__ == "__main__":
    run_simulation()
