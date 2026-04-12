#!/usr/bin/env python3
"""
Daily Self-Improvement for BTC-5m v7.0
Reads from real_trades_log.jsonl + settlement_cache.json
"""
import json, os, time, datetime

LOG_PATH      = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader/logs/real_trades_log.jsonl")
CACHE_PATH    = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader/data/settlement_cache.json")
STATE_PATH    = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader/data/state.json")
REVIEW_PATH   = os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader/logs/daily_review.md")
MEMORY_PATH   = os.path.expanduser("~/.openclaw/workspace/memory/auto-daily-review.md")

def load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except:
        return {}

def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except:
        return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def load_24h_trades():
    """Load successful trades from last 24h and cross-ref with settlement"""
    cache = load_cache()
    cutoff = time.time() - 86400
    trades = []
    
    try:
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    t = json.loads(line.strip())
                    # Parse ts (ISO string)
                    ts_str = t.get("ts", "")
                    if not ts_str:
                        continue
                    # Handle both ISO and unix formats
                    if isinstance(ts_str, (int, float)) or ts_str.replace(".", "").isdigit():
                        ts = float(ts_str)
                    else:
                        try:
                            dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            ts = dt.timestamp()
                        except:
                            continue
                    
                    if ts < cutoff:
                        continue
                    
                    if t.get("status") != "SUCCESS":
                        continue
                    
                    # Get actual outcome from settlement cache
                    slot = t.get("slot", 0)
                    actual = cache.get(str(slot))
                    
                    direction = t.get("direction", "")
                    won = (actual == direction) if actual else None
                    
                    trades.append({
                        "ts": ts,
                        "ts_str": ts_str,
                        "direction": direction,
                        "confidence": t.get("confidence", 0),
                        "slot": slot,
                        "actual": actual,
                        "won": won,
                        "indicators": t.get("indicators", {}),
                        "v3_features": t.get("v3_features", {}),
                        "traditional": t.get("traditional", {}),
                        "amount": t.get("amount", 0),
                    })
                except:
                    continue
    except:
        pass
    
    return trades

def compute_stats(trades):
    """Compute per-indicator and overall stats"""
    total = len(trades)
    won = sum(1 for t in trades if t.get("won") is True)
    lost = sum(1 for t in trades if t.get("won") is False)
    pending = sum(1 for t in trades if t.get("won") is None)
    acc = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0
    
    # Per-indicator stats (from v3 model features)
    v3_feats = {
        "imbalance": {"wins": 0, "total": 0},
        "up_count": {"wins": 0, "total": 0},
        "down_count": {"wins": 0, "total": 0},
        "total_amt": {"wins": 0, "total": 0},
    }
    for t in trades:
        if t.get("won") is None:
            continue
        d = t.get("v3_features", {})
        for k in v3_feats:
            v3_feats[k]["total"] += 1
            if t["won"]:
                v3_feats[k]["wins"] += 1
    
    # Traditional indicator stats
    ind_stats = {
        "rsi": {"wins": 0, "total": 0},
        "macd": {"wins": 0, "total": 0},
        "ema_cross": {"wins": 0, "total": 0},
        "vol_delta": {"wins": 0, "total": 0},
    }
    
    for t in trades:
        if t.get("won") is None:
            continue
        ind = t.get("indicators", {})
        w = t["won"]
        
        # RSI - track all, count wins separately
        rsi = ind.get("rsi", 50)
        if rsi < 40 or rsi > 60:
            ind_stats["rsi"]["total"] += 1
            if w:
                ind_stats["rsi"]["wins"] += 1
        
        # MACD
        macd = ind.get("macd", 0)
        ind_stats["macd"]["total"] += 1
        if w:
            ind_stats["macd"]["wins"] += 1
        
        # EMA
        ec = ind.get("ema_cross", 0)
        ind_stats["ema_cross"]["total"] += 1
        if w:
            ind_stats["ema_cross"]["wins"] += 1
        
        # Vol Delta
        vd = ind.get("vol_delta", 0)
        if abs(vd) > 0.1:
            ind_stats["vol_delta"]["total"] += 1
            if w:
                ind_stats["vol_delta"]["wins"] += 1
    
    return {
        "total": total, "won": won, "lost": lost, "pending": pending,
        "accuracy": acc,
        "v3_features": v3_feats,
        "indicators": ind_stats,
    }

def run_review():
    now = datetime.datetime.now()
    now_hkt = now.strftime("%Y-%m-%d %H:%M:%S HKT")
    
    state = load_state()
    trades = load_24h_trades()
    stats = compute_stats(trades)
    
    print(f"\n{'='*60}")
    print(f"🔄 DAILY SELF-IMPROVEMENT — {now_hkt}")
    print(f"📊 24h Stats: {stats['total']} trades | {stats['won']}W/{stats['lost']}L | {stats['accuracy']}% accuracy")
    
    if stats['total'] == 0:
        print("⚠️ No trades in last 24h. Skipping.")
        return None
    
    # v3 features analysis
    print(f"\n📐 v3 ML Features:")
    v3 = stats.get("v3_features", {})
    for k, s in v3.items():
        if s.get("total", 0) > 0:
            acc = s["wins"] / s["total"] * 100
            print(f"  {k}: {s['wins']}/{s['total']} = {acc:.0f}%")
    
    # Indicator analysis
    print(f"\n📐 Indicators:")
    inds = stats.get("indicators", {})
    for k, s in inds.items():
        if s.get("total", 0) > 0:
            acc = s["wins"] / s["total"] * 100
            print(f"  {k}: {s['wins']}/{s['total']} = {acc:.0f}%")
    
    # Update state
    state["last_review"] = now_hkt
    state["last_24h_stats"] = {
        "total": stats["total"],
        "won": stats["won"],
        "accuracy": stats["accuracy"],
    }
    save_state(state)
    
    # Write review
    review = f"""# Daily Review — {now_hkt}

## Performance (24h)
- Total Trades: {stats['total']}
- Wins: {stats['won']} | Losses: {stats['lost']} | Pending: {stats['pending']}
- Accuracy: {stats['accuracy']}%

## v3 ML Features Performance
"""
    for k, s in v3.items():
        if s.get("total", 0) > 0:
            acc = s["wins"] / s["total"] * 100
            review += f"- {k}: {s['wins']}/{s['total']} = {acc:.0f}%\n"
    
    review += f"""
## Recent Trades (last 10)
"""
    for t in trades[-10:]:
        review += f"- {t['ts_str'][11:19]} | {t['direction']} | conf={t['confidence']:.1%} | actual={t['actual'] or '?'} | won={t['won']}\n"
    
    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    with open(REVIEW_PATH, "w") as f:
        f.write(review)
    
    # Update memory
    memory = f"""# Auto Daily Review — {now_hkt}
## BTC-5m v7.0 Stats
- Trades: {stats['total']} | Accuracy: {stats['accuracy']}%
"""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        f.write(memory)
    
    print(f"\n✅ Review complete. {stats['total']} trades analyzed.")
    return stats

if __name__ == "__main__":
    run_review()
