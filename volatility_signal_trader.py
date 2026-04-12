#!/usr/bin/env python3
"""
Volatility-Signal Balanced Trader (VSB)
========================================
新紙質交易策略，基於橫向研究結論設計。

核心改變（相對於原5m系統）：
  1. 方向平衡：LONG + SHORT 都可以，不再只下單一方向
     - 5-agent 決定方向（不只是ML）
     - 允許兩個方向都有倉位
  2. 波幅確認：range_15m > $50 才交易，否則 WAIT
     - 橫行市（range < $30）不交易
     - 大波幅（range > $100）倉位 ×1.5
  3. 倉位動態：
     - base = balance × 10%
     - vol_mult:  <$50 → ×0.5 | $50-100 → ×1.0 | >$100 → ×1.5
     - conf_mult: conf≥60% → ×1.3 | conf<50% → ×0.7
     - min: $2 | max: $15
  4. HOLD 8%：虧損超過8%才持有，否則立即結算
  5. 無時段限制：任何時候都可能交易

5-agent 系統：
  Polymarket (w=3.0) + Binance (w=2.0) + Sentiment (w=2.0)
  + ML (w=1.5) + History (w=1.5)
  → 加權投票決定方向

15分鐘窗口（與MEXC 15m版同步）：
  - entry: 15分鐘開始的價格
  - exit:  15分鐘結束的價格
  - 對應 Polymarket BTC-5m 市場的3個連續slot
"""
import os, json, time, requests, asyncio
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(os.path.expanduser("~/.openclaw/workspace/skills/btc-5m-trader"))
LOG  = BASE / "logs" / "vsb_trades.jsonl"
HELD = BASE / "logs" / "vsb_held.json"
TRACK = BASE / "logs" / "vsb_tracked_slots.json"

BINANCE = "https://api.binance.com"
MEXC_GAMMA = "https://gamma-api.polymarket.com"

# ── 策略參數 ────────────────────────────────────────────────────────────────
SLOT_DURATION      = 900        # 15分鐘
MIN_ENTRY_SECS     = 120        # 至少2分鐘緩衝
LEVERAGE           = 3
BALANCE            = 100.0
POSITION_PCT       = 0.10       # 10% base
HOLD_LOSS_THRESH   = 0.08      # 8% → hold
PROFIT_THRESH      = 0.001     # 任何利潤 → exit
VOL_LOW            = 30.0      # <$30 → 不交易
VOL_MID            = 50.0      # >$50 → 允許交易
VOL_HIGH           = 100.0     # >$100 → 大波幅 ×1.5
CONF_HIGH          = 0.58      # ≥58% → conf ×1.3
CONF_LOW           = 0.50       # <50% → conf ×0.7

# 5-agent weights
WEIGHTS = {"polymarket":3.0,"binance":2.0,"sentiment":2.0,"ml":1.5,"history":1.5}

# ── 5-Agent Signal System ─────────────────────────────────────────────────
async def fetch_all_signals(slot_ts: int):
    """並行獲取所有5個Agent信號"""
    import sys; sys.path.insert(0, str(BASE))
    try:
        from btc5m_task.agents import CoordinatorAgent
        coord = CoordinatorAgent(timeout=25.0)
        return await coord.run(slot_ts=slot_ts)
    except Exception as e:
        print(f"  ⚠️ Coordinator error: {e}")
        return None

def calc_pnl(entry, exit_px, direction, size):
    if direction == "LONG":
        ret = (exit_px - entry) / entry
    else:
        ret = (entry - exit_px) / entry
    return size * LEVERAGE * ret, ret

# ── Data ───────────────────────────────────────────────────────────────────
def klines_1m(limit=100):
    try:
        r = requests.get(f"{BINANCE}/api/v3/klines",
                        params={"symbol":"BTCUSDT","interval":"1m","limit":limit}, timeout=8)
        return [[float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in r.json()]
    except: return []

def klines_15m(limit=60):
    try:
        r = requests.get(f"{BINANCE}/api/v3/klines",
                        params={"symbol":"BTCUSDT","interval":"15m","limit":limit}, timeout=8)
        return [[float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in r.json()]
    except: return []

def live_price():
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/price", params={"symbol":"BTCUSDT"}, timeout=8)
        return float(r.json()["price"])
    except: return None

def bid_ask():
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/bookTicker", params={"symbol":"BTCUSDT"}, timeout=8)
        d = r.json()
        return float(d["bidPrice"]), float(d["askPrice"])
    except: return None, None

def get_range_15m(k15):
    """15分鐘波幅"""
    if len(k15) < 2: return 50.0
    return k15[-1][2] - k15[-1][3]  # high - low of last 15m candle

# ── Position Sizing ─────────────────────────────────────────────────────────
def calc_size(balance, vol_range, conf):
    base = balance * POSITION_PCT
    # vol mult
    if vol_range < VOL_LOW:
        return 0.0  # 不交易
    elif vol_range < VOL_MID:
        vm = 0.5
    elif vol_range > VOL_HIGH:
        vm = 1.5
    else:
        vm = 1.0
    # conf mult
    if conf >= CONF_HIGH:
        cm = 1.3
    elif conf < CONF_LOW:
        cm = 0.7
    else:
        cm = 1.0
    size = base * vm * cm
    return max(2.0, min(15.0, size))

# ── Main ──────────────────────────────────────────────────────────────────
async def run():
    now = int(time.time())
    slot = (now // SLOT_DURATION) * SLOT_DURATION
    secs_left = (slot + SLOT_DURATION) - now

    print(f"\n{'='*55}")
    print(f"  VSB Trader — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Slot: {slot} | {secs_left}s left")
    print(f"{'='*55}")

    if secs_left < MIN_ENTRY_SECS:
        print(f"  ⏳ too close ({secs_left}s), skip")
        return

    # ── Data fetch (parallel) ────────────────────────────────────────────────
    k1 = klines_1m(100)
    k15 = klines_15m(60)
    bp, ap = bid_ask()
    cp = bp or live_price() or (k1[-1][4] if k1 else None)

    if not k1 or not k15 or not cp:
        print("  ❌ no data"); return

    closes_1m = [float(x[4]) for x in k1]
    closes_15m = [float(x[4]) for x in k15]

    range_15m = get_range_15m(k15)
    print(f"  BTC: ${cp:,.2f} | 15m range: ${range_15m:.2f}")

    # ── 5-Agent ──────────────────────────────────────────────────────────────
    print(f"  🔄 Running 5-agent system...")
    decision = await fetch_all_signals(slot)

    if decision is None:
        print("  ❌ Coordinator failed"); return

    d = decision.to_dict()
    sig = d.get("direction")   # UP / DOWN / None
    conf = d.get("confidence", 0.5)
    score = d.get("score", 0.0)
    meta  = d.get("metadata", {})

    print(f"  → Agent direction: {sig or 'WAIT'} | conf={conf:.0%} | score={score:+.2f}")
    print(f"    multipliers: vol={meta.get('vol_mult','?')}x conf={meta.get('conf_mult','?')}x")

    # ── Slot dedup ──────────────────────────────────────────────────────────
    tracked = set(json.load(open(TRACK)) if os.path.exists(TRACK) else [])
    if slot in tracked:
        print("  ⏳ already traded this slot")
        return

    # ── Held position ───────────────────────────────────────────────────────
    held = json.load(open(HELD)) if os.path.exists(HELD) else None
    entry_px = 0.0; exit_px = 0.0; pnl_val = 0.0
    position = "WAIT"; note = ""; new_held = None; closed_15m = False

    if held:
        held_dir = held["direction"]
        held_ent = held["entry_price"]
        held_sz  = held.get("position_size", 5.0)
        cur_pnl, cur_ret = calc_pnl(held_ent, cp, held_dir, held_sz)
        ret_pct = abs(cur_ret) * 100

        print(f"  📌 Held: {held_dir} @{held_ent:.2f} → {cp:.2f} | {cur_pnl:+.2f} ({ret_pct:.1f}%)")

        if cur_pnl > 0:
            exit_px = cp
            pnl_val, _ = calc_pnl(held_ent, exit_px, held_dir, held_sz)
            position = f"EXIT_{held_dir}"; note = f"HOLD→exit profit ${pnl_val:+.2f}"
            print(f"  ✅ EXIT {held_dir} profit ${pnl_val:+.2f}")
            new_held = None
        elif cur_ret <= -HOLD_LOSS_THRESH:
            exit_px = cp
            pnl_val, _ = calc_pnl(held_ent, exit_px, held_dir, held_sz)
            position = f"FORCE_CLOSE_{held_dir}"; note = f"loss={pnl_val:.2f}"
            print(f"  ❌ FORCE CLOSE {held_dir} loss ${pnl_val:.2f} ({ret_pct:.1f}%)")
            new_held = None
        else:
            position = f"HOLDING_{held_dir}"
            entry_px = held_ent
            new_held = {
                "direction": held_dir, "entry_price": held_ent,
                "slot": held["slot"], "position_size": held_sz,
                "opened_at": held.get("opened_at", datetime.now(timezone.utc).isoformat()),
                "hold_count": held.get("hold_count", 0) + 1,
                "worst_pnl": min(held.get("worst_pnl", 0), cur_pnl),
            }
            note = f"Holding {held_dir} {cur_pnl:+.2f} ({ret_pct:.1f}% < 8%)"
            print(f"  ⏸️ HOLD {held_dir} {cur_pnl:+.2f}")
    else:
        # ── New entry ───────────────────────────────────────────────────────
        if sig is None or sig == "WAIT":
            print(f"  ⏳ No direction signal (WAIT)")
        elif range_15m < VOL_LOW:
            print(f"  ⏳ Range ${range_15m:.2f} < ${VOL_LOW} (no trade in sideways)")
        else:
            entry_px = closes_1m[-1]
            position = "LONG" if sig == "UP" else "SHORT"

            # Calculate position size
            pos_size = calc_size(BALANCE, range_15m, conf)
            if pos_size < 2.0:
                print(f"  ⏳ pos_size ${pos_size:.2f} < $2 minimum, skip")
            else:
                # Get 15m candle close for this slot
                slot_end_ms = (slot + SLOT_DURATION) * 1000
                try:
                    r = requests.get(f"{BINANCE}/api/v3/klines",
                                   params={"symbol":"BTCUSDT","interval":"15m",
                                           "startTime":slot*1000,"endTime":slot_end_ms,"limit":1}, timeout=8)
                    cdl = r.json()
                    if cdl and len(cdl) > 0:
                        ct = int(cdl[0][6])
                        if ct < now * 1000:
                            exit_px = float(cdl[0][4])
                            closed_15m = True
                        else:
                            exit_px = cp
                    else:
                        exit_px = cp
                except:
                    exit_px = cp

                cur_pnl, cur_ret = calc_pnl(entry_px, exit_px, position, pos_size)
                ret_pct = abs(cur_ret) * 100

                if cur_pnl > 0:
                    pnl_val = cur_pnl
                    note = f"WIN {position} ${pnl_val:+.2f}"
                    print(f"  ✅ WIN {position} @ ${entry_px:.2f}→${exit_px:.2f} | ${pnl_val:+.2f} ({ret_pct:.1f}%)")
                    new_held = None
                elif cur_ret <= -HOLD_LOSS_THRESH:
                    pnl_val = cur_pnl
                    new_held = {
                        "direction": position, "entry_price": entry_px,
                        "slot": slot, "position_size": pos_size,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "hold_count": 0, "worst_pnl": pnl_val,
                    }
                    note = f"HOLD {position} loss=${pnl_val:.2f} ({ret_pct:.1f}%)"
                    print(f"  ⏸️ HOLD {position} loss=${pnl_val:.2f} ({ret_pct:.1f}%)")
                else:
                    pnl_val = cur_pnl
                    note = f"SETTLE {position} ${pnl_val:+.2f}"
                    print(f"  ⚖️ {position} @ ${entry_px:.2f}→${exit_px:.2f} | ${pnl_val:+.2f}")
                    new_held = None

                tracked.add(slot)
                json.dump(list(tracked), open(TRACK,"w"))

    # ── Save state ─────────────────────────────────────────────────────────
    if new_held:
        json.dump(new_held, open(HELD,"w"), default=str)
    elif os.path.exists(HELD):
        os.remove(HELD)

    # ── Log ─────────────────────────────────────────────────────────────────
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slot": slot,
        "price": cp, "bid": bp, "ask": ap,
        "direction": position,
        "agent_dir": sig,
        "agent_conf": round(conf, 4),
        "agent_score": round(score, 4),
        "range_15m": round(range_15m, 2),
        "position_size": new_held.get("position_size") if new_held else (BALANCE * POSITION_PCT),
        "entry_price": entry_px,
        "exit_price": exit_px,
        "closed_15m": closed_15m,
        "pnl": round(pnl_val, 4) if pnl_val else 0.0,
        "note": note,
        "agent_vote_log": meta.get("vote_log", []),
        "held": new_held,
    }
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG,"a") as f:
        f.write(json.dumps(log, default=str)+"\n")

    emoji = {"LONG":"🟢","SHORT":"🔴","WAIT":"⚪",
              "EXIT_LONG":"✅🟢","EXIT_SHORT":"✅🔴",
              "FORCE_CLOSE_LONG":"❌🟢","FORCE_CLOSE_SHORT":"❌🔴",
              "HOLDING_LONG":"⏸️🟢","HOLDING_SHORT":"⏸️🔴",
              "WIN_LONG":"✅🟢","WIN_SHORT":"✅🔴",
              "SETTLE_LONG":"⚖️🟢","SETTLE_SHORT":"⚖️🔴"}.get(position,"⚪")
    print(f"  {emoji} {position} @ ${cp:.2f} | PnL: {pnl_val:+.4f} | {note}")

def run_sync():
    asyncio.run(run())

if __name__ == "__main__":
    run_sync()
