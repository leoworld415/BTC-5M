#!/usr/bin/env python3
"""
Polymarket BTC-5m Research Collector
每小時抓取並存儲 BTC-5min 市場結果到 SQLite
Slug format: btc-updown-5m-{slot_timestamp}
"""
import os, sys, json, time, sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Quiet mode for wrapper/cron — suppress all progress output
if os.getenv("REPORT_QUIET"):
    _print = print
    def print(*a, **kw): pass

SKILL_DIR = Path(__file__).parent
DATA_DIR   = SKILL_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH    = DATA_DIR / "btc5m_research.db"

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
HEADERS   = {"Content-Type": "application/json"}

def fetch_market(slot_ts):
    """Fetch single BTC-5m market by slot timestamp"""
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"slug": slug},
            headers=HEADERS,
            timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        pass
    return None

def parse_slot_from_market(market):
    """Extract 5-min slot timestamp from market question"""
    # e.g. "Bitcoin Up or Down - March 24, 9:25AM-9:30AM ET"
    q = market.get("question", "")
    # Parse the timestamp from slug instead
    slug = market.get("slug", "")
    if "btc-updown-5m-" in slug:
        try:
            return int(slug.split("-")[-1])
        except:
            pass
    return None

def slot_resolution(market):
    """
    Determine UP or DOWN from market resolution.
    Returns: 'UP', 'DOWN', 'MARKET_ERROR', or None (unsettled)
    """
    if not market.get("closed"):
        return None  # not settled yet
    
    res = market.get("resolution", "").strip().upper()
    
    # Direct resolution
    if res in ("YES", "UP", "BUY", "1", "TRUE"):
        return "UP"
    if res in ("NO", "DOWN", "SELL", "0", "FALSE"):
        return "DOWN"
    
    # Try outcome prices — higher price = the winner
    try:
        prices = json.loads(market.get("outcomePrices", "[]"))
        if len(prices) >= 2:
            # outcome[0] corresponds to "Up" typically, but check outcomes array
            outcomes = market.get("outcomes", [])
            up_idx = None
            for i, o in enumerate(outcomes):
                if "up" in o.lower():
                    up_idx = i
                    break
            if up_idx is not None and float(prices[up_idx]) >= 0.5:
                return "UP"
            elif up_idx is not None and float(prices[up_idx]) < 0.5:
                return "DOWN"
            # Fallback: higher price = winner
            if float(prices[0]) > float(prices[1]):
                return "UP"
            else:
                return "DOWN"
    except:
        pass
    
    return "MARKET_ERROR"

# ── SQLite ───────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS slots (
            slot_ts       INTEGER PRIMARY KEY,
            question      TEXT,
            closed        INTEGER,
            resolution    TEXT,
            outcome_prices TEXT,
            outcomes      TEXT,
            volume        REAL,
            up_price      REAL,
            down_price    REAL,
            winner        TEXT,
            collected_at  TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at  TEXT,
            slots_checked INTEGER,
            new_slots     INTEGER,
            settled       INTEGER
        )
    """)
    conn.commit()
    return conn

def slot_exists(conn, slot_ts):
    c = conn.cursor()
    c.execute("SELECT 1 FROM slots WHERE slot_ts = ?", (slot_ts,))
    return c.fetchone() is not None

def get_settled_count(conn):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM slots WHERE winner IN ('UP', 'DOWN')")
    return c.fetchone()[0]

def save_slot(conn, slot_ts, market, winner):
    try:
        prices = json.loads(market.get("outcomePrices", "[]"))
        up_price = float(prices[0]) if len(prices) > 0 else None
        down_price = float(prices[1]) if len(prices) > 1 else None
    except:
        up_price = down_price = None

    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO slots
        (slot_ts, question, closed, resolution, outcome_prices, outcomes, volume,
         up_price, down_price, winner, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        slot_ts,
        market.get("question"),
        1 if market.get("closed") else 0,
        market.get("resolution"),
        market.get("outcomePrices"),
        json.dumps(market.get("outcomes", [])),
        market.get("volume"),
        up_price,
        down_price,
        winner,
        datetime.now(timezone.utc).isoformat()
    ))

def get_stats(conn):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM slots")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM slots WHERE winner IN ('UP', 'DOWN')")
    settled = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM slots WHERE winner = 'UP'")
    up_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM slots WHERE winner = 'DOWN'")
    down_count = c.fetchone()[0]
    # Latest slot
    c.execute("SELECT MAX(slot_ts) FROM slots")
    latest = c.fetchone()[0]
    return {
        "total": total, "settled": settled,
        "up": up_count, "down": down_count,
        "latest_slot": latest
    }

# ── Research ─────────────────────────────────────────────────────────────────
def runResearch(start_slot, end_slot, conn):
    """Fetch range of slots, return (checked, new, settled) counts"""
    checked = 0
    new_count = 0
    settled = 0
    
    # Iterate backwards from end_slot to start_slot
    slot = end_slot
    commit_every = 50
    while slot >= start_slot:
        checked += 1
        
        if not slot_exists(conn, slot):
            market = fetch_market(slot)
            if market:
                winner = slot_resolution(market)
                if winner in ("UP", "DOWN"):
                    new_count += 1
                    settled += 1
                elif winner is None:
                    # Not settled yet, but still save record
                    new_count += 1
                else:
                    # MARKET_ERROR or other
                    new_count += 1
                
                save_slot(conn, slot, market, winner)
                if new_count % 50 == 0:
                    conn.commit()  # periodic commit to prevent data loss
            time.sleep(0.3)  # rate limit
        
        slot -= 300  # go to previous 5-min slot
        
        if checked % 100 == 0:
            print(f"  Checked {checked} slots... (new={new_count}, settled={settled})")
    
    return checked, new_count, settled

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_ts = int(time.time())
    current_slot = (now_ts // 300) * 300  # round down to 5 min
    
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BTC-5m Research Collector")
    print(f"{'='*60}")
    
    conn = init_db()
    stats = get_stats(conn)
    print(f"📊 Current DB: {stats['total']} slots, {stats['settled']} settled")
    if stats['latest_slot']:
        latest_dt = datetime.fromtimestamp(stats['latest_slot'], tz=timezone.utc)
        print(f"   Latest slot: {stats['latest_slot']} ({latest_dt.strftime('%Y-%m-%d %H:%M UTC')})")
    
    # How many slots to check?
    # Default: go back 500 slots (~1.7 days), from last checked to now
    # Determine start/end range
    if stats['latest_slot']:
        # Incremental: from last checked to now (500 slots back)
        LAST_CHECKED = stats['latest_slot']
        start_slot = max(LAST_CHECKED - 500*300, current_slot - 600*300)
        end_slot = current_slot - 300
    else:
        # First run: backfill 500 slots (~1.7 days)
        LAST_CHECKED = current_slot - 600*300
        start_slot = LAST_CHECKED
        end_slot = current_slot - 300
    
    if end_slot <= LAST_CHECKED and stats['latest_slot'] is not None:
        print(f"✅ No new slots to check (up to date)")
        conn.close()
        return
    
    print(f"\n📡 Checking slots {start_slot} → {end_slot}")
    print(f"   (~{(end_slot - start_slot)//300 + 1} slots)")
    
    checked, new, settled = runResearch(start_slot, end_slot, conn)
    
    # Log
    c = conn.cursor()
    c.execute("""
        INSERT INTO collection_log (collected_at, slots_checked, new_slots, settled)
        VALUES (?, ?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), checked, new, settled))
    conn.commit()
    
    stats = get_stats(conn)
    conn.close()
    
    print(f"\n✅ Run complete:")
    print(f"   Checked: {checked} slots")
    print(f"   New: +{new} records, +{settled} settled")
    print(f"   DB now: {stats['total'] + new} total slots")

if __name__ == "__main__":
    main()
