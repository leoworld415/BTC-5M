#!/usr/bin/env python3
"""
BTC-5m Hourly Research Analyzer
每小時運行：抓取最新數據 + 深度分析 + 生成洞察報告
"""
import os, sys, json, time, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Quiet mode for wrapper/cron
if os.getenv("REPORT_QUIET"):
    _print = print
    def print(*a, **kw): pass
from collections import defaultdict
import statistics

SKILL_DIR = Path(__file__).parent
DATA_DIR   = SKILL_DIR / "data"
DB_PATH    = DATA_DIR / "btc5m_research.db"
REPORT_DIR = SKILL_DIR / "research_reports"
REPORT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SKILL_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# DB Queries
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)

def query_recent_slots(conn, hours=48):
    """Get slots from the last N hours"""
    cutoff = int(time.time()) - hours * 3600
    c = conn.cursor()
    c.execute("""
        SELECT slot_ts, question, closed, resolution, outcome_prices,
               up_price, down_price, winner
        FROM slots
        WHERE slot_ts > ?
        ORDER BY slot_ts DESC
    """, (cutoff,))
    return c.fetchall()

def query_all_settled(conn):
    c = conn.cursor()
    c.execute("""
        SELECT slot_ts, up_price, down_price, winner
        FROM slots
        WHERE winner IN ('UP', 'DOWN')
        ORDER BY slot_ts DESC
    """)
    return c.fetchall()

def query_by_hour(conn):
    """Win rate by UTC hour"""
    c = conn.cursor()
    c.execute("""
        SELECT
            (slot_ts / 300) % 24 as hour_utc,
            winner,
            COUNT(*) as count
        FROM slots
        WHERE winner IN ('UP', 'DOWN')
        GROUP BY hour_utc, winner
    """)
    rows = c.fetchall()
    
    hour_stats = defaultdict(lambda: {"up": 0, "down": 0})
    for hour, winner, count in rows:
        if winner == "UP":
            hour_stats[hour]["up"] = count
        else:
            hour_stats[hour]["down"] = count
    return hour_stats

def query_streaks(conn):
    """Analyze consecutive UP/DOWN streaks"""
    slots = query_all_settled(conn)
    if not slots:
        return []
    
    slots = sorted(slots, key=lambda x: x[0])
    streaks = []
    current_streak = 1
    current_dir = slots[0][3]
    
    for i in range(1, len(slots)):
        if slots[i][3] == current_dir:
            current_streak += 1
        else:
            streaks.append((current_dir, current_streak))
            current_dir = slots[i][3]
            current_streak = 1
    streaks.append((current_dir, current_streak))
    return streaks

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────
def calc_wr_by_hour(hour_stats):
    """Calculate win rate by UTC hour"""
    results = []
    for hour in range(24):
        s = hour_stats.get(hour, {"up": 0, "down": 0})
        total = s["up"] + s["down"]
        if total >= 10:  # only show hours with enough data
            wr = s["up"] / total * 100
            results.append((hour, wr, total, s["up"], s["down"]))
    return sorted(results, key=lambda x: -x[1])

def calc_streak_stats(streaks):
    """Streak length distribution"""
    up_streaks = [s[1] for s in streaks if s[0] == "UP"]
    down_streaks = [s[1] for s in streaks if s[0] == "DOWN"]
    
    stats = {}
    for label, streaks_list in [("UP", up_streaks), ("DOWN", down_streaks)]:
        if streaks_list:
            stats[label] = {
                "count": len(streaks_list),
                "avg": round(statistics.mean(streaks_list), 2),
                "max": max(streaks_list),
                "median": statistics.median(streaks_list),
                "min": min(streaks_list),
                ">=3": sum(1 for s in streaks_list if s >= 3),
                ">=5": sum(1 for s in streaks_list if s >= 5),
            }
    return stats

def calc_recent_performance(conn, lookback=50):
    """Recent win rate (last N settled slots)"""
    c = conn.cursor()
    c.execute("""
        SELECT winner FROM slots
        WHERE winner IN ('UP', 'DOWN')
        ORDER BY slot_ts DESC
        LIMIT ?
    """, (lookback,))
    rows = c.fetchall()
    if not rows:
        return None
    ups = sum(1 for r in rows if r[0] == "UP")
    return {"total": len(rows), "up": ups, "down": len(rows)-ups, "wr": round(ups/len(rows)*100, 1)}

def calc_momentum_edge(conn, lookback=100):
    """Analyze if momentum (same direction as previous) gives edge"""
    slots = query_all_settled(conn)
    if len(slots) < 20:
        return None
    
    slots = sorted(slots, key=lambda x: x[0])
    momentum_wins = 0
    momentum_total = 0
    
    for i in range(1, len(slots)):
        prev_dir = slots[i-1][3]
        curr_dir = slots[i][3]
        momentum_total += 1
        if curr_dir == prev_dir:
            momentum_wins += 1
    
    return {
        "momentum_wr": round(momentum_wins/momentum_total*100, 1) if momentum_total > 0 else None,
        "momentum_count": momentum_total,
        "edge": round(momentum_wins/momentum_total - 0.5, 3) if momentum_total > 0 else 0
    }

def calc_price_edge(conn):
    """Analyze if there is a price edge (overpriced/underpriced UP)"""
    c = conn.cursor()
    c.execute("""
        SELECT slot_ts, up_price, down_price, winner
        FROM slots
        WHERE winner IN ('UP', 'DOWN') AND up_price IS NOT NULL AND down_price IS NOT NULL
        ORDER BY slot_ts DESC
        LIMIT 500
    """)
    slots = c.fetchall()
    if not slots:
        return None
    
    correct = 0
    total = 0
    edges = []
    
    for slot_ts, up_price, down_price, winner in slots:
        total += 1
        implied_up = up_price
        actual = 1 if winner == "UP" else 0
        
        # Bookie's edge: if implied_up > 50%, UP is "overpriced"
        edge = implied_up - actual
        edges.append(edge)
        
        if (implied_up >= 0.5 and winner == "UP") or (implied_up < 0.5 and winner == "DOWN"):
            correct += 1
    
    if total == 0:
        return None
    
    avg_edge = statistics.mean(edges) if edges else 0
    return {
        "total_with_prices": total,
        "bookie_correct": round(correct/total*100, 1),
        "avg_implied_edge": round(avg_edge, 3),
        "up_overpriced": sum(1 for e in edges if e > 0) / total * 100,
        "down_overpriced": sum(1 for e in edges if e < 0) / total * 100 * -1
    }

def calc_time_since_last(conn):
    """Time since last UP and DOWN"""
    c = conn.cursor()
    c.execute("""
        SELECT slot_ts, winner FROM slots
        WHERE winner IN ('UP', 'DOWN')
        ORDER BY slot_ts DESC
        LIMIT 20
    """)
    rows = c.fetchall()
    if not rows:
        return None
    now = time.time()
    last_up = last_down = None
    for slot_ts, winner in rows:
        if winner == "UP" and last_up is None:
            last_up = slot_ts
        if winner == "DOWN" and last_down is None:
            last_down = slot_ts
        if last_up and last_down:
            break
    return {
        "last_up_slots_ago": (now - last_up) / 300 if last_up else None,
        "last_down_slots_ago": (now - last_down) / 300 if last_down else None,
        "last_up_ts": last_up,
        "last_down_ts": last_down
    }

def find_hot_cold_streaks(conn, min_streak=4):
    """Find current hot/cold streaks"""
    c = conn.cursor()
    c.execute("""
        SELECT slot_ts, winner FROM slots
        WHERE winner IN ('UP', 'DOWN')
        ORDER BY slot_ts DESC
        LIMIT 50
    """)
    rows = c.fetchall()
    if not rows:
        return None
    
    # Count consecutive from most recent
    most_recent = rows[0][1]
    streak = 1
    for i in range(1, len(rows)):
        if rows[i][1] == most_recent:
            streak += 1
        else:
            break
    
    return {
        "current_direction": most_recent,
        "current_streak": streak,
        "is_hot": most_recent == "UP" and streak >= min_streak,
        "is_cold": most_recent == "DOWN" and streak >= min_streak
    }

def generate_markdown_report(conn, all_slots):
    """Generate full research report"""
    now = datetime.now(timezone.utc)
    hkt = now.astimezone(timezone(timedelta(hours=8)))
    
    report = []
    report.append(f"# 🔬 BTC-5m 每小時研究報告")
    report.append(f"**生成時間:** {hkt.strftime('%Y-%m-%d %H:%M')} HKT / {now.strftime('%H:%M')} UTC")
    report.append("")
    
    # ── 1. DB Status ────────────────────────────────────────────────────────
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM slots")
    total_slots = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM slots WHERE winner IN ('UP', 'DOWN')")
    settled = c.fetchone()[0]
    c.execute("SELECT MAX(slot_ts) FROM slots WHERE winner IN ('UP', 'DOWN')")
    latest = c.fetchone()[0]
    
    report.append(f"## 📊 數據庫狀態")
    report.append(f"| 指標 | 數值 |")
    report.append(f"|------|------|")
    report.append(f"| 總記錄槽位 | {total_slots} |")
    report.append(f"| 已結算 | {settled} |")
    report.append(f"| 最新結算 | {datetime.fromtimestamp(latest, tz=timezone.utc).strftime('%m-%d %H:%M UTC') if latest else 'N/A'} |")
    report.append("")
    
    # ── 2. Recent Performance ───────────────────────────────────────────────
    report.append(f"## 📈 最近表現 (最近50局)")
    recent = calc_recent_performance(conn, 50)
    if recent:
        report.append(f"| 指標 | 數值 |")
        report.append(f"|------|------|")
        report.append(f"| 胜率 | **{recent['wr']}%** |")
        report.append(f"| UP次數 | {recent['up']} |")
        report.append(f"| DOWN次數 | {recent['down']} |")
        report.append("")
        
        # Signal
        if recent['wr'] >= 55:
            signal = "🟢 優勢明顯，可考慮正常倉位"
        elif recent['wr'] <= 45:
            signal = "🔴 劣勢明顯，保守操作"
        else:
            signal = "🟡 中性區間，觀望為主"
        report.append(f"**信號:** {signal}")
        report.append("")
    
    # ── 3. Hot/Cold Streaks ──────────────────────────────────────────────────
    report.append(f"## 🔥 連續走勢")
    hot = find_hot_cold_streaks(conn)
    if hot:
        report.append(f"| 指標 | 數值 |")
        report.append(f"|------|------|")
        report.append(f"| 當前方向 | **{hot['current_direction']}** |")
        report.append(f"| 連續次數 | {hot['current_streak']} |")
        report.append(f"| 狀態 | {'🔥 HOT' if hot['is_hot'] else '❄️ COLD' if hot['is_cold'] else '➡️ 中性'} |")
        report.append("")
    
    time_since = calc_time_since_last(conn)
    if time_since:
        report.append(f"**最後UP:** {time_since['last_up_slots_ago']:.1f} slots ago")
        report.append(f"**最後DOWN:** {time_since['last_down_slots_ago']:.1f} slots ago")
        report.append("")
    
    # ── 4. Win Rate by Hour ──────────────────────────────────────────────────
    report.append(f"## ⏰ UTC時段胜率 (每小時)")
    hour_wr = calc_wr_by_hour(query_by_hour(conn))
    if hour_wr:
        report.append(f"| UTC小時 | 胜率 | 總局數 | UP | DOWN | 信號 |")
        report.append(f"|--------|------|--------|-----|-------|------|")
        for hour, wr, total, up, down in hour_wr:
            signal = "🟢" if wr >= 55 else "🔴" if wr <= 45 else "🟡"
            report.append(f"| {hour:02d}:00 | **{wr:.1f}%** | {total} | {up} | {down} | {signal} |")
        report.append("")
        # Best and worst hours
        if hour_wr:
            best = hour_wr[0]
            worst = hour_wr[-1]
            report.append(f"**最佳時段:** {best[0]:02d}:00 UTC ({best[1]:.1f}%) | **最差時段:** {worst[0]:02d}:00 UTC ({worst[1]:.1f}%)")
            report.append("")
    
    # ── 5. Momentum Edge ─────────────────────────────────────────────────────
    report.append(f"## 💨 動量效應")
    momentum = calc_momentum_edge(conn)
    if momentum:
        report.append(f"| 指標 | 數值 |")
        report.append(f"|------|------|")
        report.append(f"| 動量胜率 | **{momentum['momentum_wr']}%** |")
        report.append(f"| 統計局數 | {momentum['momentum_count']} |")
        edge = momentum['edge']
        report.append(f"| 相對50%優勢 | {edge:+.1%} |")
        
        if edge > 0.02:
            signal = "🟢 動量策略有效：順勢而為"
        elif edge < -0.02:
            signal = "🔴 反動量策略有效：逆勢操作"
        else:
            signal = "🟡 動量無明顯優勢"
        report.append(f"| 分析 | {signal} |")
        report.append("")
    
    # ── 6. Price Edge (Bookie's Edge) ───────────────────────────────────────
    report.append(f"## 📉 價格偏差分析")
    price_edge = calc_price_edge(conn)
    if price_edge:
        report.append(f"| 指標 | 數值 |")
        report.append(f"|------|------|")
        report.append(f"| 統計樣本 | {price_edge['total_with_prices']} |")
        report.append(f"| 價格預測正確率 | **{price_edge['bookie_correct']}%** |")
        report.append(f"| UP被高估比例 | {price_edge['up_overpriced']:.1f}% |")
        report.append(f"| DOWN被高估比例 | {price_edge['down_overpriced']:.1f}% |")
        report.append("")
    
    # ── 7. Streak Distribution ───────────────────────────────────────────────
    report.append(f"## 📊 連續走勢分布")
    streaks = query_streaks(conn)
    streak_stats = calc_streak_stats(streaks)
    if streak_stats:
        for label, s in streak_stats.items():
            report.append(f"**{label} 連續:**")
            report.append(f"| 指標 | 數值 |")
            report.append(f"|------|------|")
            report.append(f"| 總連續次數 | {s['count']} |")
            report.append(f"| 平均長度 | {s['avg']} |")
            report.append(f"| 最長 | {s['max']} |")
            report.append(f"| 中位數 | {s['median']} |")
            report.append(f"| ≥3次占比 | {s['>=3']}/{s['count']} ({s['>=3']/s['count']*100:.1f}%) |")
            report.append(f"| ≥5次占比 | {s['>=5']}/{s['count']} ({s['>=5']/s['count']*100:.1f}%) |")
            report.append("")
    
    # ── 8. Research Insights ────────────────────────────────────────────────
    report.append(f"## 💡 研究洞察")
    
    insights = []
    
    # Hot/cold streak insight
    if hot and hot['current_streak'] >= 4:
        if hot['is_hot']:
            insights.append(f"🔥 **火熱UP:** 連續{hot['current_streak']}次UP，關注何時反轉（通常≥5次後反轉概率提升）")
        else:
            insights.append(f"❄️ **冰冷DOWN:** 連續{hot['current_streak']}次DOWN，關注超跌反彈機會")
    
    # Recent WR
    if recent:
        if recent['wr'] >= 58:
            insights.append(f"🟢 **近50局强势:** {recent['wr']}%胜率，市場趨勢明顯")
        elif recent['wr'] <= 42:
            insights.append(f"🔴 **近50局弱勢:** {recent['wr']}%胜率，市場混沌或趨勢逆轉")
    
    # Momentum
    if momentum and momentum['momentum_wr']:
        if momentum['momentum_wr'] >= 53:
            insights.append(f"💨 **動量策略:** 胜率{momentum['momentum_wr']}%，順勢而為有效")
        elif momentum['momentum_wr'] <= 47:
            insights.append(f"🔄 **反動量策略:** 胜率{momentum['momentum_wr']}%，逆勢操作更佳")
    
    # Time-based
    if hour_wr:
        best_hour = hour_wr[0]
        worst_hour = hour_wr[-1]
        if best_hour[1] >= 60:
            insights.append(f"⏰ **最佳時段:** {best_hour[0]:02d}:00 UTC，胜率{best_hour[1]:.1f}%")
        if worst_hour[1] <= 40:
            insights.append(f"⏰ **避開時段:** {worst_hour[0]:02d}:00 UTC，胜率{worst_hour[1]:.1f}%")
    
    if not insights:
        insights.append("📊 數據不足，等待更多樣本...")
    
    for insight in insights:
        report.append(f"- {insight}")
    report.append("")
    
    # ── 9. Recommended Actions ───────────────────────────────────────────────
    report.append(f"## 🎯 建議行動")
    
    actions = []
    
    # Time filter
    current_hour_utc = (int(time.time()) // 3600) % 24
    current_wr = None
    for hour, wr, total, up, down in hour_wr:
        if hour == current_hour_utc and total >= 10:
            current_wr = wr
            break
    
    if current_wr:
        if current_wr >= 55:
            actions.append(f"✅ 當前UTC {current_hour_utc:02d}:00 胜率{current_wr:.1f}%，**正常交易**")
        elif current_wr <= 45:
            actions.append(f"⚠️ 當前UTC {current_hour_utc:02d}:00 胜率{current_wr:.1f}%，**降低倉位或觀望**")
        else:
            actions.append(f"➖ 當前UTC {current_hour_utc:02d}:00 胜率{current_wr:.1f}%，**保守操作**")
    
    # Hot/cold
    if hot and hot['current_streak'] >= 5:
        if hot['is_hot']:
            actions.append(f"🔥 UP已連續{hot['current_streak']}次，考慮等待冷卻後再跟")
        else:
            actions.append(f"❄️ DOWN已連續{hot['current_streak']}次，關注反轉機會")
    
    # Momentum
    if momentum and abs(momentum['edge']) > 0.02:
        if momentum['edge'] > 0:
            actions.append(f"💨 動量策略優勢+{momentum['edge']:.1%}，傾向跟隨趨勢")
        else:
            actions.append(f"🔄 反動量策略優勢{momentum['edge']:.1%}，關注逆勢信號")
    
    if not actions:
        actions.append("📊 等待更多數據支持...")
    
    for action in actions:
        report.append(f"- {action}")
    report.append("")
    
    report.append(f"---")
    report.append(f"*Report generated by 小獅子 🦁 BTC-5m Research System*")
    
    return "\n".join(report)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'='*60}")
    print(f"[{now_str}] BTC-5m Hourly Research")
    print(f"{'='*60}")
    
    conn = get_db()
    if not conn:
        print("❌ DB not found. Run research_collector.py first.")
        return
    
    # Quick stats
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM slots WHERE winner IN ('UP', 'DOWN')")
    settled = c.fetchone()[0]
    print(f"📊 DB: {settled} settled slots")
    
    # Generate report
    print(f"\n🔬 Generating research report...")
    report = generate_markdown_report(conn, None)
    
    # Save report
    today = datetime.now().strftime('%Y-%m-%d')
    report_path = REPORT_DIR / f"research_{today}.md"
    
    # Append to daily report (or create)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n{report}")
    
    # Also save latest report separately
    latest_path = REPORT_DIR / "latest.md"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    print(f"   Latest: {latest_path}")
    
    # Print summary
    print(f"\n{'='*40}")
    print("KEY INSIGHTS:")
    for line in report.split("\n"):
        if line.startswith("## "):
            print(f"  {line}")
        elif line.startswith("**"):
            print(f"  {line}")
    
    conn.close()

if __name__ == "__main__":
    main()
