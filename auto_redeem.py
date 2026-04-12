#!/usr/bin/env python3
"""
auto_redeem.py — 自動 Redeem 已結算 WIN 倉位
==========================================
流程：
  1. 從 Polymarket Gamma API 拿當前帳號的 positions
  2. 篩出 redeemable（market resolved, 持有 winning token > 0）
  3. 呼叫 CTF Exchange 合約的 redeemPositions()
  4. 印出 tx hash 和回收 USDC 金額

用法：
  python3 auto_redeem.py          # 正式執行
  python3 auto_redeem.py --dry    # 只列出，不送 tx
"""

import os, sys, json, time, requests
from web3 import Web3

DRY_RUN = "--dry" in sys.argv

# ── 配置 ─────────────────────────────────────────────────────────────────────
POLYGON_RPC   = "https://1rpc.io/matic"
CHAIN_ID      = 137
FUNDER        = "0x8d8BA13d2c3D1935bF0b8BD2052AC73e8E329376"

# Polymarket 合約地址（Polygon mainnet）
CTF_EXCHANGE  = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
CTF_CONTRACT  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # ERC1155 Conditional Tokens
USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# CTF redeemPositions ABI（最小化）
CTF_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"}
        ],
        "name": "payoutDenominator",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "uint256", "name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ── 工具函數 ─────────────────────────────────────────────────────────────────

def load_key():
    key_path = os.path.expanduser("~/.openclaw/eth_key.txt")
    return open(key_path).read().strip()

def fetch_positions(wallet):
    """從 data-api 拿帳號的所有 positions"""
    url = f"https://data-api.polymarket.com/positions?user={wallet}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()  # list of position dicts, each has 'redeemable' bool field

# ── 主邏輯 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🦁 BTC-5m Auto Redeem")
    print(f"  {'[DRY RUN - 不送 tx]' if DRY_RUN else '[LIVE - 送真實 tx]'}")
    print("=" * 60)

    key = load_key()
    w3  = Web3(Web3.HTTPProvider(POLYGON_RPC))

    if not w3.is_connected():
        print("❌ Polygon RPC 連線失敗")
        return

    account = w3.eth.account.from_key(key)
    wallet  = account.address
    print(f"👛 Wallet: {wallet}")
    print(f"💰 FUNDER: {FUNDER}")

    # 初始化合約
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_CONTRACT), abi=CTF_ABI)

    # 拿 positions
    print("\n📡 拉取 positions...")
    try:
        positions = fetch_positions(FUNDER)
    except Exception as e:
        print(f"❌ Gamma API 失敗: {e}")
        return

    print(f"  總 positions: {len(positions)}")

    redeemable = []
    for pos in positions:
        size = float(pos.get("size", 0))
        if size < 0.001:
            continue

        # data-api 直接提供 redeemable flag
        if not pos.get("redeemable", False):
            continue

        cond_id = pos.get("conditionId", "")
        outcome = pos.get("outcome", "")
        title   = pos.get("title", "")[:60]
        asset   = pos.get("asset", "")
        outcome_index = pos.get("outcomeIndex", 0)  # 0=YES/UP, 1=NO/DOWN

        if not cond_id:
            continue

        redeemable.append({
            "question":     title,
            "cond_id":      cond_id,
            "outcome":      outcome,
            "outcome_index":outcome_index,
            "size":         size,
            "token_id":     asset,
            "cur_price":    float(pos.get("curPrice", 1)),
            "cash_pnl":     float(pos.get("cashPnl", 0)),
        })

    print(f"\n✅ 可 Redeem 的 positions: {len(redeemable)}")

    if not redeemable:
        print("\n🔍 沒有可 Redeem 的 positions。")
        print("   原因可能：")
        print("   1. 市場未結算")
        print("   2. 贏注已自動結算入帳")
        print("   3. 帳號沒有未 redeem 的餘額")
        return

    total_redeemable = sum(p['size'] for p in redeemable)
    total_pnl = sum(p['cash_pnl'] for p in redeemable)
    print(f"  總可回收 USDC: ~${total_redeemable:.2f} | 預估 PnL: +${total_pnl:.2f}")
    for i, pos in enumerate(redeemable):
        print(f"\n  [{i+1}] {pos['question']}")
        print(f"       Outcome: {pos['outcome']} | Size: {pos['size']:.4f} USDC")
        print(f"       OutcomeIndex: {pos['outcome_index']} | ConditionId: {pos['cond_id'][:20]}...")
        print(f"       PnL: +${pos['cash_pnl']:.2f}")

    if DRY_RUN:
        print("\n  [DRY RUN 結束，不送 tx]")
        return

    # 送 redeemPositions tx
    nonce = w3.eth.get_transaction_count(wallet)
    gas_price = w3.eth.gas_price

    success_count = 0
    for pos in redeemable:
        cond_id  = pos["cond_id"]
        print(f"\n📤 Redeeming: {pos['question']}")

        try:
            cond_bytes = bytes.fromhex(cond_id.replace("0x", ""))
            parent_coll = bytes(32)  # 0x000...000 (root collection)

            # Polymarket 2-outcome 市場：
            # outcomeIndex 0 = YES/UP → indexSets = [1]
            # outcomeIndex 1 = NO/DOWN → indexSets = [2]
            idx = pos.get("outcome_index", 0)
            index_sets = [1 << idx]  # 0 -> 1, 1 -> 2

            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_CONTRACT),
                parent_coll,
                cond_bytes,
                index_sets,
            ).build_transaction({
                "from":     wallet,
                "nonce":    nonce,
                "gas":      200000,
                "gasPrice": int(gas_price * 1.1),
                "chainId":  CHAIN_ID,
            })

            signed = w3.eth.account.sign_transaction(tx, key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  ✅ Tx 送出: {tx_hash.hex()}")
            print(f"  🔗 https://polygonscan.com/tx/{tx_hash.hex()}")

            # 等待確認
            print("  ⏳ 等待確認...")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                print(f"  ✅ 確認成功！Gas 使用: {receipt.gasUsed}")
                success_count += 1
            else:
                print(f"  ❌ Tx 失敗（status=0）")

            nonce += 1
            time.sleep(2)

        except Exception as e:
            print(f"  ❌ Redeem 失敗: {e}")

    print(f"\n{'='*60}")
    print(f"  🏁 完成：{success_count}/{len(redeemable)} 成功 Redeem")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
