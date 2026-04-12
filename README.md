# BTC-5m Polymarket Trading System

**⚠️ This repo contains NO API keys or wallet credentials.**

## Quick Setup

```bash
# 1. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install requests numpy pandas scikit-learn flaml lightgbm xgboost py-clob-client web3 beautifulsoup4 lxml

# 3. Add your credentials
cp your_key.txt ~/.openclaw/eth_key.txt
cp your_creds.json ~/.openclaw/fresh_creds.json

# 4. Run
SIM_MODE=1 python3 real_trader_newv1.py   # Paper trade
python3 real_trader_newv1.py                # Live trade
```

## Core Files

| File | Purpose |
|------|---------|
| `real_trader_newv1.py` | Main v5B strategy |
| `strategy_v5A.py` | Strategy variant |
| `SKILL.md` | Full documentation |
| `backtest.py` | Backtest engine |
| `train_beta_v1.py` | ML model training |

## Disclaimer

Cryptocurrency trading is high-risk. No guarantees.
