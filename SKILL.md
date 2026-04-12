# BTC-5m Polymarket 交易系統 — ConservativeV1 + v43b PRIMARY

**Version:** ConservativeV1-v43b | **Updated:** 2026-04-09 07:04 HKT | **Author:** Little Lion 🦁

---

## 系統架構（2026-04-09 最新）

```
┌─────────────────────────────────────────────────────────────┐
│        Polymarket BTC-5m ConservativeV1 + v43b          │
│  real_trader_conservative.py                             │
│                                                          │
│  模型優先級：                                            │
│  • v43b (LogReg balanced, 14F, CV=62.6%) ← PRIMARY ⭐   │
│  • FLAML v42 (ExtraTree, 43F, CV=53.5%) ← Fallback     │
│  • v17 / v35 / v41 ← 最後備用                          │
│                                                          │
│  Active Weights:
- Feature1: 0.4
- Feature2: 0.6
- Feature3: 0.3
- Feature4: 0.7                                         │
│  - imbalance: 0/1 = 0%                                  │
│  - up_count: 0/1 = 0%                                   │
│  - down_count: 0/1 = 0%                                 │
│  - total_amt: 0/1 = 0%                                  │
│                                                          │
│  修復內容：                                              │
│  • SIM Bug Fix: 熔斷機制超賣              │
└─────────────────────────────────────────────────────────────┘