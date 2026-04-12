#!/usr/bin/env python3
"""
FLAML AutoML Trainer v42
2026-04-06

- 使用 v3 完整數據集 (2582 samples × 43 features)
- FLAML 自動選最優模型 (lgbm / xgboost / rf / lr 等)
- 增量訓練: 從 research DB v39_research 表抓新數據追加到 ml_dataset_v3.json
- 輸出: ml_model_v42_flaml.pkl + ml_features_v42.json + 訓練報告
"""
import os, sys, json, time, sqlite3, pickle, warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from flaml import AutoML

warnings.filterwarnings('ignore')

SKILL = Path(__file__).parent
DATA  = SKILL / 'data'
DB    = DATA / 'btc5m_research.db'
V3    = DATA / 'ml_dataset_v3.json'
MODEL_OUT   = DATA / 'ml_model_v42_flaml.pkl'
FEATURES_OUT = DATA / 'ml_features_v42.json'
SCALER_OUT  = DATA / 'scaler_v42_flaml.pkl'
META_OUT    = DATA / 'ml_model_v42_meta.json'

FLAML_TIME_BUDGET = int(os.environ.get('FLAML_BUDGET', '120'))  # seconds

# ═══════════════════════════════════════════════════════
# STEP 1: 載入 v3 基礎數據集
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("FLAML v42 Training — BTC-5m AutoML")
print("=" * 60)

with open(V3) as f:
    v3 = json.load(f)

feature_names = v3['features']
base_samples  = v3['samples']
print(f"✅ v3 base dataset: {len(base_samples)} samples × {len(feature_names)} features")

# ═══════════════════════════════════════════════════════
# STEP 2: 增量數據 — 從 DB v39_research 抓新結算記錄
# ═══════════════════════════════════════════════════════
new_samples = []
if DB.exists():
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()

        # Check if v39_research table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='v39_research'")
        if c.fetchone():
            # Get columns
            c.execute("PRAGMA table_info(v39_research)")
            cols = [r[1] for r in c.fetchall()]

            # Only use rows with actual_outcome (settled)
            c.execute("""
                SELECT * FROM v39_research
                WHERE actual_outcome IN ('UP','DOWN')
                ORDER BY slot_ts ASC
            """)
            rows = c.fetchall()
            print(f"📊 v39_research settled rows: {len(rows)}")

            # Convert to feature dicts using the 43 features
            for row in rows:
                row_dict = dict(zip(cols, row))
                label = 1 if row_dict.get('actual_outcome') == 'UP' else 0
                # Map v39_research columns to v3 feature names
                sample = {}
                for feat in feature_names:
                    val = row_dict.get(feat, 0.0)
                    sample[feat] = float(val) if val is not None else 0.0
                sample['label'] = label
                sample['slot_ts'] = row_dict.get('slot_ts', 0)
                new_samples.append(sample)

        conn.close()
    except Exception as e:
        print(f"⚠️ DB incremental load error: {e}")

# Deduplicate by slot_ts
base_slots = set(s.get('slot_ts', 0) for s in base_samples)
new_unique = [s for s in new_samples if s.get('slot_ts', 0) not in base_slots]
print(f"📈 Incremental new samples: {len(new_unique)} (deduplicated)")

all_samples = base_samples + new_unique
print(f"📊 Total training samples: {len(all_samples)}")

# ═══════════════════════════════════════════════════════
# STEP 3: Build X, y
# ═══════════════════════════════════════════════════════
X = np.array([[s.get(f, 0.0) for f in feature_names] for s in all_samples])
y = np.array([s['label'] for s in all_samples])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\n🔢 Dataset: {X.shape[0]} × {X.shape[1]}")
print(f"   UP={sum(y)} ({sum(y)/len(y)*100:.1f}%) | DOWN={len(y)-sum(y)} ({(1-sum(y)/len(y))*100:.1f}%)")

# Leakage check
corrs = [(f, abs(np.corrcoef(X[:,i], y)[0,1])) for i, f in enumerate(feature_names)]
top_corrs = sorted(corrs, key=lambda x: x[1], reverse=True)[:5]
print("\n🔍 Top correlations (should all be < 0.15):")
leakage = False
for f, c in top_corrs:
    flag = "⚠️ LEAKAGE!" if c > 0.3 else "✅"
    print(f"   {flag} {f}: r={c:.4f}")
    if c > 0.3:
        leakage = True
if leakage:
    print("⛔ Detected leakage features — removing before training")
    safe_feats = [f for f, c in corrs if c <= 0.3]
    feat_idx   = [feature_names.index(f) for f in safe_feats]
    X = X[:, feat_idx]
    feature_names = safe_feats
    print(f"   Reduced to {len(feature_names)} safe features")

# ═══════════════════════════════════════════════════════
# STEP 4: FLAML AutoML
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"🚀 FLAML AutoML (budget={FLAML_TIME_BUDGET}s)")
print(f"   Estimators: lgbm, xgboost, rf, extra_tree, lrl1")
print(f"{'='*60}")

automl = AutoML()
automl.fit(
    X_train=X,
    y_train=y,
    task='classification',
    metric='accuracy',
    time_budget=FLAML_TIME_BUDGET,
    estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'lrl1'],
    eval_method='cv',
    n_splits=5,
    seed=42,
    verbose=1,
    log_file_name='',
)

best_model = automl.model.estimator
best_estimator = automl.best_estimator
best_config   = automl.best_config
best_loss     = automl.best_loss

print(f"\n✅ FLAML Best: {best_estimator} | val_accuracy={1-best_loss:.4f}")
print(f"   Config: {best_config}")

# ═══════════════════════════════════════════════════════
# STEP 5: CV Verification (overfitting check)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("5-Fold CV Verification")
print(f"{'='*60}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
cv_mean = cv_scores.mean()
print(f"CV: {cv_mean*100:.2f}% ±{cv_scores.std()*100:.2f}%  {cv_scores}")
print(f"vs random: {(cv_mean-0.5)*100:+.2f}%")

# ═══════════════════════════════════════════════════════
# STEP 6: StandardScaler + Retrain on full data
# ═══════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
best_model.fit(X_scaled, y)

# ═══════════════════════════════════════════════════════
# STEP 7: Save
# ═══════════════════════════════════════════════════════
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(best_model, f)
with open(SCALER_OUT, 'wb') as f:
    pickle.dump(scaler, f)
with open(FEATURES_OUT, 'w') as f:
    json.dump(feature_names, f)

meta = {
    'version':        'v42_flaml',
    'trained_at':     datetime.now().isoformat(),
    'n_samples':      len(all_samples),
    'n_features':     len(feature_names),
    'base_samples':   len(base_samples),
    'incremental':    len(new_unique),
    'best_estimator': best_estimator,
    'best_config':    best_config,
    'cv_accuracy':    round(cv_mean, 4),
    'cv_std':         round(cv_scores.std(), 4),
    'flaml_budget':   FLAML_TIME_BUDGET,
}
with open(META_OUT, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ SAVED:")
print(f"   Model:    {MODEL_OUT}")
print(f"   Scaler:   {SCALER_OUT}")
print(f"   Features: {FEATURES_OUT}")
print(f"   Meta:     {META_OUT}")
print(f"\n📊 Summary:")
print(f"   Estimator : {best_estimator}")
print(f"   CV Acc    : {cv_mean*100:.2f}% (vs 50% random)")
print(f"   Edge      : {(cv_mean-0.5)*100:+.2f}%")
print(f"   n_samples : {len(all_samples)} ({len(new_unique)} incremental new)")
print(f"{'='*60}")
