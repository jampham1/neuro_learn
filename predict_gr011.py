"""
predict_gr011.py
================
Runs the XGBoost model trained on GR010 on GR011's session data
and produces genuine predictions of days_to_next_switch — without
using any look-forward labels from GR011.

Usage:
    python predict_gr011.py

Outputs:
    GR011_predictions.csv   — one row per session with predicted DTN
    GR011_pred_chart.png    — predicted vs actual (if actuals available)

Requires:
    model_switch.json       — trained on GR010 by train_switch_model.py
    GR011_switch_labels.csv — GR011 session data (real or synthetic)
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path('model_switch.json')
GR011_CSV   = Path('GR011_switch_labels.csv')
OUT_CSV     = Path('GR011_predictions.csv')
OUT_CHART   = Path('GR011_pred_chart.png')

# ── Load model ─────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run train_switch_model.py on GR010 data first."
    )

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
expected_features = model.get_booster().feature_names
print(f"Model loaded. Expects {len(expected_features)} features:")
print(" ", expected_features)

# ── Load GR011 data ────────────────────────────────────────────────────────────
if not GR011_CSV.exists():
    raise FileNotFoundError(f"GR011 data not found: {GR011_CSV}")

df = pd.read_csv(GR011_CSV).sort_values('day_index').reset_index(drop=True)
print(f"\nLoaded {len(df)} GR011 sessions")
print(f"Columns available: {list(df.columns)}")

# ── Reconstruct any missing features ──────────────────────────────────────────
# The synthetic GR011 CSV may not have all lag/delta features that
# train_switch_model.py computed from raw .mat data. We recompute them here
# from whatever columns are present, matching the training feature engineering.

lag_base_cols = [
    'perf_final', 'l_perf_final', 'r_perf_final',
    'rt_median', 'n_trials', 'last_block_perf',
]

for col in lag_base_cols:
    if col not in df.columns:
        continue
    lag1_col  = f'{col}_lag1'
    lag2_col  = f'{col}_lag2'
    delta_col = f'{col}_delta'
    roll5_col = f'{col}_roll5'

    if lag1_col not in df.columns:
        df[lag1_col] = df[col].shift(1)
    if lag2_col not in df.columns:
        df[lag2_col] = df[col].shift(2)
    if delta_col not in df.columns:
        df[delta_col] = df[col].diff()
    if roll5_col not in df.columns:
        df[roll5_col] = df[col].rolling(5, min_periods=2).mean()

# Encode categorical modality columns if present
for raw_col, enc_col in [
    ('session_start_mod', 'session_start_mod_enc'),
    ('session_end_mod',   'session_end_mod_enc'),
    ('last_block_mod',    'last_block_mod_enc'),
]:
    if raw_col in df.columns and enc_col not in df.columns:
        df[enc_col] = (df[raw_col] == 'Vision').astype(float)

# Recompute within_phase_mastery if possible (sigmoid fit per phase)
# Falls back to a simple linear proxy if scipy not available
if 'within_phase_mastery' not in df.columns:
    try:
        from scipy.optimize import curve_fit

        def sigmoid(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        wpm = []
        for pid, grp in df.groupby('phase_id'):
            xs = grp['session_in_phase'].values.astype(float)
            ys = grp['last_block_perf'].values
            try:
                popt, _ = curve_fit(
                    sigmoid, xs, ys,
                    p0=[0.9, 0.3, 5], maxfev=2000,
                    bounds=([0.5, 0.01, 0], [1.5, 5, 30])
                )
                fitted = sigmoid(xs, *popt)
                # within_phase_mastery = current / asymptote
                wpm.extend((ys / popt[0]).clip(0, 1).tolist())
            except Exception:
                # Fallback: linear ramp within phase
                wpm.extend((xs / (xs.max() + 1)).tolist())

        df['within_phase_mastery'] = wpm
        print("Computed within_phase_mastery via sigmoid fit")
    except ImportError:
        # Simple proxy: progress_to_switch if available, else linear
        if 'progress_to_switch' in df.columns:
            df['within_phase_mastery'] = df['progress_to_switch']
        else:
            df['within_phase_mastery'] = df.groupby('phase_id')['session_in_phase'].transform(
                lambda x: x / (x.max() + 1)
            )
        print("within_phase_mastery: using proxy (scipy not available)")

    # Lag/delta for within_phase_mastery
    df['within_phase_mastery_lag1']  = df['within_phase_mastery'].shift(1)
    df['within_phase_mastery_delta'] = df['within_phase_mastery'].diff()

# global_mastery proxy: cumulative progress across phases
if 'global_mastery' not in df.columns:
    max_phase = df['phase_id'].max()
    df['global_mastery'] = (
        (df['phase_id'] - 1) / max(max_phase, 1) +
        df.get('within_phase_mastery', df['session_in_phase'] / 10).clip(0, 1) / max(max_phase, 1)
    ).clip(0, 1)
    df['global_mastery_delta'] = df['global_mastery'].diff()
    print("global_mastery: computed as phase-fraction proxy")

# ── Align features to model's expected feature list ───────────────────────────
print(f"\nAligning to {len(expected_features)} model features...")

missing = [f for f in expected_features if f not in df.columns]
extra   = [c for c in df.columns if c not in expected_features]

if missing:
    print(f"  Missing features (will fill with 0): {missing}")
    for col in missing:
        df[col] = 0.0

if extra:
    print(f"  Extra columns ignored: {extra[:10]}{'...' if len(extra)>10 else ''}")

# Build feature matrix in exact model order
X = df[expected_features].copy()

# Fill any remaining NaNs with column medians (same as training)
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

# ── Predict ────────────────────────────────────────────────────────────────────
print("\nRunning predictions...")
preds = model.predict(X)
preds = np.clip(preds, 0, None)  # DTN can't be negative

df['dtn_predicted'] = preds

# ── Compare to actual labels if available ─────────────────────────────────────
has_actual = 'days_to_next_switch' in df.columns and df['days_to_next_switch'].notna().sum() > 10

if has_actual:
    valid = df['days_to_next_switch'].notna()
    actual  = df.loc[valid, 'days_to_next_switch'].values
    predicted = df.loc[valid, 'dtn_predicted'].values
    mae  = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"\nPrediction accuracy vs synthetic labels:")
    print(f"  MAE  = {mae:.2f} sessions")
    print(f"  RMSE = {rmse:.2f} sessions")
    print(f"  R²   = {r2:.4f}")
    print(f"\n  Note: these metrics compare against synthetic look-forward labels,")
    print(f"  not ground truth. They measure model consistency, not real accuracy.")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_cols = ['day_index', 'date', 'phase_id', 'session_in_phase',
            'session_end_mod', 'last_block_perf', 'dtn_predicted']
if has_actual:
    out_cols.append('days_to_next_switch')

df[out_cols].to_csv(OUT_CSV, index=False)
print(f"\nSaved predictions to {OUT_CSV}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('GR011 — Model Predictions (GR010-trained XGBoost)', fontsize=13)

# Panel 1: Performance trajectory
ax1 = axes[0]
vision_mask = df['session_end_mod'].str.lower().str.contains('vision', na=False)
ax1.scatter(df.loc[vision_mask, 'day_index'],
            df.loc[vision_mask, 'last_block_perf'],
            color='#1e4d8c', s=20, zorder=3, label='Vision')
ax1.scatter(df.loc[~vision_mask, 'day_index'],
            df.loc[~vision_mask, 'last_block_perf'],
            color='#c05a1a', s=20, zorder=3, label='Audio')
if 'last_block_perf_roll5' in df.columns:
    ax1.plot(df['day_index'], df['last_block_perf_roll5'],
             color='#2d6a4f', lw=2, label='5-sess avg')
ax1.set_ylabel('Performance')
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9)
ax1.set_title('Learning trajectory')

# Add phase boundaries
for pid, grp in df.groupby('phase_id'):
    first_day = grp['day_index'].iloc[0]
    if first_day > df['day_index'].iloc[0]:
        ax1.axvline(first_day, color='#c8b878', lw=0.8, ls='--', alpha=0.5)

# Panel 2: Predicted vs actual DTN
ax2 = axes[1]
ax2.plot(df['day_index'], df['dtn_predicted'],
         color='#8b4fa8', lw=2, label='Predicted DTN', zorder=3)

if has_actual:
    ax2.plot(df['day_index'], df['days_to_next_switch'],
             color='#888', lw=1.5, ls='--', alpha=0.7, label='Actual DTN (synthetic)')

# Mark phase boundaries
for pid, grp in df.groupby('phase_id'):
    first_day = grp['day_index'].iloc[0]
    if first_day > df['day_index'].iloc[0]:
        ax2.axvline(first_day, color='#c8b878', lw=0.8, ls='--', alpha=0.5)

ax2.set_ylabel('Days to next switch')
ax2.set_xlabel('Session (day index)')
ax2.legend(fontsize=9)
ax2.set_title('Predicted days to next modality switch')
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUT_CHART, dpi=150, bbox_inches='tight')
print(f"Saved chart to {OUT_CHART}")

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n── Last 5 sessions ──────────────────────────────────────────────")
print(df[out_cols].tail(10).to_string(index=False))
print("\nDone.")

# ── Auto-embed into index.html if present ──────────────────────────────────────
html_path = Path('index.html')
embed_path = Path('embed_predictions.py')
if html_path.exists() and embed_path.exists():
    import subprocess
    print("\nAuto-embedding predictions into index.html...")
    result = subprocess.run(['python', 'embed_predictions.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Embed failed:", result.stderr)
else:
    print("\nTo update index.html with these predictions, run:")
    print("  python embed_predictions.py")