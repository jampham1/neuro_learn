"""
UPDATED TRAINING MODEL TO PREDICT TIME UNTIL NEXT MODALITY SWITCH

GR010 — Days-to-Next-Switch Predictor
======================================
Reframes the learning trajectory problem around a causally grounded,
protocol-native target: how many sessions until the mouse hits criterion
and triggers the next modality switch?

Outputs:
  model_switch.json          — trained XGBoost model
  GR010_switch_labels.csv    — feature CSV with days_to_next_switch labels
  shap_switch.png            — SHAP feature importance
  GR010_switch_predictions.csv — walk-forward predictions vs actual
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURES_CSV  = Path('GR010_features.csv')
OUT_CSV       = Path('GR010_switch_labels.csv')
PRED_CSV      = Path('GR010_switch_predictions.csv')
MODEL_PATH    = Path('model_switch.json')
SHAP_PATH     = Path('shap_switch.png')
MIN_TRAIN     = 20       # sessions before walk-forward CV starts
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'day_index', 'session_in_phase', 'phase_id', 'n_modality_switches',
    'n_trials', 'engage_rate',
    'perf_final', 'perf_mean', 'perf_slope', 'perf_var',
    'l_perf_final', 'l_perf_mean', 'l_perf_slope',
    'r_perf_final', 'r_perf_mean', 'r_perf_slope',
    'perf_asymmetry',
    'last_block_perf', 'pre_switch_perf', 'post_switch_perf', 'switch_perf_drop',
    'rt_median', 'rt_mean', 'rt_var', 'rt_impulsive_frac', 'rt_slow_frac',
    'psychometric_slope', 'psychometric_threshold', 'psychometric_lapse',
    'perf_final_lag1', 'perf_final_lag2', 'perf_final_delta',
    'last_block_perf_lag1', 'last_block_perf_lag2', 'last_block_perf_delta',
    'rt_median_lag1', 'rt_median_delta',
    'perf_final_roll5', 'last_block_perf_roll5',
    'single_session_phase', 'opto_session', 'assisted', 'auto_reward',
    'session_start_mod_enc', 'session_end_mod_enc', 'last_block_mod_enc',
]

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
)


# ── Label computation ─────────────────────────────────────────────────────────
def compute_days_to_next_switch(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, days_to_next_switch = number of sessions until the
    NEXT modality switch occurs (i.e. the next session where
    n_modality_switches > 0 OR phase_id changes from the previous session).

    Sessions after the final switch are labelled NaN (censored — we never
    observed another switch, so we can't compute a true label).

    Also computes:
      progress_to_switch  — 1 - (days_to_next_switch / max_days_in_phase)
                            bounded [0,1]; how far through the current phase
      sessions_since_switch — how many sessions since the last switch
    """
    n = len(df)
    days_to_next = np.full(n, np.nan)

    # A switch event is any session where phase_id changed from previous session
    # OR where the session itself started on a different modality than it ended
    # (mid-session switch that constitutes phase progress)
    switch_flags = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if df['phase_id'].iloc[i] != df['phase_id'].iloc[i-1]:
            switch_flags[i] = True

    # For each session i, find the next switch event
    switch_indices = np.where(switch_flags)[0]
    for i in range(n):
        future_switches = switch_indices[switch_indices > i]
        if len(future_switches) > 0:
            days_to_next[i] = future_switches[0] - i
        # else: censored (NaN)

    df = df.copy()
    df['days_to_next_switch'] = days_to_next

    # sessions_since_last_switch
    sessions_since = np.zeros(n, dtype=float)
    last_sw = 0
    for i in range(n):
        if switch_flags[i]:
            last_sw = i
        sessions_since[i] = i - last_sw
    df['sessions_since_switch'] = sessions_since

    # progress_to_switch: how far into the current phase
    # = sessions_since / (sessions_since + days_to_next)
    # bounded [0,1]; gives 1.0 when days_to_next == 0 (switch happening now)
    progress = np.full(n, np.nan)
    for i in range(n):
        dtn = days_to_next[i]
        ss  = sessions_since[i]
        if not np.isnan(dtn):
            total = ss + dtn
            progress[i] = round(ss / total, 4) if total > 0 else 0.0
    df['progress_to_switch'] = progress

    # Flag sessions within 1 session of a switch
    df['switch_imminent'] = (days_to_next <= 1).astype(float)

    return df, switch_flags


# ── Encoding ──────────────────────────────────────────────────────────────────
def encode(df):
    for col in ['session_start_mod', 'session_end_mod', 'last_block_mod']:
        if col in df.columns:
            df[col + '_enc'] = (df[col].astype(str).str.strip() == 'Vision').astype(float)
    return df


# ── Walk-forward CV ───────────────────────────────────────────────────────────
def walk_forward(df, X, target, min_train):
    days, y_true, y_pred = [], [], []
    n = len(df)
    for i in range(min_train, n):
        y_tr = df[target].iloc[:i]
        valid = ~y_tr.isna()
        if valid.sum() < 10:
            continue
        y_te = df[target].iloc[i]
        if np.isnan(y_te):
            continue
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[:i][valid].fillna(-1), y_tr[valid], verbose=False)
        pred = float(np.clip(m.predict(X.iloc[[i]].fillna(-1))[0], 0, None))
        days.append(int(df['day_index'].iloc[i]))
        y_true.append(float(y_te))
        y_pred.append(pred)
    return np.array(days), np.array(y_true), np.array(y_pred)


# ── SHAP plot ─────────────────────────────────────────────────────────────────
def plot_shap(model, X, out_path):
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X.fillna(-1))
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    feat_names = X.columns.tolist()
    order      = np.argsort(mean_shap)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0a0e14')
    for ax in axes:
        ax.set_facecolor('#111620')
        ax.tick_params(colors='#4a6070')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2a3a')

    axes[0].barh(
        [feat_names[i] for i in order[::-1]],
        mean_shap[order[::-1]],
        color='#3b8bd4', height=0.6
    )
    axes[0].set_xlabel('Mean |SHAP value|', color='#4a6070')
    axes[0].set_title('Top 20 features — days to next switch', color='#c8d8e8')
    axes[0].tick_params(labelsize=9, colors='#4a6070')

    top10 = order[:10]
    for rank, fi in enumerate(top10[::-1]):
        sv   = shap_vals[:, fi]
        fv   = X.fillna(-1).iloc[:, fi].values
        fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        axes[1].scatter(
            sv,
            np.full_like(sv, rank) + np.random.normal(0, 0.08, len(sv)),
            c=fv_n, cmap='RdBu_r', alpha=0.6, s=20, vmin=0, vmax=1
        )
    axes[1].set_yticks(range(len(top10)))
    axes[1].set_yticklabels([feat_names[i] for i in top10[::-1]], fontsize=9, color='#4a6070')
    axes[1].axvline(0, color='#2a3a50', linewidth=0.8)
    axes[1].set_xlabel('SHAP value', color='#4a6070')
    axes[1].set_title('Direction of effect', color='#c8d8e8')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor='#0a0e14')
    plt.close()
    print(f"  Saved → {out_path}")


# ── Prediction plot ───────────────────────────────────────────────────────────
def plot_predictions(df, days, y_true, y_pred, switch_flags, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor('#0a0e14')
    fig.suptitle('GR010 — Days-to-next-switch predictions', color='#c8d8e8', fontsize=13)

    day_idx = df['day_index'].values

    for ax in axes:
        ax.set_facecolor('#111620')
        ax.tick_params(colors='#4a6070')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2a3a')
        ax.grid(True, color='#1e2a3a', linewidth=0.5)
        # Phase shading
        last_phase = df['phase_id'].iloc[0]
        seg_start  = day_idx[0]
        for i in range(1, len(df)):
            if df['phase_id'].iloc[i] != last_phase or i == len(df)-1:
                if (df['phase_id'].iloc[max(0,i-1)] % 2) == 0:
                    ax.axvspan(seg_start, day_idx[i], alpha=0.04,
                               color='#ef9f27', linewidth=0)
                ax.axvline(day_idx[i], color='#ef9f27', alpha=0.2,
                           linewidth=0.5, linestyle='--')
                last_phase = df['phase_id'].iloc[i]
                seg_start  = day_idx[i]

    # Panel 1: days to next switch — actual vs predicted
    ax = axes[0]
    ax.plot(day_idx, df['days_to_next_switch'],
            color='#3b8bd4', linewidth=1.5, label='Actual days to switch', zorder=3)
    ax.scatter(days, y_pred, color='#e8943a', s=18, zorder=4,
               label='Predicted', marker='x')
    # Error bars
    for d, yt, yp in zip(days, y_true, y_pred):
        ax.plot([d, d], [yt, yp], color='#e8943a', alpha=0.3, linewidth=0.8)
    # Switch events
    sw_days = day_idx[switch_flags]
    for sd in sw_days:
        ax.axvline(sd, color='#1d9e75', alpha=0.5, linewidth=1.2)
    ax.set_ylabel('Days to next switch', color='#4a6070')
    ax.legend(fontsize=8, facecolor='#111620', labelcolor='#c8d8e8',
              edgecolor='#1e2a3a')
    ax.set_title('Predicted vs actual · green lines = switch events', color='#c8d8e8', fontsize=10)

    # Panel 2: progress to switch
    ax = axes[1]
    ax.fill_between(day_idx, df['progress_to_switch'].fillna(0),
                    alpha=0.25, color='#1d9e75')
    ax.plot(day_idx, df['progress_to_switch'],
            color='#1d9e75', linewidth=1.5, label='Progress to switch')
    ax.axhline(1.0, color='#e8943a', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_ylabel('Progress to switch (0→1)', color='#4a6070')
    ax.set_ylim(0, 1.1)
    ax.set_title('Phase completion progress · 1.0 = switch threshold reached', color='#c8d8e8', fontsize=10)

    # Panel 3: last_block_perf with switch imminent flags
    ax = axes[2]
    ax.plot(day_idx, df['last_block_perf'],
            color='#c8d8e8', linewidth=1.5, label='Last block perf')
    ax.plot(day_idx, df['last_block_perf_roll5'],
            color='#3b8bd4', linewidth=2, label='5-session rolling avg')
    # Highlight imminent switch sessions
    imminent = df[df['switch_imminent'] == 1]
    ax.scatter(imminent['day_index'], imminent['last_block_perf'],
               color='#e8943a', s=40, zorder=5, label='Switch imminent (≤1 day)')
    ax.set_ylabel('Performance', color='#4a6070')
    ax.set_xlabel('Day index', color='#4a6070')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor='#111620', labelcolor='#c8d8e8',
              edgecolor='#1e2a3a')
    ax.set_title('Performance · orange dots = switch expected within 1 session', color='#c8d8e8', fontsize=10)

    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    fig.text(0.99, 0.01, f'Walk-forward MAE: {mae:.2f} sessions  |  R²: {r2:.4f}',
             ha='right', va='bottom', color='#4a6070', fontsize=9,
             fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor='#0a0e14')
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    df = df.sort_values('day_index').reset_index(drop=True)
    df = encode(df)

    print("Computing days_to_next_switch labels ...")
    df, switch_flags = compute_days_to_next_switch(df)

    n_switches = switch_flags.sum()
    n_labelled = df['days_to_next_switch'].notna().sum()
    n_censored = df['days_to_next_switch'].isna().sum()
    print(f"  Switch events detected : {n_switches}")
    print(f"  Labelled sessions      : {n_labelled}")
    print(f"  Censored (post-final)  : {n_censored}")

    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved labels → {OUT_CSV}\n")

    # Feature matrix
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].copy()

    # Walk-forward CV
    print("Running walk-forward CV on days_to_next_switch ...")
    days, y_true, y_pred = walk_forward(df, X, 'days_to_next_switch', MIN_TRAIN)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"  MAE : {mae:.2f} sessions")
    print(f"  R²  : {r2:.4f}\n")

    # Save predictions
    pred_df = pd.DataFrame({
        'day_index':            days,
        'days_to_next_actual':  y_true,
        'days_to_next_pred':    y_pred,
        'error':                y_pred - y_true,
    })
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"  Saved predictions → {PRED_CSV}")

    # Train final model on all labelled data
    print("Training final model ...")
    valid = ~df['days_to_next_switch'].isna()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X[valid].fillna(-1), df['days_to_next_switch'][valid], verbose=False)
    model.save_model(str(MODEL_PATH))
    print(f"  Saved model → {MODEL_PATH}")

    # SHAP
    print("Computing SHAP values ...")
    plot_shap(model, X[valid], SHAP_PATH)

    # Prediction plot
    print("Generating prediction plots ...")
    plot_predictions(df, days, y_true, y_pred, switch_flags,
                     Path('GR010_switch_trajectory.png'))

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Target : days_to_next_switch")
    print(f"  MAE    : {mae:.2f} sessions  (model is off by ~{mae:.1f} days on average)")
    print(f"  R²     : {r2:.4f}")

if __name__ == '__main__':
    main()
