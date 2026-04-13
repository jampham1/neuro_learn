"""
GR010 Learning Trajectory — XGBoost Predictor + SHAP Analysis
=============================================================
Trains two XGBoost regressors:
  1. Predict global_mastery     (0–100% overall progress)
  2. Predict within_phase_mastery (0–100% within current phase)

Uses walk-forward cross-validation (train on days 1..N, predict N+1)
so evaluation is honest — no future data leaks into training.

Outputs:
  model_global.json          — saved XGBoost model (global mastery)
  model_phase.json           — saved XGBoost model (within-phase mastery)
  GR010_predictions.csv      — predicted vs actual for every session
  shap_global.png            — SHAP feature importance (global mastery)
  shap_phase.png             — SHAP feature importance (within-phase mastery)
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
CSV_PATH      = Path('../GR010_features.csv')
OUT_DIR       = Path('..')
MIN_TRAIN_DAYS = 20   # minimum sessions before we start predicting
# ──────────────────────────────────────────────────────────────────────────────

# ── Feature columns to use ────────────────────────────────────────────────────
FEATURE_COLS = [
    # Session structure
    'day_index', 'session_in_phase', 'phase_id', 'n_modality_switches',
    'n_trials', 'engage_rate',

    # Raw performance
    'perf_final', 'perf_mean', 'perf_slope', 'perf_var',
    'l_perf_final', 'l_perf_mean', 'l_perf_slope',
    'r_perf_final', 'r_perf_mean', 'r_perf_slope',
    'perf_asymmetry',

    # Modality-aware performance
    'last_block_perf', 'pre_switch_perf', 'post_switch_perf', 'switch_perf_drop',

    # Reaction time
    'rt_median', 'rt_mean', 'rt_var', 'rt_impulsive_frac', 'rt_slow_frac',

    # Psychometric
    'psychometric_slope', 'psychometric_threshold', 'psychometric_lapse',

    # Lag features (previous sessions)
    'perf_final_lag1', 'perf_final_lag2', 'perf_final_delta',
    'last_block_perf_lag1', 'last_block_perf_lag2', 'last_block_perf_delta',
    'global_mastery_lag1', 'global_mastery_lag2', 'global_mastery_delta',
    'within_phase_mastery_lag1', 'within_phase_mastery_delta',
    'rt_median_lag1', 'rt_median_delta',

    # Rolling averages
    'perf_final_roll5', 'last_block_perf_roll5',

    # Flags
    'single_session_phase', 'opto_session', 'assisted', 'auto_reward',
]

TARGETS = {
    'global_mastery':       'model_global.json',
    'within_phase_mastery': 'model_phase.json',
}

XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values('day_index').reset_index(drop=True)

    # Encode modality as binary (Vision=1, Audio=0)
    for col in ['session_start_mod', 'session_end_mod', 'last_block_mod']:
        if col in df.columns:
            df[col + '_enc'] = (df[col].str.strip() == 'Vision').astype(float)

    return df


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    extra = [c for c in df.columns if c.endswith('_enc')]
    cols  = [c for c in FEATURE_COLS + extra if c in df.columns]
    return df[cols].copy()


def walk_forward_cv(df: pd.DataFrame, X: pd.DataFrame,
                    target: str, min_train: int):
    """
    Walk-forward validation: for each day N >= min_train,
    train on days 0..N-1, predict day N.
    Returns arrays of (day_index, y_true, y_pred).
    """
    days, y_true, y_pred = [], [], []
    n = len(df)

    for i in range(min_train, n):
        X_train = X.iloc[:i].copy()
        y_train = df[target].iloc[:i].copy()
        X_test  = X.iloc[[i]].copy()
        y_test  = df[target].iloc[i]

        # Drop rows where target is NaN
        valid = ~y_train.isna()
        if valid.sum() < 10 or np.isnan(y_test):
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train[valid].fillna(-1),
            y_train[valid],
            verbose=False
        )
        pred = float(model.predict(X_test.fillna(-1))[0])
        pred = float(np.clip(pred, 0, 100))

        days.append(int(df['day_index'].iloc[i]))
        y_true.append(float(y_test))
        y_pred.append(pred)

    return np.array(days), np.array(y_true), np.array(y_pred)


def train_final_model(X: pd.DataFrame, y: pd.Series,
                      model_path: Path) -> xgb.XGBRegressor:
    """Train on all available data and save."""
    valid = ~y.isna()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X[valid].fillna(-1), y[valid], verbose=False)
    model.save_model(str(model_path))
    print(f"  Saved model → {model_path}")
    return model


def plot_shap(model, X: pd.DataFrame, target_name: str, out_path: Path):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X.fillna(-1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'SHAP analysis — {target_name}', fontsize=13)

    # Bar plot: mean |SHAP|
    mean_shap = np.abs(shap_vals).mean(axis=0)
    feat_names = X.columns.tolist()
    order = np.argsort(mean_shap)[::-1][:20]
    axes[0].barh(
        [feat_names[i] for i in order[::-1]],
        mean_shap[order[::-1]],
        color='#378ADD', height=0.6
    )
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].set_title('Top 20 features by importance')
    axes[0].tick_params(labelsize=9)

    # Beeswarm-style scatter: SHAP value vs feature value for top 10
    top10 = order[:10]
    ax = axes[1]
    for rank, fi in enumerate(top10[::-1]):
        sv   = shap_vals[:, fi]
        fv   = X.fillna(-1).iloc[:, fi].values
        fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        scatter = ax.scatter(
            sv,
            np.full_like(sv, rank) + np.random.normal(0, 0.08, len(sv)),
            c=fv_n, cmap='RdBu_r', alpha=0.6, s=20, vmin=0, vmax=1
        )
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([feat_names[i] for i in top10[::-1]], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('SHAP value (impact on prediction)')
    ax.set_title('Top 10 features: direction of effect')
    plt.colorbar(scatter, ax=ax, label='Feature value (low→high)')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved SHAP plot → {out_path}")


def plot_predictions(results: dict, df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('GR010 — Walk-forward predictions vs actual', fontsize=13)

    colors = {'global_mastery': '#378ADD', 'within_phase_mastery': '#1D9E75'}
    labels = {'global_mastery': 'Global mastery %',
              'within_phase_mastery': 'Within-phase mastery %'}

    for ax, (target, (days, y_true, y_pred)) in zip(axes, results.items()):
        ax.plot(df['day_index'], df[target], color='#888780',
                linewidth=1.5, alpha=0.6, label='Actual (all sessions)')
        ax.scatter(days, y_true,  color=colors[target], s=18,
                   zorder=3, label='Actual (predicted sessions)')
        ax.scatter(days, y_pred,  color='#D85A30', s=18, marker='x',
                   zorder=4, label='Predicted')
        for d, yt, yp in zip(days, y_true, y_pred):
            ax.plot([d, d], [yt, yp], color='#D85A30', alpha=0.3,
                    linewidth=0.8)

        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        ax.set_ylabel(labels[target])
        ax.set_title(f'{labels[target]}  |  MAE={mae:.2f}  R²={r2:.3f}')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        # Mark phase boundaries
        phase_starts = df.groupby('phase_id')['day_index'].min()
        for ps in phase_starts[1:]:
            ax.axvline(ps, color='#EF9F27', linewidth=0.6,
                       alpha=0.5, linestyle='--')

    axes[-1].set_xlabel('Day index')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved prediction plot → {out_path}")


def main():
    print(f"Loading {CSV_PATH} ...")
    df = load_data(CSV_PATH)
    X  = get_features(df)
    print(f"Sessions: {len(df)}  |  Features: {X.shape[1]}\n")

    all_preds = {'day_index': df['day_index'].tolist()}
    results   = {}

    for target, model_file in TARGETS.items():
        print(f"{'='*55}")
        print(f"Target: {target}")
        print(f"{'='*55}")

        # Walk-forward CV
        print("  Running walk-forward cross-validation ...")
        days, y_true, y_pred = walk_forward_cv(
            df, X, target, MIN_TRAIN_DAYS)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"  Walk-forward MAE : {mae:.2f} percentage points")
        print(f"  Walk-forward R²  : {r2:.4f}")
        results[target] = (days, y_true, y_pred)

        # Store predictions
        pred_series = pd.Series(np.nan, index=df.index)
        for d, yp in zip(days, y_pred):
            idx = df.index[df['day_index'] == d]
            if len(idx):
                pred_series[idx[0]] = yp
        all_preds[f'{target}_pred'] = pred_series.tolist()
        all_preds[f'{target}_actual'] = df[target].tolist()

        # Train final model on all data
        print("  Training final model on all sessions ...")
        model = train_final_model(
            X, df[target], OUT_DIR / model_file)

        # SHAP analysis
        print("  Computing SHAP values ...")
        plot_shap(model, X, target,
                  OUT_DIR / f'shap_{target.split("_")[0]}.png')
        print()

    # Save predictions CSV
    pred_df = pd.DataFrame(all_preds)
    pred_df.to_csv(OUT_DIR / 'GR010_predictions.csv', index=False)
    print(f"Saved predictions → GR010_predictions.csv")

    # Combined prediction plot
    plot_predictions(results, df, OUT_DIR / 'GR010_learning_trajectory.png')

    # Summary table
    print("\n── Summary ──────────────────────────────────────────")
    print(f"{'Target':<26}  {'MAE':>8}  {'R²':>8}")
    print(f"{'-'*26}  {'-'*8}  {'-'*8}")
    for target, (days, y_true, y_pred) in results.items():
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"{target:<26}  {mae:>8.2f}  {r2:>8.4f}")


if __name__ == '__main__':
    main()