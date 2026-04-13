"""
GR010 Daily Pipeline — run this once after each training session
================================================================
1. Finds today's new .mat file
2. Extracts its features and appends to the master CSV
3. Recomputes labels (days_to_next_switch etc.)
4. Runs the monitor and prints the experimenter report

Usage:
    python run_daily.py
    python run_daily.py --mat /path/to/specific/file.mat   (override auto-detect)
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io
import xgboost as xgb
from pathlib import Path
from datetime import datetime, date

# Import feature extraction functions from extract_features.py
import importlib.util, sys

def load_module(path):
    spec = importlib.util.spec_from_file_location("extract_features", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path('/Users/jamespham/PycharmProjects/v1_acc/data/GR010')
FEATURES_CSV   = Path('GR010_switch_labels.csv')
MODEL_PATH     = Path('model_switch.json')
EXTRACTOR_PATH = Path('extract_features.py')
MIN_TRIALS     = 200
EXCLUDE        = {3, 5, 166}

UNDERPERFORM_THRESHOLD = 0.15
SWITCH_IMMINENT_DAYS   = 1
PERFORMANCE_FLOOR      = 0.45

FEATURE_COLS = [
    'day_index', 'session_in_phase', 'phase_id', 'n_modality_switches',
    'n_trials', 'engage_rate',
    'perf_final', 'perf_mean', 'perf_slope', 'perf_var',
    'l_perf_final', 'l_perf_mean', 'l_perf_slope',
    'r_perf_final', 'r_perf_mean', 'r_perf_slope', 'perf_asymmetry',
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

DIVIDER = '─' * 58
THICK   = '═' * 58

def color(text, code):
    codes = {'red':'\033[91m','green':'\033[92m','yellow':'\033[93m',
             'blue':'\033[94m','bold':'\033[1m','reset':'\033[0m'}
    return f"{codes.get(code,'')}{text}{codes['reset']}"

def pct(v, d=1):
    return f"{float(v)*100:.{d}f}%" if v is not None and not np.isnan(float(v)) else "N/A"

# ── Step 1: Find today's new .mat file ────────────────────────────────────────
def find_new_mat(data_root, existing_csv):
    """Find .mat files not yet in the CSV."""
    existing_dates = set()
    if existing_csv.exists():
        df = pd.read_csv(existing_csv)
        existing_dates = set(df['date'].astype(str))

    candidates = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue
        date_str = folder.name[:8]
        sparrow  = folder / 'SpatialSparrow'
        if not sparrow.exists():
            continue
        for mat in sorted(sparrow.glob('*.mat')):
            if date_str not in existing_dates:
                candidates.append((date_str, mat))

    return candidates


# ── Step 2: Extract + append new session ─────────────────────────────────────
def append_new_session(mat_path, existing_csv, extractor):
    sd = extractor.load_session(mat_path)
    n  = int(sd.get('nTrials', 0))
    if n < MIN_TRIALS:
        print(f"  Skipping {mat_path.name} — only {n} trials")
        return False

    date_str = mat_path.parent.parent.name[:8]

    # Load existing CSV to get next session_num and day_index
    if existing_csv.exists():
        df_existing = pd.read_csv(existing_csv)
        next_session = int(df_existing['session_num'].max()) + 1
    else:
        df_existing  = pd.DataFrame()
        next_session = 1

    try:
        feats = extractor.extract_session_features(sd, next_session, date_str)
    except Exception as e:
        print(color(f"  Feature extraction failed: {e}", 'red'))
        return False

    new_row = pd.DataFrame([feats])

    # Append and recompute lags/rolling features
    df_combined = pd.concat([df_existing, new_row], ignore_index=True)
    df_combined['day_index'] = range(1, len(df_combined) + 1)

    for col in ['perf_final', 'l_perf_final', 'r_perf_final',
                'rt_median', 'n_trials', 'last_block_perf']:
        if col in df_combined.columns:
            df_combined[f'{col}_lag1']  = df_combined[col].shift(1)
            df_combined[f'{col}_lag2']  = df_combined[col].shift(2)
            df_combined[f'{col}_delta'] = df_combined[col].diff()
    for col in ['perf_final', 'last_block_perf']:
        if col in df_combined.columns:
            df_combined[f'{col}_roll5'] = df_combined[col].rolling(5, min_periods=2).mean()

    # Recompute phase labels
    df_combined = recompute_phases(df_combined, extractor)

    df_combined.to_csv(existing_csv, index=False)
    print(color(f"  Appended session {next_session} ({date_str}, n={n}) → {existing_csv}", 'green'))
    return True


def recompute_phases(df, extractor):
    """Re-run phase segmentation and switch labels on full dataframe."""
    # Encode modality
    for col in ['session_start_mod', 'session_end_mod', 'last_block_mod']:
        if col in df.columns:
            df[col + '_enc'] = (df[col].astype(str).str.strip() == 'Vision').astype(float)

    # Phase assignment
    phase_ids, session_in_phase = [], []
    current_phase, count = 1, 0
    prev_end = None
    for _, row in df.iterrows():
        if prev_end is not None and str(row.get('session_start_mod','')).strip() != str(prev_end).strip():
            current_phase += 1
            count = 0
        count += 1
        phase_ids.append(current_phase)
        session_in_phase.append(count)
        prev_end = str(row.get('session_end_mod','')).strip()
    df['phase_id']         = phase_ids
    df['session_in_phase'] = session_in_phase

    # Switch labels
    n = len(df)
    switch_flags = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if df['phase_id'].iloc[i] != df['phase_id'].iloc[i-1]:
            switch_flags[i] = True
    switch_indices = np.where(switch_flags)[0]

    days_to_next   = np.full(n, np.nan)
    sessions_since = np.zeros(n, dtype=float)
    last_sw = 0
    for i in range(n):
        if switch_flags[i]:
            last_sw = i
        sessions_since[i] = i - last_sw
        future = switch_indices[switch_indices > i]
        if len(future) > 0:
            days_to_next[i] = future[0] - i

    df['days_to_next_switch'] = days_to_next
    df['sessions_since_switch'] = sessions_since

    progress = np.full(n, np.nan)
    for i in range(n):
        dtn = days_to_next[i]
        ss  = sessions_since[i]
        if not np.isnan(dtn):
            total = ss + dtn
            progress[i] = round(ss / total, 4) if total > 0 else 0.0
    df['progress_to_switch'] = progress
    df['switch_imminent']    = (days_to_next <= 1).astype(float)

    phase_sizes = df.groupby('phase_id')['session_in_phase'].max()
    df['single_session_phase'] = df['phase_id'].map(phase_sizes == 1).astype(int)

    return df


# ── Step 3: Run monitor on latest session ────────────────────────────────────
def run_monitor(df, model):
    today = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) > 1 else None

    for col in ['session_start_mod', 'session_end_mod', 'last_block_mod']:
        if col in df.columns:
            df[col + '_enc'] = (df[col].astype(str).str.strip() == 'Vision').astype(float)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_today   = df[feat_cols].iloc[[-1]].fillna(-1)
    pred_days = float(np.clip(model.predict(X_today)[0], 0, None))

    ss = float(today.get('sessions_since_switch', 0))
    predicted_progress = ss / (ss + pred_days) if (ss + pred_days) > 0 else 0.0
    actual_progress    = float(today.get('progress_to_switch', np.nan))

    prev_pred_prog = None
    if prev is not None:
        X_prev = df[feat_cols].iloc[[-2]].fillna(-1)
        prev_pred_days = float(np.clip(model.predict(X_prev)[0], 0, None))
        ss_prev = float(prev.get('sessions_since_switch', 0))
        prev_pred_prog = ss_prev / (ss_prev + prev_pred_days) if (ss_prev + prev_pred_days) > 0 else 0.0

    # Recent switch rate (last 10 sessions)
    recent = df.tail(10)
    switch_sessions = (recent['phase_id'].diff().fillna(0) != 0).sum()
    avg_phase_len   = round(10 / switch_sessions, 1) if switch_sessions > 0 else None

    lbp  = today.get('last_block_perf', np.nan)
    roll = today.get('last_block_perf_roll5', np.nan)
    rt   = today.get('rt_median', np.nan)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{THICK}")
    print(color("  GR010 DAILY SESSION MONITOR", 'bold'))
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(THICK)

    print(f"\n  {'DAY INDEX':<30} {int(today['day_index'])}")
    print(f"  {'DATE':<30} {today['date']}")
    print(f"  {'PHASE':<30} {int(today['phase_id'])}  (session {int(today['session_in_phase'])} in phase)")
    print(f"  {'START MODALITY':<30} {today.get('session_start_mod','?')}")
    print(f"  {'END MODALITY':<30} {today.get('session_end_mod','?')}")
    print(f"  {'SWITCHES TODAY':<30} {int(today.get('n_modality_switches',0))}")
    print(f"  {'TRIALS':<30} {int(today.get('n_trials',0))}")

    print(f"\n{DIVIDER}")
    print(color("  PERFORMANCE", 'bold'))
    print(DIVIDER)
    lbp_col = 'green' if not np.isnan(lbp) and lbp >= 0.7 else \
              'yellow' if not np.isnan(lbp) and lbp >= PERFORMANCE_FLOOR else 'red'
    print(f"  {'Last block performance':<30} {color(pct(lbp), lbp_col)}")
    print(f"  {'5-session rolling avg':<30} {pct(roll)}")
    print(f"  {'Median RT':<30} {f'{rt*1000:.1f} ms' if not np.isnan(rt) else 'N/A'}")
    print(f"  {'Engage rate':<30} {pct(today.get('engage_rate', np.nan))}")

    print(f"\n{DIVIDER}")
    print(color("  SWITCH PREDICTION", 'bold'))
    print(DIVIDER)
    print(f"  {'Predicted days to switch':<30} {color(f'{pred_days:.1f} sessions', 'blue')}")
    print(f"  {'Predicted phase progress':<30} {predicted_progress*100:.1f}%")
    print(f"  {'Actual phase progress':<30} {actual_progress*100:.1f}%" if not np.isnan(actual_progress) else f"  {'Actual phase progress':<30} N/A (first session in phase)")
    print(f"  {'Sessions since last switch':<30} {int(ss)}")
    if prev_pred_prog is not None:
        delta = predicted_progress - prev_pred_prog
        print(f"  {'Progress vs yesterday':<30} {color(f'{delta*100:+.1f}%', 'green' if delta >= 0 else 'yellow')}")
    if avg_phase_len is not None:
        note = color(' ← rapid-switch phase', 'yellow') if avg_phase_len <= 2 else ''
        print(f"  {'Recent avg phase length':<30} {avg_phase_len} sessions{note}")

    # ── Flags ─────────────────────────────────────────────────────────────────
    flags = []
    if prev_pred_prog is not None and (prev_pred_prog - predicted_progress) > UNDERPERFORM_THRESHOLD:
        drop = prev_pred_prog - predicted_progress
        flags.append(('red', 'UNDERPERFORMANCE DETECTED',
            f"Progress dropped {drop*100:.1f}% from yesterday's prediction.",
            ["Check water restriction schedule and intake volume",
             "Inspect animal health (weight, grooming, activity level)",
             "Review protocol settings for any recent changes",
             "Consider shortening tomorrow's session if trend continues"]))

    if not np.isnan(lbp) and lbp < PERFORMANCE_FLOOR:
        flags.append(('red', 'PERFORMANCE BELOW FLOOR',
            f"Last block perf ({pct(lbp)}) is below minimum threshold ({pct(PERFORMANCE_FLOOR)}).",
            ["Verify water restriction is correctly applied",
             "Check spout positioning and reward delivery hardware",
             "Review stimulus parameters — consider reducing difficulty temporarily"]))

    if pred_days <= SWITCH_IMMINENT_DAYS or (avg_phase_len is not None and avg_phase_len <= 1.5):
        effective = min(pred_days, avg_phase_len) if avg_phase_len else pred_days
        flags.append(('yellow', 'MODALITY SWITCH EXPECTED',
            f"Switch predicted within {effective:.1f} session(s).",
            [f"Expect modality transition: {str(today.get('session_end_mod','?')).strip()} → next",
             "Monitor trial-by-trial RewardedModality field in TrialSettings",
             "Increase trial count tomorrow if possible to capture full switch event",
             "Note exact switch trial index in lab notebook for neural data alignment",
             "Verify .mat file saves correctly — this is a high-value session"]))

    if int(today.get('n_modality_switches', 0)) >= 3:
        flags.append(('blue', 'EXPERT-LEVEL SESSION',
            f"{int(today.get('n_modality_switches',0))} switches — mouse at peak performance.",
            ["Consider increasing task difficulty (harder stimulus contrast)",
             "High-value session for neural data analysis",
             "Document task parameters carefully for this phase"]))

    print(f"\n{DIVIDER}")
    print(color("  STATUS FLAGS", 'bold'))
    print(DIVIDER)

    if not flags:
        print(color("\n  ✓  All systems nominal. No action required.\n", 'green'))
    else:
        for fc, title, summary, actions in flags:
            print(f"\n  {color(f'▶  {title}', fc)}")
            print(f"     {summary}")
            for a in actions:
                print(f"       • {a}")

    # ── Tomorrow ──────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(color("  TOMORROW'S FORECAST", 'bold'))
    print(DIVIDER)
    effective_days = min(pred_days, avg_phase_len) if avg_phase_len and avg_phase_len < pred_days * 0.7 else pred_days
    remaining = max(0, effective_days - 1)
    if remaining <= 0:
        print(color("  ⚠  Switch likely during tomorrow's session.", 'yellow'))
        print("     Monitor trial-by-trial for RewardedModality change.")
    elif remaining <= 2:
        print(color(f"  →  Switch approaching in ~{remaining:.1f} session(s). Increase monitoring.", 'yellow'))
    else:
        print(color(f"  →  Stable phase. ~{remaining:.1f} sessions to next switch.", 'green'))

    print(f"\n{THICK}\n")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = Path('old_model/GR010_monitor_log.csv')
    log_row  = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'day_index': int(today['day_index']),
        'date': today['date'],
        'phase_id': int(today['phase_id']),
        'session_in_phase': int(today['session_in_phase']),
        'last_block_perf': round(float(lbp), 4) if not np.isnan(lbp) else None,
        'pred_days_to_switch': round(pred_days, 2),
        'pred_progress': round(predicted_progress, 4),
        'actual_progress': round(actual_progress, 4) if not np.isnan(actual_progress) else None,
        'avg_phase_len_recent': avg_phase_len,
        'n_flags': len(flags),
        'flags': '|'.join(f[1] for f in flags) if flags else 'nominal',
    }])
    if log_path.exists():
        log_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_row.to_csv(log_path, index=False)
    print(f"  Log saved → {log_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', type=str, default=None,
                        help='Path to specific .mat file (skip auto-detect)')
    args = parser.parse_args()

    # Load extractor module
    if not EXTRACTOR_PATH.exists():
        print(color(f"ERROR: {EXTRACTOR_PATH} not found in current directory.", 'red'))
        sys.exit(1)
    extractor = load_module(EXTRACTOR_PATH)

    # Load model
    if not MODEL_PATH.exists():
        print(color(f"ERROR: {MODEL_PATH} not found. Run train_switch_model.py first.", 'red'))
        sys.exit(1)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # Find and process new session(s)
    if args.mat:
        mat_files = [(Path(args.mat).parent.parent.name[:8], Path(args.mat))]
    else:
        mat_files = find_new_mat(DATA_ROOT, FEATURES_CSV)

    if not mat_files:
        print(color("No new sessions found since last run.", 'yellow'))
        print("Running monitor on most recent session in CSV...")
    else:
        for date_str, mat_path in mat_files:
            print(f"\nProcessing: {mat_path.name}")
            append_new_session(mat_path, FEATURES_CSV, extractor)

    # Run monitor on updated CSV
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        run_monitor(df, model)
    else:
        print(color("No feature CSV found.", 'red'))

if __name__ == '__main__':
    main()
