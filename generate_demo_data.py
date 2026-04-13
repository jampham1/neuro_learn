"""
Generate demo data for mouse GR011 — 80 sessions of training
plus 10 future sessions for testing run_daily.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

def sigmoid(x, L=1.0, k=0.15, x0=20):
    return L / (1 + np.exp(-k * (x - x0)))

def generate_sessions(n=80, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Phase structure: Vision phase 1 (sessions 1-22), then alternating
    phase_boundaries = [0, 22, 30, 38, 44, 50, 55, 60, 64, 68, 71, 74, 77, 80]
    phase_mods       = ['Vision','Audio','Vision','Audio','Vision','Audio',
                        'Vision','Audio','Vision','Audio','Vision','Audio','Vision','Audio']

    def get_phase(s):
        for i in range(len(phase_boundaries)-1):
            if phase_boundaries[i] <= s < phase_boundaries[i+1]:
                return i+1, phase_mods[i], phase_mods[i+1] if i+1 < len(phase_mods) else phase_mods[i]
        return len(phase_boundaries)-1, phase_mods[-1], phase_mods[-1]

    prev_end_mod = 'Vision'
    for s in range(1, n+1):
        phase_id, start_mod, end_mod = get_phase(s-1)

        # Base learning performance — sigmoid rising within each phase
        phase_start = phase_boundaries[phase_id-1]
        phase_s     = s - phase_start
        base_perf   = sigmoid(phase_s, L=0.92, k=0.25, x0=6) + rng.normal(0, 0.04)
        base_perf   = float(np.clip(base_perf, 0.1, 1.0))

        # Switches per session — increases over time
        if s < 22:
            n_switches = 0
        elif s < 40:
            n_switches = rng.integers(0, 2)
        elif s < 60:
            n_switches = rng.integers(1, 3)
        else:
            n_switches = rng.integers(2, 5)

        # Performance asymmetry between spouts
        l_bias = rng.uniform(-0.15, 0.15)
        l_perf = float(np.clip(base_perf + l_bias + rng.normal(0, 0.03), 0.05, 1.0))
        r_perf = float(np.clip(base_perf - l_bias + rng.normal(0, 0.03), 0.05, 1.0))

        # RT decreases as mouse becomes expert
        rt_base = 0.22 - sigmoid(s, L=0.15, k=0.06, x0=30)
        rt_med  = float(np.clip(rt_base + rng.normal(0, 0.02), 0.04, 0.5))

        # Phase & session-in-phase
        sip = s - phase_boundaries[phase_id-1]

        # Last block perf (slightly noisier)
        lbp = float(np.clip(base_perf + rng.normal(0, 0.06), 0.05, 1.0))

        # Switch dynamics
        pre_sw  = float(np.clip(base_perf + 0.1 + rng.normal(0, 0.03), 0.0, 1.0)) if n_switches > 0 else np.nan
        post_sw = float(np.clip(base_perf - 0.15 + rng.normal(0, 0.05), 0.0, 1.0)) if n_switches > 0 else np.nan
        sw_drop = float(pre_sw - post_sw) if n_switches > 0 else np.nan

        # Compute days_to_next_switch (look-ahead, simplified)
        next_boundary = next((b for b in phase_boundaries if b > s-1), None)
        dtn = float(next_boundary - (s-1)) if next_boundary else np.nan
        ss  = float(sip - 1)
        prog = ss / (ss + dtn) if (not np.isnan(dtn) and ss + dtn > 0) else 0.0

        n_trials = int(rng.integers(300, 680))

        # Date: start from 2023-06-01
        from datetime import date, timedelta
        base_date = date(2023, 6, 1)
        session_date = (base_date + timedelta(days=s-1)).strftime('%Y%m%d')

        rows.append({
            'day_index':              s,
            'session_num':            s,
            'date':                   session_date,
            'n_trials':               n_trials,
            'n_rewarded':             int(n_trials * lbp * 0.9),
            'n_punished':             int(n_trials * (1-lbp) * 0.7),
            'n_did_not_choose':       int(n_trials * 0.03),
            'n_did_not_lever':        int(n_trials * 0.02),
            'engage_rate':            round(0.95 - rng.uniform(0, 0.05), 3),
            'perf_final':             round(base_perf, 4),
            'perf_mean':              round(base_perf - 0.05, 4),
            'perf_slope':             round(rng.uniform(-0.001, 0.003), 5),
            'perf_var':               round(rng.uniform(0.01, 0.06), 4),
            'l_perf_final':           round(l_perf, 4),
            'l_perf_mean':            round(l_perf - 0.03, 4),
            'l_perf_slope':           round(rng.uniform(-0.001, 0.002), 5),
            'l_perf_var':             round(rng.uniform(0.01, 0.05), 4),
            'r_perf_final':           round(r_perf, 4),
            'r_perf_mean':            round(r_perf - 0.03, 4),
            'r_perf_slope':           round(rng.uniform(-0.001, 0.002), 5),
            'r_perf_var':             round(rng.uniform(0.01, 0.05), 4),
            'perf_asymmetry':         round(l_perf - r_perf, 4),
            'last_block_perf':        round(lbp, 4),
            'pre_switch_perf':        round(pre_sw, 4) if not np.isnan(pre_sw) else np.nan,
            'post_switch_perf':       round(post_sw, 4) if not np.isnan(post_sw) else np.nan,
            'switch_perf_drop':       round(sw_drop, 4) if not np.isnan(sw_drop) else np.nan,
            'targ_stim_mean':         float(rng.choice([45, 90, 135])),
            'dist_stim_mean':         float(rng.choice([5, 18])),
            'stim_contrast':          float(rng.choice([27, 45, 117])),
            'modality':               start_mod,
            'psychometric_slope':     round(rng.uniform(0.02, 0.12), 4),
            'psychometric_threshold': round(rng.uniform(40, 90), 2),
            'psychometric_lapse':     round(rng.uniform(0.02, 0.12), 3),
            'rt_median':              round(rt_med, 5),
            'rt_mean':                round(rt_med + 0.01, 5),
            'rt_var':                 round(rng.uniform(0.001, 0.015), 5),
            'rt_impulsive_frac':      round(float(np.clip(0.6 + sigmoid(s, 0.3, 0.05, 40) + rng.normal(0,0.05), 0, 1)), 3),
            'rt_slow_frac':           round(rng.uniform(0.02, 0.1), 3),
            'opto_session':           0,
            'assisted':               1 if s <= 5 else 0,
            'auto_reward':            1 if s <= 10 else 0,
            'single_spout':           0,
            'animal_weight':          round(rng.uniform(22.0, 26.0), 1),
            'n_modality_switches':    int(n_switches),
            'session_start_mod':      start_mod,
            'session_end_mod':        end_mod if n_switches > 0 else start_mod,
            'last_block_mod':         end_mod if n_switches > 0 else start_mod,
            'phase_id':               int(phase_id),
            'session_in_phase':       int(sip),
            'last_block_perf_roll5':  round(lbp, 4),
            'single_session_phase':   0,
            'days_to_next_switch':    round(dtn, 1) if not np.isnan(dtn) else np.nan,
            'sessions_since_switch':  float(ss),
            'progress_to_switch':     round(prog, 4),
            'switch_imminent':        1.0 if (not np.isnan(dtn) and dtn <= 1) else 0.0,
            'session_start_mod_enc':  1.0 if start_mod == 'Vision' else 0.0,
            'session_end_mod_enc':    1.0 if (end_mod if n_switches > 0 else start_mod) == 'Vision' else 0.0,
            'last_block_mod_enc':     1.0 if (end_mod if n_switches > 0 else start_mod) == 'Vision' else 0.0,
        })

    df = pd.DataFrame(rows)
    # Add lag features
    for col in ['perf_final', 'l_perf_final', 'r_perf_final', 'rt_median',
                'n_trials', 'last_block_perf']:
        df[f'{col}_lag1']  = df[col].shift(1)
        df[f'{col}_lag2']  = df[col].shift(2)
        df[f'{col}_delta'] = df[col].diff()
    for col in ['perf_final', 'last_block_perf']:
        df[f'{col}_roll5'] = df[col].rolling(5, min_periods=2).mean()
    return df


# ── Saves into the current working directory — run from your project root ─────
PROJECT_DIR = Path('.')

# Generate 80 historical sessions
df_hist = generate_sessions(80)
hist_path = PROJECT_DIR / 'GR011_switch_labels.csv'
df_hist.to_csv(hist_path, index=False)
print(f"Saved {hist_path} — {len(df_hist)} sessions")

# Generate 10 future sessions (days 81-90)
df_future = generate_sessions(90).iloc[80:]
df_future = df_future.reset_index(drop=True)
df_future['day_index'] = range(81, 91)

# Save each as a separate CSV to simulate daily additions
out_dir = PROJECT_DIR / 'future_sessions'
out_dir.mkdir(exist_ok=True)
for i, (_, row) in enumerate(df_future.iterrows()):
    fname = out_dir / f"GR011_session_{81+i:03d}_{int(row['date'])}.csv"
    pd.DataFrame([row]).to_csv(fname, index=False)
    print(f"  Saved future session {81+i}: {fname.name}")

print(f"\nDone. To test the daily monitor:")
print(f"  1. GR011_switch_labels.csv is now in your project folder")
print(f"  2. Each day, run:")
print(f"       python append_future_session.py future_sessions/GR011_session_081_20230820.csv")
print(f"       python daily_monitor.py")
print(f"  3. Repeat with sessions 082, 083 ... 090")