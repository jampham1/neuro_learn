"""
GR010 Session Feature Extractor — with modality-aware mastery labels
=====================================================================
Extracts per-session behavioral features from .mat files and computes
mastery labels that correctly handle mid-session modality switches.

Output: GR010_features.csv
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy.optimize import curve_fit
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT        = Path('//data/GR010')
OUTPUT_CSV       = Path('GR010_features.csv')
MIN_TRIALS       = 200
EXCLUDE_SESSIONS = {3, 5, 166}
RT_TIMEOUT       = 2.99   # WaitForResponse durations >= this are non-responses
# ──────────────────────────────────────────────────────────────────────────────


def load_session(mat_path: Path) -> dict:
    mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    return mat['SessionData']


def safe_arr(val, n_trials: int = None) -> np.ndarray:
    """Return val as flat float64, unpacking packed-bit storage if needed."""
    if val is None:
        return np.array([], dtype=float)
    arr = np.atleast_1d(np.array(val, dtype=float)).flatten()
    if n_trials is not None and len(arr) == 2 * n_trials:
        arr = arr[1::2]
    return arr


def load_perf(sd: dict, key: str) -> np.ndarray:
    v = sd.get(key)
    if v is None:
        return np.array([], dtype=float)
    return np.array(v, dtype=float).flatten()


def compute_slope(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])


def fit_psychometric(stim_values, prob_left):
    def sigmoid(x, alpha, beta):
        return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))
    try:
        valid = ~np.isnan(prob_left) & ~np.isnan(stim_values)
        if valid.sum() < 3:
            return np.nan, np.nan, np.nan
        popt, _ = curve_fit(sigmoid, stim_values[valid], prob_left[valid],
                            p0=[np.median(stim_values), 0.1], maxfev=2000)
        threshold, slope = popt
        lapse = float(np.mean(np.abs(prob_left[valid] -
                                     sigmoid(stim_values[valid], *popt))))
        return float(slope), float(threshold), lapse
    except Exception:
        return np.nan, np.nan, np.nan


def extract_rt(raw_events) -> np.ndarray:
    rts = []
    trials = raw_events.get('Trial', []) if isinstance(raw_events, dict) else raw_events
    for trial in (trials if isinstance(trials, (list, np.ndarray)) else []):
        if not isinstance(trial, dict):
            continue
        states = trial.get('States', {})
        if not isinstance(states, dict):
            continue
        wfr = states.get('WaitForResponse')
        if wfr is None:
            continue
        wfr = np.atleast_1d(wfr).flatten()
        if len(wfr) < 2 or np.isnan(wfr[0]) or np.isnan(wfr[1]):
            continue
        rt = float(wfr[1] - wfr[0])
        if rt < RT_TIMEOUT:
            rts.append(rt)
    return np.array(rts)


def get_trial_modalities(sd: dict) -> np.ndarray:
    """
    Return an array of RewardedModality strings, one per trial,
    extracted from TrialSettings. Returns empty array if unavailable.
    """
    ts = sd.get('TrialSettings')
    if ts is None:
        return np.array([])
    ts = np.atleast_1d(ts)
    mods = []
    for t in ts:
        if isinstance(t, dict):
            mods.append(str(t.get('RewardedModality', 'Unknown')).strip())
        else:
            mods.append('Unknown')
    return np.array(mods)


def get_modality_blocks(mods: np.ndarray) -> list[dict]:
    """
    Split trial modality array into contiguous blocks.
    Returns list of dicts: {modality, start_trial, end_trial, n_trials}
    """
    if len(mods) == 0:
        return []
    blocks = []
    current_mod   = mods[0]
    current_start = 0
    for i in range(1, len(mods)):
        if mods[i] != current_mod:
            blocks.append({
                'modality':    current_mod,
                'start_trial': current_start,
                'end_trial':   i - 1,
                'n_trials':    i - current_start
            })
            current_mod   = mods[i]
            current_start = i
    blocks.append({
        'modality':    current_mod,
        'start_trial': current_start,
        'end_trial':   len(mods) - 1,
        'n_trials':    len(mods) - current_start
    })
    return blocks


def block_performance(rewarded: np.ndarray, correct_side: np.ndarray,
                      response_side: np.ndarray,
                      start: int, end: int) -> float:
    """Fraction correct within a trial range [start, end] inclusive."""
    r = rewarded[start:end + 1]
    valid = ~np.isnan(r)
    if valid.sum() == 0:
        return np.nan
    return float(np.nanmean(r[valid]))


def extract_session_features(sd: dict, session_num: int, date_str: str) -> dict:
    feats = {'session_num': session_num, 'date': date_str}

    n = int(sd.get('nTrials', 0))
    feats['n_trials'] = n

    # ── Boolean outcome arrays ────────────────────────────────────────────────
    rewarded   = safe_arr(sd.get('Rewarded'),    n)
    punished   = safe_arr(sd.get('Punished'),     n)
    did_not_ch = safe_arr(sd.get('DidNotChoose'), n)
    did_not_lv = safe_arr(sd.get('DidNotLever'),  n)

    feats['n_rewarded']       = int(np.nansum(rewarded))
    feats['n_punished']       = int(np.nansum(punished))
    feats['n_did_not_choose'] = int(np.nansum(did_not_ch))
    feats['n_did_not_lever']  = int(np.nansum(did_not_lv))
    feats['engage_rate']      = round(
        1.0 - (feats['n_did_not_choose'] + feats['n_did_not_lever']) / max(n, 1), 4)

    # ── Session-level performance curves ──────────────────────────────────────
    perf  = load_perf(sd, 'Performance')
    lperf = load_perf(sd, 'lPerformance')
    rperf = load_perf(sd, 'rPerformance')

    for prefix, arr in [('perf', perf), ('l_perf', lperf), ('r_perf', rperf)]:
        clean = arr[~np.isnan(arr)]
        feats[f'{prefix}_final'] = float(clean[-1])      if len(clean) > 0 else np.nan
        feats[f'{prefix}_mean']  = float(np.mean(clean)) if len(clean) > 0 else np.nan
        feats[f'{prefix}_slope'] = compute_slope(arr)
        feats[f'{prefix}_var']   = float(np.var(clean))  if len(clean) > 1 else np.nan

    feats['perf_asymmetry'] = (
        feats['l_perf_final'] - feats['r_perf_final']
        if not (np.isnan(feats['l_perf_final']) or np.isnan(feats['r_perf_final']))
        else np.nan)

    # ── Modality blocks ───────────────────────────────────────────────────────
    mods   = get_trial_modalities(sd)
    blocks = get_modality_blocks(mods)

    feats['n_modality_switches'] = max(0, len(blocks) - 1)
    feats['session_start_mod']   = blocks[0]['modality']  if blocks else 'Unknown'
    feats['session_end_mod']     = blocks[-1]['modality'] if blocks else 'Unknown'

    # Performance within each block — last block is the active learning phase
    if len(blocks) > 0:
        last_block = blocks[-1]
        lb_perf = block_performance(rewarded, None, None,
                                    last_block['start_trial'],
                                    last_block['end_trial'])
        feats['last_block_mod']       = last_block['modality']
        feats['last_block_n_trials']  = last_block['n_trials']
        feats['last_block_perf']      = lb_perf
    else:
        feats['last_block_mod']       = 'Unknown'
        feats['last_block_n_trials']  = np.nan
        feats['last_block_perf']      = np.nan

    # Recovery slope: perf in first 20 trials of last block vs preceding block
    if len(blocks) >= 2:
        prev_block = blocks[-2]
        prev_end   = prev_block['end_trial']
        curr_start = blocks[-1]['start_trial']
        curr_end   = blocks[-1]['end_trial']

        # Performance at end of previous block (last 20 trials)
        pre_switch_start = max(prev_block['start_trial'], prev_end - 19)
        pre_switch_perf  = block_performance(rewarded, None, None,
                                             pre_switch_start, prev_end)

        # Performance in first 20 trials of new block
        post_switch_end  = min(curr_start + 19, curr_end)
        post_switch_perf = block_performance(rewarded, None, None,
                                             curr_start, post_switch_end)

        feats['pre_switch_perf']    = pre_switch_perf
        feats['post_switch_perf']   = post_switch_perf
        feats['switch_perf_drop']   = (pre_switch_perf - post_switch_perf
                                       if not (np.isnan(pre_switch_perf) or
                                               np.isnan(post_switch_perf))
                                       else np.nan)
    else:
        feats['pre_switch_perf']  = np.nan
        feats['post_switch_perf'] = np.nan
        feats['switch_perf_drop'] = np.nan

    # ── Stimulus info ─────────────────────────────────────────────────────────
    targ = safe_arr(sd.get('TargStim'), n)
    dist = safe_arr(sd.get('DistStim'), n)
    feats['targ_stim_mean'] = float(np.nanmean(targ)) if len(targ) > 0 else np.nan
    feats['dist_stim_mean'] = float(np.nanmean(dist)) if len(dist) > 0 else np.nan
    feats['stim_contrast']  = feats['targ_stim_mean'] - feats['dist_stim_mean']

    # ── Psychometric curve ────────────────────────────────────────────────────
    stim_side_vals = safe_arr(sd.get('StimSideValues'), n)
    response_side  = safe_arr(sd.get('ResponseSide'),   n)

    if len(stim_side_vals) > 0 and len(response_side) > 0:
        unique_stims = np.unique(stim_side_vals[~np.isnan(stim_side_vals)])
        prob_left = np.array([
            np.nanmean(response_side[stim_side_vals == s] == 0)
            if (stim_side_vals == s).sum() > 0 else np.nan
            for s in unique_stims
        ])
        slope, threshold, lapse = fit_psychometric(unique_stims, prob_left)
    else:
        slope, threshold, lapse = np.nan, np.nan, np.nan

    feats['psychometric_slope']     = slope
    feats['psychometric_threshold'] = threshold
    feats['psychometric_lapse']     = lapse

    # ── Reaction times ────────────────────────────────────────────────────────
    rts = extract_rt(sd.get('RawEvents', {}))
    if len(rts) > 0:
        feats['rt_median']         = float(np.nanmedian(rts))
        feats['rt_mean']           = float(np.nanmean(rts))
        feats['rt_var']            = float(np.nanvar(rts))
        feats['rt_impulsive_frac'] = float(np.mean(rts < 0.2))
        feats['rt_slow_frac']      = float(np.mean(rts > 0.5))
    else:
        for k in ['rt_median', 'rt_mean', 'rt_var', 'rt_impulsive_frac', 'rt_slow_frac']:
            feats[k] = np.nan

    # ── Opto & scaffold flags ─────────────────────────────────────────────────
    feats['opto_session']  = int(np.any(safe_arr(sd.get('optoType',    0), n) != 0))
    feats['assisted']      = int(np.any(safe_arr(sd.get('Assisted',    0), n) > 0))
    feats['auto_reward']   = int(np.any(safe_arr(sd.get('AutoReward',  0), n) > 0))
    feats['single_spout']  = int(np.any(safe_arr(sd.get('SingleSpout', 0), n) > 0))

    # ── Animal weight ─────────────────────────────────────────────────────────
    weight = sd.get('animalWeight', np.nan)
    feats['animal_weight'] = float(weight) if np.isscalar(weight) else np.nan

    return feats


def assign_global_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a global phase_id and session_in_phase based on the
    dominant modality across sessions. A new phase starts when
    the session_start_mod changes from the previous session,
    OR when the previous session ended on a different modality
    than the current session starts on.
    """
    phase_ids         = []
    session_in_phase  = []
    current_phase     = 1
    count_in_phase    = 0
    prev_end_mod      = None

    for _, row in df.iterrows():
        start_mod = row['session_start_mod']
        if prev_end_mod is not None and start_mod != prev_end_mod:
            current_phase  += 1
            count_in_phase  = 0
        count_in_phase += 1
        phase_ids.append(current_phase)
        session_in_phase.append(count_in_phase)
        prev_end_mod = row['session_end_mod']

    df['phase_id']        = phase_ids
    df['session_in_phase']= session_in_phase
    return df


def compute_mastery_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-phase and global mastery scores.

    Within-phase mastery: fit sigmoid to last_block_perf within each phase,
    use asymptote as ceiling. Mastery % = current / asymptote * 100.

    Global mastery: (phases_completed + within_phase_frac) / total_phases * 100
    """
    def sigmoid_fit(x, y):
        def sig(t, L, k, t0):
            return L / (1 + np.exp(-k * (t - t0)))
        clean = ~np.isnan(y)
        if clean.sum() < 4:
            return np.nanmax(y[clean]) if clean.sum() > 0 else np.nan
        try:
            popt, _ = curve_fit(sig, x[clean], y[clean],
                                 p0=[0.9, 0.1, np.median(x[clean])],
                                 bounds=([0, 0, 0], [1.0, 10, max(x[clean]) + 1]),
                                 maxfev=5000)
            return float(popt[0])   # L = asymptote
        except Exception:
            return float(np.nanmax(y[clean]))

    n_phases = df['phase_id'].max()
    within_phase_mastery = np.full(len(df), np.nan)
    global_mastery       = np.full(len(df), np.nan)

    for phase in range(1, n_phases + 1):
        mask = df['phase_id'] == phase
        idx  = df.index[mask]
        x    = df.loc[mask, 'session_in_phase'].values.astype(float)
        y    = df.loc[mask, 'last_block_perf'].values.astype(float)

        asymptote = sigmoid_fit(x, y)

        for i, (xi, yi) in enumerate(zip(x, y)):
            if np.isnan(yi) or np.isnan(asymptote) or asymptote == 0:
                wp = np.nan
            else:
                wp = min(100.0, round((yi / asymptote) * 100, 2))
            within_phase_mastery[idx[i]] = wp

            # Global: completed phases + fractional current phase
            phases_done = phase - 1
            wp_frac     = (wp / 100.0) if not np.isnan(wp) else 0.0
            global_mastery[idx[i]] = round(
                (phases_done + wp_frac) / n_phases * 100, 2)

    df['within_phase_mastery'] = within_phase_mastery
    df['global_mastery']       = global_mastery
    df['n_phases_total']       = n_phases

    # Flag single-session phases — sigmoid can't fit, mastery=100% by default
    phase_sizes = df.groupby('phase_id')['session_in_phase'].max()
    df['single_session_phase'] = df['phase_id'].map(phase_sizes == 1).astype(int)

    return df


def find_mat_files(root: Path):
    entries = []
    counter = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        sparrow_dir = folder / 'SpatialSparrow'
        if not sparrow_dir.exists():
            continue
        for mat_file in sorted(sparrow_dir.glob('*.mat')):
            counter += 1
            entries.append((counter, folder.name[:8], mat_file))
    return entries


def main():
    print(f"Scanning {DATA_ROOT} ...")
    sessions = find_mat_files(DATA_ROOT)
    print(f"Found {len(sessions)} session files.\n")

    rows, skipped, errors = [], 0, 0

    for session_num, date_str, mat_path in sessions:
        if session_num in EXCLUDE_SESSIONS:
            print(f"  [{session_num:03d}] {date_str} — EXCLUDED (buggy session)")
            skipped += 1
            continue
        try:
            sd = load_session(mat_path)
        except Exception as e:
            print(f"  [{session_num:03d}] {date_str} — LOAD ERROR: {e}")
            errors += 1
            continue

        n_trials = int(sd.get('nTrials', 0))
        if n_trials < MIN_TRIALS:
            print(f"  [{session_num:03d}] {date_str} — SKIPPED (n={n_trials} < {MIN_TRIALS})")
            skipped += 1
            continue

        try:
            feats = extract_session_features(sd, session_num, date_str)
            rows.append(feats)
            print(f"  [{session_num:03d}] {date_str} — OK "
                  f"(n={n_trials}, switches={feats['n_modality_switches']}, "
                  f"end_mod={feats['session_end_mod']})")
        except Exception as e:
            print(f"  [{session_num:03d}] {date_str} — EXTRACT ERROR: {e}")
            errors += 1

    print(f"\nKept: {len(rows)}  |  Skipped: {skipped}  |  Errors: {errors}")

    if len(rows) == 0:
        print("No sessions extracted.")
        return

    df = pd.DataFrame(rows)
    df.insert(0, 'day_index', range(1, len(df) + 1))

    # ── Phase segmentation ────────────────────────────────────────────────────
    df = assign_global_phase(df)

    # ── Mastery labels ────────────────────────────────────────────────────────
    df = compute_mastery_labels(df)

    # ── Lag & rolling features ────────────────────────────────────────────────
    for col in ['perf_final', 'l_perf_final', 'r_perf_final',
                'rt_median', 'n_trials', 'last_block_perf',
                'within_phase_mastery', 'global_mastery']:
        if col in df.columns:
            df[f'{col}_lag1']  = df[col].shift(1)
            df[f'{col}_lag2']  = df[col].shift(2)
            df[f'{col}_delta'] = df[col].diff()

    for col in ['perf_final', 'last_block_perf', 'global_mastery']:
        if col in df.columns:
            df[f'{col}_roll5'] = df[col].rolling(5, min_periods=2).mean()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} sessions → {OUTPUT_CSV}")
    print(f"Phases detected: {df['phase_id'].max()}\n")

    preview_cols = [c for c in [
        'day_index', 'date', 'n_modality_switches',
        'session_start_mod', 'session_end_mod',
        'phase_id', 'session_in_phase',
        'last_block_perf', 'within_phase_mastery', 'global_mastery'
    ] if c in df.columns]
    print(df[preview_cols].to_string(index=False))


if __name__ == '__main__':
    main()