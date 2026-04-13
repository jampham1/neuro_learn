"""
append_future_session.py
========================
Simulates adding a new daily session to test the monitor pipeline.

Usage:
    python append_future_session.py future_sessions/GR011_session_081_20230820.csv
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

LABELS_CSV = Path('GR011_switch_labels.csv')

if len(sys.argv) < 2:
    print("Usage: python append_future_session.py <future_session.csv>")
    sys.exit(1)

new_row = pd.read_csv(sys.argv[1])
existing = pd.read_csv(LABELS_CSV)
combined = pd.concat([existing, new_row], ignore_index=True)
combined['day_index'] = range(1, len(combined)+1)

# Recompute lag features for new row
for col in ['perf_final','l_perf_final','r_perf_final','rt_median','n_trials','last_block_perf']:
    if col in combined.columns:
        combined[f'{col}_lag1']  = combined[col].shift(1)
        combined[f'{col}_lag2']  = combined[col].shift(2)
        combined[f'{col}_delta'] = combined[col].diff()
for col in ['perf_final','last_block_perf']:
    if col in combined.columns:
        combined[f'{col}_roll5'] = combined[col].rolling(5, min_periods=2).mean()

combined.to_csv(LABELS_CSV, index=False)
day = int(combined['day_index'].iloc[-1])
date = combined['date'].iloc[-1]
lbp  = combined['last_block_perf'].iloc[-1]
print(f"✓ Appended session day {day} ({date}) — last_block_perf={lbp:.3f}")
print(f"  CSV now has {len(combined)} sessions.")
print(f"\nNow run: python daily_monitor.py")
