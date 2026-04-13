"""
embed_predictions.py
====================
After running predict_gr011.py, run this script to bake the model
predictions into index.html so the website shows predicted DTN
instead of look-forward synthetic labels.

Usage:
    python predict_gr011.py          # generates GR011_predictions.csv
    python embed_predictions.py      # updates index.html with predictions

What changes in the HTML:
  - days_to_next_switch for each session is replaced with dtn_predicted
  - A note in the stat card clarifies it is a model prediction
  - The chart tooltip shows "Predicted" rather than implying ground truth
"""

import json, re, sys
import pandas as pd
from pathlib import Path

PRED_CSV   = Path('GR011_predictions.csv')
LABELS_CSV = Path('GR011_switch_labels.csv')
HTML_PATH  = Path('index.html')

# ── Load data ──────────────────────────────────────────────────────────────────
if not PRED_CSV.exists():
    sys.exit("ERROR: GR011_predictions.csv not found. Run predict_gr011.py first.")
if not LABELS_CSV.exists():
    sys.exit("ERROR: GR011_switch_labels.csv not found.")
if not HTML_PATH.exists():
    sys.exit("ERROR: index.html not found.")

preds  = pd.read_csv(PRED_CSV)[['day_index', 'dtn_predicted']].set_index('day_index')
labels = pd.read_csv(LABELS_CSV).sort_values('day_index').reset_index(drop=True)

# Merge predictions in, replacing days_to_next_switch
labels['days_to_next_switch'] = labels['day_index'].map(preds['dtn_predicted'])

# Also update switch_imminent based on predicted DTN
labels['switch_imminent'] = (labels['days_to_next_switch'] <= 1).astype(float)

# ── Build the JS data array ────────────────────────────────────────────────────
keep_cols = [
    'day_index', 'date', 'phase_id', 'session_in_phase',
    'n_modality_switches', 'session_start_mod', 'session_end_mod',
    'last_block_perf', 'perf_final', 'l_perf_final', 'r_perf_final',
    'rt_median', 'rt_impulsive_frac',
    'days_to_next_switch',   # now = dtn_predicted
    'progress_to_switch', 'sessions_since_switch', 'switch_imminent',
    'n_trials', 'engage_rate', 'switch_perf_drop', 'perf_asymmetry',
    'animal_weight',
]
keep_cols = [c for c in keep_cols if c in labels.columns]
subset = labels[keep_cols].where(pd.notnull(labels[keep_cols]), None)
new_data_js = f"const GR011 = {json.dumps(subset.to_dict(orient='records'))};"

# ── Patch the HTML ─────────────────────────────────────────────────────────────
html = HTML_PATH.read_text()

# Replace the GR011 data array
pattern = r'const GR011 = \[.*?\];'
if not re.search(pattern, html, flags=re.DOTALL):
    sys.exit("ERROR: Could not find 'const GR011 = [...]' in index.html.")

# Also add the rolling average recompute line after the data if not present
roll5_line = """
// Compute 5-session rolling average in JS
GR011.forEach((r, i) => {
  const win = GR011.slice(Math.max(0, i - 4), i + 1)
    .map(x => x.last_block_perf).filter(v => v != null);
  r.last_block_perf_roll5 = win.reduce((a, b) => a + b, 0) / win.length;
});"""

new_block = new_data_js
if 'last_block_perf_roll5' not in new_data_js:
    new_block += roll5_line

html = re.sub(pattern, new_block, html, flags=re.DOTALL)

# Update the stat card sub-label to clarify it is a model prediction
html = html.replace(
    '<div class="sub">expected sessions</div>',
    '<div class="sub">model prediction (GR010)</div>'
)

# Update tooltip label from generic to "Predicted DTN"
html = html.replace(
    "['Days to switch',r.days_to_next_switch!=null?r.days_to_next_switch.toFixed(1):'—']",
    "['Predicted DTN',r.days_to_next_switch!=null?r.days_to_next_switch.toFixed(1)+' sess':'—']"
)

HTML_PATH.write_text(html)

n_preds = preds['dtn_predicted'].notna().sum()
print(f"✓ Embedded {n_preds} model predictions into index.html")
print(f"  days_to_next_switch now reflects dtn_predicted from GR010-trained model")
print(f"  Stat card updated: 'model prediction (GR010)'")
print(f"  Tooltip updated: 'Predicted DTN'")
print(f"\nOpen index.html in your browser to see the result.")
