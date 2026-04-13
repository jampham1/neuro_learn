# NeuroLearn — Mouse Learning Trajectory Predictor

A machine learning pipeline for predicting when a mouse will reach criterion
and trigger a modality switch during visual/auditory discrimination training.
Built for the V1/ACC lab's SpatialSparrow task.

---

## What it does

Each day after a training session, the pipeline reads raw behavioral `.mat`
files, extracts session-level features, and runs a trained XGBoost model to
predict how many sessions remain until the mouse hits criterion and the
protocol triggers the next modality switch. If anything looks unusual —
a performance drop, an imminent switch, or an expert-level session — the
daily monitor flags it for the experimenter.

---

## How it works

The prediction target is `days_to_next_switch`: a finite, protocol-native
label computed by looking forward to the next modality switch event. This
avoids the ambiguity of percentage-based mastery scores and produces
predictions that are directly actionable ("switch expected in 1 session").

The model is an XGBoost regressor trained on 135 sessions from mouse GR010
using walk-forward cross-validation — no future data leaks into predictions.
Top predictive features are recent performance trends, within-phase progress,
and reaction time dynamics.



## Setup

```bash
# Clone the repo and create a virtual environment
git clone <repo-url>
cd v1_acc
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

**Python 3.11+ recommended.**

---

## Workflow

### 1. Extract features from raw GR010 data
```bash
python extract_features.py
# → GR010_features.csv
```

### 2. Train the model on GR010
```bash
python train_switch_model.py
# → model_switch.json
# → GR010_switch_labels.csv
# → shap_switch.png
```

### 3. Daily monitoring for a new mouse (GR011)
```bash
# After each session, run the full pipeline:
python run_daily.py
# Auto-detects new .mat files, appends features, runs monitor, prints flags
```

### 4. Generate model predictions and update the website
```bash
python predict_gr011.py
# → GR011_predictions.csv
# → GR011_pred_chart.png
# → index.html updated automatically
```

---

## Demo workflow (synthetic GR011 data)

To test the daily pipeline without real data:

```bash
# Generate 80-session demo mouse + 10 future sessions
python generate_demo_data.py

# Simulate a new day arriving — add one session at a time
python append_future_session.py future_sessions/GR011_session_081_20230820.csv
python daily_monitor.py

# Repeat for sessions 082–090
python append_future_session.py future_sessions/GR011_session_082_20230821.csv
python daily_monitor.py

# Reset back to 80 sessions at any time
python -c "
import pandas as pd
df = pd.read_csv('GR011_switch_labels.csv')
df[df.day_index <= 80].to_csv('GR011_switch_labels.csv', index=False)
print('Reset to 80 sessions')
"
```

## Key design decisions

**Why `days_to_next_switch` instead of mastery %?**
Mastery percentage requires knowing the total number of phases in advance,
which is only knowable in hindsight. `days_to_next_switch` is computed
directly from protocol events, resets cleanly at each phase boundary, and
produces predictions that are immediately actionable.

**Why walk-forward cross-validation?**
Standard k-fold CV would allow future sessions to inform predictions of
earlier ones — an unrealistic evaluation for a real-time forecasting system.
Walk-forward CV trains only on past sessions and predicts forward, giving an
honest estimate of live performance.

**Why XGBoost over an LSTM?**
135 sessions from one mouse is too few for deep learning to generalise
reliably. XGBoost with hand-engineered lag and delta features achieves
competitive accuracy and is fully interpretable via SHAP values.

---

## Requirements

See `requirements.txt`. Core dependencies:

- `numpy`, `pandas`, `scipy` — data handling
- `xgboost` — model training and inference
- `scikit-learn` — walk-forward CV, metrics
- `shap` — feature importance
- `matplotlib` — plots
