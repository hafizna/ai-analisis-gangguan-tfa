"""
Tier 2 — PETIR Binary Classifier
=================================
Trains a Decision Tree on labeled_features.csv to distinguish PETIR from
all other fault causes (LAYANG, POHON, HEWAN, KONDUKTOR, BENDA_ASING).

Design rationale
----------------
- Only rows that passed BOTH quality filters (scaling_ok=True, duration_ok=True)
  are used for training.
- Rows already handled by Tier 1 rules (KONDUKTOR with phase change, or failed
  reclose) are excluded, so the classifier only needs to separate what rules
  cannot.
- Binary target: PETIR=1, everything else=0.
- Features chosen for availability and discrimination power:
    fault_duration_ms   — lightning is brief (typically <120ms)
    fault_count         — lightning rarely recurs (usually 1)
    i0_i1_ratio         — lightning has asymmetric ground current
    voltage_sag_depth_pu — lightning causes sharp but shallow sag
    di_dt_max           — lightning has fast front (log-scaled)
    peak_fault_current_a — lightning tends lower peak (log-scaled)
    reclose_enc         — 0=False, 1=True, 0.5=None
- Depth is capped at 3 to avoid overfitting on the tiny non-PETIR set.
- Model is saved as pipeline/models/petir_tree.pkl (joblib).

Run from the pipeline/ directory:
    python models/train.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

FEATURES_CSV = Path(__file__).parent.parent / "data" / "features" / "labeled_features.csv"
MODEL_OUT    = Path(__file__).parent / "petir_tree.pkl"

# Features fed to the classifier
FEATURE_COLS = [
    "fault_duration_ms",
    "fault_count",
    "i0_i1_ratio",
    "voltage_sag_depth_pu",
    "di_dt_max",
    "peak_fault_current_a",
    "reclose_enc",   # engineered below
]

# Tier 1 rule signatures — rows matching these were already classified by rules,
# so we exclude them from the Tier 2 training set.
def is_tier1_handled(row) -> bool:
    """Return True if a Tier 1 rule would fire on this row."""
    fault_count    = int(row.get("fault_count", 1))
    faulted_phases = str(row.get("faulted_phases", ""))
    duration_ms    = float(row.get("fault_duration_ms", 0))
    reclose_ok     = row.get("reclose_successful")
    trip_type      = str(row.get("trip_type", ""))

    phase_count    = faulted_phases.count("+") + 1 if faulted_phases else 1

    # Rule 1 — KONDUKTOR phase change
    if fault_count >= 2 and phase_count == 2 and duration_ms > 80:
        return True
    # Rule 2 — three-pole failed reclose
    reclose_failed = (reclose_ok is False or str(reclose_ok) == "False")
    if reclose_failed and trip_type == "three_pole":
        return True
    # Rule 3 — any failed reclose
    if reclose_failed:
        return True
    return False


def encode_reclose(val) -> float:
    """Map reclose_successful → numeric: True→1, False→0, None/nan→0.5"""
    if val is True or str(val).lower() == "true":
        return 1.0
    if val is False or str(val).lower() == "false":
        return 0.0
    return 0.5   # unknown / not attempted


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Quality filter
    df = df[df["scaling_ok"].astype(str).str.lower() == "true"].copy()
    df = df[df["duration_ok"].astype(str).str.lower() == "true"].copy()

    # Exclude rows that Tier 1 rules already handle
    df = df[~df.apply(is_tier1_handled, axis=1)].copy()

    print(f"After quality + Tier-1 filter: {len(df)} rows")
    print(df["label"].value_counts().to_string())
    print()

    # Engineer reclose feature
    df["reclose_enc"] = df["reclose_successful"].apply(encode_reclose)

    # Log-scale features with large dynamic range
    df["di_dt_max"]              = np.log1p(df["di_dt_max"].fillna(0).clip(lower=0))
    df["peak_fault_current_a"]   = np.log1p(df["peak_fault_current_a"].fillna(0).clip(lower=0))

    # Fill remaining NaN with column median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Binary target
    y = (df["label"] == "PETIR").astype(int)
    X = df[FEATURE_COLS]

    return X, y, df


def train(csv_path: Path = FEATURES_CSV, model_out: Path = MODEL_OUT):
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run batch_extract.py first")
        sys.exit(1)

    X, y, df = load_and_prepare(csv_path)

    n_petir     = int(y.sum())
    n_non_petir = int((y == 0).sum())
    print(f"Training set: {n_petir} PETIR  /  {n_non_petir} non-PETIR")

    if n_non_petir == 0:
        print("ERROR: no non-PETIR rows after filtering — cannot train binary classifier")
        sys.exit(1)

    # Class weights to handle imbalance
    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    # Cross-validation (only if enough samples per class for stratified splits)
    if n_non_petir >= 3:
        n_splits = min(5, n_non_petir)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        print(f"Cross-val F1 ({n_splits}-fold): {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print("Too few non-PETIR samples for cross-validation — skipping CV")

    # Full-data fit
    clf.fit(X, y)

    # Training-set report
    y_pred = clf.predict(X)
    labels_present = sorted(y.unique())
    target_names = ["non-PETIR" if l == 0 else "PETIR" for l in labels_present]
    print("\nTraining-set report:")
    print(classification_report(y, y_pred, labels=labels_present, target_names=target_names))

    # Human-readable tree
    print("Decision tree structure:")
    print(export_text(clf, feature_names=FEATURE_COLS))

    # Feature importances
    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        if imp > 0:
            print(f"  {feat:<30} {imp:.3f}")

    # Save
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"clf": clf, "feature_cols": FEATURE_COLS}, f)
    print(f"\nModel saved -> {model_out}")

    return clf


if __name__ == "__main__":
    train()
