"""
End-to-End Inference
====================
Classify the fault cause for a single COMTRADE file.

Processing order
----------------
1. Parse COMTRADE  →  Record
2. determine_protection  →  ProtectionResult
3. detect_fault          →  FaultEvent
4. extract_distance_features  →  DistanceFeatures
5. flatten_features       →  feature dict (same schema as labeled_features.csv)
6. Tier 1 rule engine     →  RuleResult  (if a rule fires → done)
7. Tier 2 PETIR classifier →  label + confidence  (if model loaded)
8. Fallback               →  UNKNOWN  (if no model or non-PETIR Tier 2)

Usage
-----
    python models/predict.py path/to/file.cfg

    # Or import and call:
    from pipeline.models.predict import classify_file
    result = classify_file("path/to/file.cfg")
    print(result)
"""

import sys
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ── path bootstrap so script works when run directly from any cwd ─────────────
_PIPELINE_DIR = Path(__file__).parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from core.comtrade_parser import parse_comtrade
from core.protection_router import determine_protection
from core.fault_detector import detect_fault
from core.feature_extractor import extract_distance_features
from models.rules import apply_rules, RuleResult
from models.train import FEATURE_COLS, encode_reclose

MODEL_PATH = Path(__file__).parent / "petir_tree.pkl"


def _transient_cause_likelihoods(row: dict) -> str:
    """
    Rule-based heuristic to estimate likelihood of each transient cause.
    Returns a formatted string like: PETIR 55% | Layang 25% | Hewan 12% | ...

    Based on domain knowledge of power system protection behaviour.
    NOT a trained model — do not interpret as statistically rigorous.
    """
    dur   = float(row.get("fault_duration_ms", 80) or 80)
    fc    = int(row.get("fault_count", 1) or 1)
    i0i1  = float(row.get("i0_i1_ratio", 0) or 0)
    peak  = float(row.get("peak_current_a", 0) or 0)
    reclose_ok = row.get("reclose_ok", None)

    scores = {"PETIR": 3.0, "Layang-Layang": 1.0, "Hewan": 1.0,
              "Benda Asing": 1.0, "Pohon": 0.5}

    # ── PETIR: short arc, high current, single event ─────────────────────────
    if dur < 60:   scores["PETIR"] *= 2.0
    elif dur < 100: scores["PETIR"] *= 1.4
    elif dur > 400: scores["PETIR"] *= 0.3

    if peak > 10000: scores["PETIR"] *= 2.0
    elif peak > 5000: scores["PETIR"] *= 1.5
    elif 0 < peak < 500: scores["PETIR"] *= 0.6

    if fc == 1: scores["PETIR"] *= 1.3
    elif fc > 3: scores["PETIR"] *= 0.4

    # ── Layang-Layang: kite swings back → multiple sub-faults, medium dur ────
    if fc >= 3: scores["Layang-Layang"] *= 3.0
    elif fc == 2: scores["Layang-Layang"] *= 1.8

    if 50 <= dur <= 350: scores["Layang-Layang"] *= 1.5
    elif dur < 30 or dur > 600: scores["Layang-Layang"] *= 0.4

    # ── Hewan: brief single-phase contact, moderate current ──────────────────
    if dur < 100 and i0i1 > 0.8: scores["Hewan"] *= 2.5
    elif i0i1 > 0.5: scores["Hewan"] *= 1.4

    if fc == 1: scores["Hewan"] *= 1.3
    if 0 < peak < 4000: scores["Hewan"] *= 1.4
    elif peak > 10000: scores["Hewan"] *= 0.3

    # ── Benda Asing: foreign object, often multiple events ───────────────────
    if fc >= 2: scores["Benda Asing"] *= 2.0
    if 80 <= dur <= 500: scores["Benda Asing"] *= 1.4

    # ── Pohon: branch contact — longer duration, may fail reclose ────────────
    if dur > 300: scores["Pohon"] *= 2.5
    if dur > 600: scores["Pohon"] *= 2.0
    if fc > 2: scores["Pohon"] *= 1.8
    if reclose_ok is False: scores["Pohon"] *= 2.0
    if peak > 8000: scores["Pohon"] *= 0.3   # tree rarely causes extreme currents

    total = sum(scores.values())
    parts = sorted(scores.items(), key=lambda x: -x[1])
    return "  |  ".join(f"{k} {v/total*100:.0f}%" for k, v in parts)


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    tier: int           # 1 = rules, 2 = ML, 0 = fallback
    rule_name: str      # populated for Tier 1 hits
    evidence: str
    # Raw feature values for audit
    features: dict


def _load_model() -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _build_feature_vector(row: dict) -> np.ndarray:
    """Build the numpy feature vector expected by the Tier 2 classifier."""
    reclose_enc = encode_reclose(row.get("reclose_successful"))

    di_dt     = float(row.get("di_dt_max", 0) or 0)
    peak_i    = float(row.get("peak_fault_current_a", 0) or 0)

    vec = [
        float(row.get("fault_duration_ms", 0) or 0),
        float(row.get("fault_count", 1) or 1),
        float(row.get("i0_i1_ratio", 0) or 0),
        float(row.get("voltage_sag_depth_pu", 0) or 0),
        np.log1p(max(di_dt, 0)),
        np.log1p(max(peak_i, 0)),
        reclose_enc,
    ]
    return np.array(vec, dtype=float).reshape(1, -1)


def classify_file(cfg_path: str) -> ClassificationResult:
    """
    Classify the fault cause for the COMTRADE file at cfg_path.

    Returns a ClassificationResult with label, confidence, tier, and evidence.
    Raises ValueError if the file cannot be parsed or no fault is detected.
    """
    cfg_path = str(cfg_path)

    # ── Step 1-4: pipeline ────────────────────────────────────────────────────
    record = parse_comtrade(cfg_path)
    if record is None:
        raise ValueError(f"COMTRADE parse failed: {cfg_path}")

    prot = determine_protection(record)
    fault = detect_fault(record)

    if fault is None:
        raise ValueError(f"No fault detected in: {cfg_path}")

    if prot.primary_protection.name == "UNKNOWN":
        # No protection type identified from status channels, but a fault was detected.
        # For transmission line recordings this typically means the DFR only captured
        # analog waveforms / generic trip outputs, not zone-specific relay signals.
        # Attempt waveform-based classification with a caveat.
        _unknown_prot_caveat = (
            " | [CATATAN: Tipe proteksi tidak teridentifikasi dari sinyal digital — "
            "diklasifikasikan berdasarkan analisis gelombang arus/tegangan saja. "
            "Verifikasi manual diperlukan.]"
        )
    elif prot.primary_protection.name != "DISTANCE":
        raise ValueError(
            f"Protection type '{prot.primary_protection.name}' is not supported "
            f"(only DISTANCE relay files are classified)"
        )
    else:
        _unknown_prot_caveat = ""

    feat = extract_distance_features(record, fault, prot)
    if feat is None:
        raise ValueError(f"Feature extraction failed: {cfg_path}")

    # ── Step 5: flatten features ──────────────────────────────────────────────
    row = _flatten(feat, fault)

    # ── Step 6: Tier 1 rules ──────────────────────────────────────────────────
    rule_result: Optional[RuleResult] = apply_rules(row)
    if rule_result is not None:
        return ClassificationResult(
            label=rule_result.label,
            confidence=rule_result.confidence,
            tier=1,
            rule_name=rule_result.rule_name,
            evidence=rule_result.evidence + _unknown_prot_caveat,
            features=row,
        )

    # ── Step 7: Tier 2 ML classifier ─────────────────────────────────────────
    model_bundle = _load_model()
    if model_bundle is not None:
        clf = model_bundle["clf"]
        X = _build_feature_vector(row)
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        if pred == 1:   # transient fault
            confidence = float(proba[1])
            likelihoods = _transient_cause_likelihoods(row)
            return ClassificationResult(
                label="GANGGUAN TRANSIEN",
                confidence=confidence,
                tier=2,
                rule_name="petir_decision_tree",
                evidence=(
                    f"Classifier ML: pola transien terdeteksi (prob={confidence:.0%})  "
                    f"dur={row.get('fault_duration_ms', 0):.0f}ms  "
                    f"fault_count={row.get('fault_count', '?')}  "
                    f"i0/i1={row.get('i0_i1_ratio', 0):.2f}  |  "
                    f"Estimasi penyebab (heuristik): {likelihoods}  |  "
                    f"Catatan: karakteristik gelombang PETIR/Layang/Hewan/Benda Asing serupa "
                    f"— konfirmasi via data cuaca atau inspeksi lapangan."
                    + _unknown_prot_caveat
                ),
                features=row,
            )
        else:
            confidence = float(proba[0])
            return ClassificationResult(
                label="NON-PETIR — PERLU INVESTIGASI",
                confidence=confidence,
                tier=2,
                rule_name="petir_decision_tree_non_petir",
                evidence=(
                    f"Classifier ML: bukan PETIR (prob non-PETIR={confidence:.0%})  "
                    f"dur={row.get('fault_duration_ms', 0):.0f}ms  "
                    f"fault_count={row.get('fault_count', '?')}  "
                    f"— penyebab spesifik belum dapat ditentukan "
                    f"(LAYANG/POHON/HEWAN/BENDA ASING butuh lebih banyak data latih)."
                    + _unknown_prot_caveat
                ),
                features=row,
            )

    # ── Step 8: fallback ──────────────────────────────────────────────────────
    return ClassificationResult(
        label="PERLU INVESTIGASI",
        confidence=0.0,
        tier=0,
        rule_name="no_model",
        evidence=(
            "Tidak ada aturan Tier 1 yang cocok dan model ML belum tersedia — jalankan models/train.py"
            + _unknown_prot_caveat
        ),
        features=row,
    )


def _flatten(feat, fault) -> dict:
    """Minimal flatten of DistanceFeatures + FaultEvent into a row dict."""
    return {
        "fault_duration_ms":    fault.duration_ms,
        "fault_inception_ms":   round(fault.inception_time * 1000, 2),
        "fault_count":          feat.fault_count,
        "faulted_phases":       "+".join(feat.faulted_phases) if feat.faulted_phases else "",
        "trip_type":            feat.trip_type,
        "zone_operated":        feat.zone_operated,
        "reclose_attempted":    feat.reclose_attempted,
        "reclose_successful":   feat.reclose_successful,
        "reclose_time_ms":      feat.reclose_time_ms,
        "di_dt_max":            feat.di_dt_max,
        "di_dt_phase":          feat.di_dt_phase,
        "peak_fault_current_a": feat.peak_fault_current_a,
        "peak_fault_phase":     feat.peak_fault_phase,
        "i0_i1_ratio":          feat.i0_i1_ratio,
        "thd_percent":          feat.thd_percent,
        "inception_angle_degrees": feat.inception_angle_degrees,
        "voltage_sag_depth_pu": feat.voltage_sag_depth_pu,
        "r_x_ratio":            feat.r_x_ratio,
        "z_magnitude_ohms":     feat.z_magnitude_ohms,
        "z_angle_degrees":      feat.z_angle_degrees,
        "station_name":         feat.station_name,
        "relay_model":          feat.relay_model,
        "voltage_kv":           feat.voltage_kv,
        "sampling_rate_hz":     feat.sampling_rate_hz,
        "record_duration_ms":   feat.record_duration_ms,
    }


def _print_result(result: ClassificationResult, cfg_path: str):
    print(f"\n{'='*60}")
    print(f"  File    : {Path(cfg_path).name}")
    print(f"  Label   : {result.label}")
    print(f"  Tier    : {result.tier}  ({result.rule_name})")
    print(f"  Conf.   : {result.confidence:.0%}")
    print(f"  Evidence: {result.evidence}")
    print(f"{'='*60}")
    feats = result.features
    print(f"  Station      : {feats.get('station_name', '-')}")
    print(f"  Relay        : {feats.get('relay_model', '-')}")
    print(f"  Zone         : {feats.get('zone_operated', '-')}")
    print(f"  Trip type    : {feats.get('trip_type', '-')}")
    print(f"  Phases       : {feats.get('faulted_phases', '-')}")
    print(f"  Duration     : {feats.get('fault_duration_ms', 0):.0f} ms")
    print(f"  fault_count  : {feats.get('fault_count', '-')}")
    print(f"  peak_I       : {feats.get('peak_fault_current_a', 0):.0f} A")
    print(f"  i0/i1        : {feats.get('i0_i1_ratio', 0):.3f}")
    print(f"  voltage sag  : {feats.get('voltage_sag_depth_pu', 0):.3f} pu")
    print(f"  Reclose ok   : {feats.get('reclose_successful', '-')}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python models/predict.py path/to/file.cfg")
        sys.exit(1)

    cfg = sys.argv[1]
    try:
        result = classify_file(cfg)
        _print_result(result, cfg)
    except ValueError as e:
        print(f"SKIP: {e}")
        sys.exit(1)
