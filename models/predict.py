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
from core.fault_detector import detect_fault, extract_soe
from core.feature_extractor import extract_distance_features
from models.rules import apply_rules, RuleResult
from models.train import FEATURE_COLS, encode_reclose

MODEL_PATH = Path(__file__).parent / "petir_tree.pkl"


def _compute_cause_scores(row: dict) -> dict:
    """
    Rule-based heuristic scores for each transient cause.
    Based on domain knowledge of power system protection behaviour.
    NOT a trained model — do not interpret as statistically rigorous.
    """
    dur        = float(row.get("fault_duration_ms", 80) or 80)
    fc         = int(row.get("fault_count", 1) or 1)
    i0i1       = float(row.get("i0_i1_ratio", 0) or 0)
    peak       = float(row.get("peak_fault_current_a", 0) or 0)
    reclose_ok = row.get("reclose_successful")

    scores = {"PETIR": 3.0, "Layang-Layang": 1.0, "Hewan": 1.0,
              "Benda Asing": 1.0, "Pohon": 0.5}

    # ── PETIR: short arc, high current, single event ─────────────────────────
    if dur < 60:    scores["PETIR"] *= 2.0
    elif dur < 100: scores["PETIR"] *= 1.4
    elif dur > 400: scores["PETIR"] *= 0.3

    if peak > 10000:          scores["PETIR"] *= 2.0
    elif peak > 5000:         scores["PETIR"] *= 1.5
    elif 0 < peak < 500:      scores["PETIR"] *= 0.6

    if fc == 1:   scores["PETIR"] *= 1.3
    elif fc > 3:  scores["PETIR"] *= 0.4

    # ── Layang-Layang: kite swings back → multiple sub-faults, medium dur ────
    if fc >= 3:   scores["Layang-Layang"] *= 3.0
    elif fc == 2: scores["Layang-Layang"] *= 1.8

    if 50 <= dur <= 350:            scores["Layang-Layang"] *= 1.5
    elif dur < 30 or dur > 600:     scores["Layang-Layang"] *= 0.4

    # ── Hewan: brief single-phase contact, moderate current ──────────────────
    if dur < 100 and i0i1 > 0.8: scores["Hewan"] *= 2.5
    elif i0i1 > 0.5:             scores["Hewan"] *= 1.4

    if fc == 1:               scores["Hewan"] *= 1.3
    if 0 < peak < 4000:       scores["Hewan"] *= 1.4
    elif peak > 10000:        scores["Hewan"] *= 0.3

    # ── Benda Asing: foreign object, often multiple events ───────────────────
    if fc >= 2:              scores["Benda Asing"] *= 2.0
    if 80 <= dur <= 500:     scores["Benda Asing"] *= 1.4

    # ── Pohon: branch contact — longer duration, may fail reclose ────────────
    if dur > 300:             scores["Pohon"] *= 2.5
    if dur > 600:             scores["Pohon"] *= 2.0
    if fc > 2:                scores["Pohon"] *= 1.8
    if reclose_ok is False:   scores["Pohon"] *= 2.0
    if peak > 8000:           scores["Pohon"] *= 0.3   # tree rarely causes extreme currents

    return scores


def _transient_cause_likelihoods(row: dict) -> str:
    """Returns a formatted string like: PETIR 55% | Layang 25% | Hewan 12% | ..."""
    scores = _compute_cause_scores(row)
    total  = sum(scores.values())
    parts  = sorted(scores.items(), key=lambda x: -x[1])
    return "  |  ".join(f"{k} {v/total*100:.0f}%" for k, v in parts)


_CAUSE_RECOMMENDATIONS = {
    "PETIR": (
        "Rekaman dapat diarsipkan sebagai indikasi gangguan petir. "
        "Bandingkan dengan data cuaca atau rekaman penangkal petir di sekitar jalur untuk konfirmasi."
    ),
    "Layang-Layang": (
        "Periksa area ROW untuk aktivitas layang-layang. "
        "Koordinasikan sosialisasi larangan bermain layang-layang di bawah SUTT dengan masyarakat sekitar jalur."
    ),
    "Hewan": (
        "Inspeksi isolator dan tower di zona gangguan untuk jejak kontak hewan. "
        "Pertimbangkan pemasangan pelindung hewan (bird/animal guard) pada tower yang rawan."
    ),
    "Benda Asing": (
        "Lakukan inspeksi visual tower dan konduktor di zona gangguan untuk benda asing "
        "(plastik, banner, tali, dll). Dokumentasikan untuk pemetaan titik rawan."
    ),
    "Pohon": (
        "Inspeksi vegetasi di sepanjang ROW pada zona gangguan. "
        "Jadwalkan pemangkasan jika terdapat pohon yang mendekati jarak aman konduktor."
    ),
}


def _transient_recommendation(row: dict) -> str:
    """Return a context-aware follow-up recommendation based on the top-scoring cause."""
    scores = _compute_cause_scores(row)
    top    = max(scores, key=scores.get)
    return _CAUSE_RECOMMENDATIONS.get(top, "Verifikasi penyebab melalui data pendukung.")


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    tier: int             # 1 = rules, 2 = ML, 0 = fallback
    rule_name: str        # populated for Tier 1 hits
    evidence: str
    recommendation: str   # follow-up action based on top cause
    # Raw feature values for audit
    features: dict
    # Extended result fields
    soe: list = None                # Sequence of Events from digital channels
    description: str = ""          # Natural language analysis narrative
    cause_pcts: list = None        # [{name, pct}] likelihood bars for transient causes


def _compute_cause_pcts(row: dict) -> list:
    """Return cause likelihoods as [{name, pct}] sorted descending."""
    scores = _compute_cause_scores(row)
    total = sum(scores.values()) or 1
    return sorted(
        [{"name": k, "pct": round(v / total * 100, 1)} for k, v in scores.items()],
        key=lambda x: -x["pct"]
    )


def _generate_description(row: dict, result: "ClassificationResult") -> str:
    """Generate a natural language analysis narrative from features."""
    phases   = row.get("faulted_phases", "-") or "-"
    ftype    = row.get("fault_type", "") or ""
    peak_i   = float(row.get("peak_fault_current_a", 0) or 0)
    sag_pu   = float(row.get("voltage_sag_depth_pu", 0) or 0)
    dur_ms   = float(row.get("fault_duration_ms", 0) or 0)
    zone     = row.get("zone_operated", "") or ""
    reclose  = row.get("reclose_successful")
    trip_t   = row.get("trip_type", "") or ""
    rec_ms   = float(row.get("record_duration_ms", 0) or 0)

    # Fault type description
    if ftype == "SLG":
        fault_desc = f"Gangguan Fasa {phases}-N (Single Line to Ground)"
    elif ftype == "DLG":
        fault_desc = f"Gangguan Fasa {phases}-N (Double Line to Ground)"
    elif ftype == "3PH":
        fault_desc = "Gangguan 3 Fasa (Three Phase)"
    elif phases and phases != "-":
        fault_desc = f"Gangguan Fasa {phases}"
    else:
        fault_desc = "Gangguan terdeteksi"

    sag_pct = sag_pu * 100
    lines = []

    # Line 1: Fault characteristics
    lines.append(
        f"Terdeteksi {fault_desc} dengan kenaikan arus puncak mencapai "
        f"{peak_i/1000:.2f} kA dan penurunan tegangan (voltage sag) sebesar {sag_pct:.1f}%."
    )

    # Line 2: Protection operation
    if zone and zone not in ("-", "UNKNOWN", ""):
        lines.append(
            f"Fungsi proteksi Zona {zone} bekerja mentrigger TRIP "
            f"({'3-fasa' if trip_t == 'three_pole' else '1-fasa'}) "
            f"dengan Fault Clearing Time (FCT) {dur_ms:.0f} ms."
        )
    elif dur_ms > 0:
        lines.append(
            f"Proteksi bekerja mentrigger TRIP dengan Fault Clearing Time (FCT) {dur_ms:.0f} ms."
        )

    # Line 3: Auto-reclose
    if reclose is True or reclose == "True":
        lines.append("Auto Reclose (AR) aktif. Status Reclose: BERHASIL (Line kembali energized).")
    elif reclose is False or reclose == "False":
        lines.append("Auto Reclose (AR) aktif. Status Reclose: GAGAL (CB tetap terbuka / Lockout).")
    else:
        lines.append("Status Auto Reclose (AR) tidak teridentifikasi dari rekaman digital.")

    # Line 4: AI prediction
    conf_pct = result.confidence * 100
    lines.append(
        f"Berdasarkan analisis pola gelombang, AI mengklasifikasikan gangguan ini sebagai "
        f"{result.label} dengan tingkat keyakinan {conf_pct:.0f}%."
    )

    return "\n".join(lines)


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

    # Always extract SOE regardless of protection type
    _soe = extract_soe(record, fault_inception_s=fault.inception_time if fault else None)

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
        _tier1_recs = {
            "KONDUKTOR / KERUSAKAN PERALATAN": (
                "Lakukan inspeksi mekanik pada tower dan konduktor di zona operasi rele. "
                "Periksa kondisi joint, klem, dan struktur tower."
            ),
            "GANGGUAN PERMANEN": (
                "Periksa kondisi jalur transmisi di zona gangguan. "
                "Verifikasi rekaman AR dan data operasi sebelum memastikan penyebab."
            ),
        }
        _r1 = ClassificationResult(
            label=rule_result.label,
            confidence=rule_result.confidence,
            tier=1,
            rule_name=rule_result.rule_name,
            evidence=rule_result.evidence + _unknown_prot_caveat,
            recommendation=_tier1_recs.get(
                rule_result.label,
                "Kumpulkan data pendukung untuk menentukan penyebab gangguan."
            ),
            features=row,
            soe=_soe,
            cause_pcts=[],
        )
        _r1.description = _generate_description(row, _r1)
        return _r1

    # ── Step 6.5: Confirmed-transient shortcut ────────────────────────────────
    # If auto-reclose SUCCEEDED and a real fault current was present, the fault
    # cleared and the line recovered → definitively transient.  No ML needed.
    # Guard: peak_i > 200A to exclude dead-time / remote-end recordings.
    _reclose_ok = row.get("reclose_successful")
    _peak_i     = float(row.get("peak_fault_current_a", 0) or 0)
    if (_reclose_ok is True or _reclose_ok == "True") and _peak_i > 200:
        likelihoods = _transient_cause_likelihoods(row)
        _r0 = ClassificationResult(
            label="GANGGUAN TRANSIEN",
            confidence=0.95,
            tier=1,
            rule_name="reclose_confirmed_transient",
            evidence=(
                f"AR berhasil — gangguan transien terkonfirmasi.  "
                f"peak_i={_peak_i:.0f}A  "
                f"dur={row.get('fault_duration_ms', 0):.0f}ms  "
                f"fault_count={row.get('fault_count', '?')}  |  "
                f"Estimasi penyebab (heuristik): {likelihoods}"
                + _unknown_prot_caveat
            ),
            recommendation=_transient_recommendation(row),
            features=row,
            soe=_soe,
            cause_pcts=_compute_cause_pcts(row),
        )
        _r0.description = _generate_description(row, _r0)
        return _r0

    # ── Step 7: Tier 2 ML classifier ─────────────────────────────────────────
    model_bundle = _load_model()
    if model_bundle is not None:
        clf = model_bundle["clf"]
        X = _build_feature_vector(row)
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        if pred == 1:   # transient fault
            confidence  = float(proba[1])
            likelihoods = _transient_cause_likelihoods(row)
            _r2a = ClassificationResult(
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
                recommendation=_transient_recommendation(row),
                features=row,
                soe=_soe,
                cause_pcts=_compute_cause_pcts(row),
            )
            _r2a.description = _generate_description(row, _r2a)
            return _r2a
        else:
            confidence_transient = float(proba[1])
            likelihoods = _transient_cause_likelihoods(row)
            _r2b = ClassificationResult(
                label="GANGGUAN TRANSIEN",
                confidence=max(confidence_transient, 0.5),
                tier=2,
                rule_name="petir_decision_tree_non_petir",
                evidence=(
                    f"Classifier ML: pola waveform tidak khas petir (prob_transien={confidence_transient:.0%})  "
                    f"dur={row.get('fault_duration_ms', 0):.0f}ms  "
                    f"fault_count={row.get('fault_count', '?')}  |  "
                    f"Estimasi penyebab (heuristik): {likelihoods}  |  "
                    f"Catatan: data latih non-PETIR terbatas — konfirmasi penyebab via "
                    f"data cuaca, CCTV, atau inspeksi lapangan."
                    + _unknown_prot_caveat
                ),
                recommendation=_transient_recommendation(row),
                features=row,
                soe=_soe,
                cause_pcts=_compute_cause_pcts(row),
            )
            _r2b.description = _generate_description(row, _r2b)
            return _r2b

    # ── Step 8: fallback ──────────────────────────────────────────────────────
    _rfb = ClassificationResult(
        label="PERLU INVESTIGASI",
        confidence=0.0,
        tier=0,
        rule_name="no_model",
        evidence=(
            "Tidak ada aturan Tier 1 yang cocok dan model ML belum tersedia — jalankan models/train.py"
            + _unknown_prot_caveat
        ),
        recommendation=(
            "Kumpulkan data pendukung (cuaca, CCTV, laporan patroli) "
            "untuk menentukan penyebab gangguan ini."
        ),
        features=row,
        soe=_soe,
        cause_pcts=[],
    )
    _rfb.description = _generate_description(row, _rfb)
    return _rfb


def _flatten(feat, fault) -> dict:
    """Minimal flatten of DistanceFeatures + FaultEvent into a row dict."""
    return {
        "fault_duration_ms":    fault.duration_ms,
        "fault_inception_ms":   round(fault.inception_time * 1000, 2),
        "fault_count":          feat.fault_count,
        "faulted_phases":       "+".join(feat.faulted_phases) if feat.faulted_phases else "",
        "fault_type":           feat.fault_type,
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
        "i0_magnitude_a":       feat.i0_magnitude_a,
        "i1_magnitude_a":       feat.i1_magnitude_a,
        "i2_magnitude_a":       feat.i2_magnitude_a,
        "thd_percent":          feat.thd_percent,
        "inception_angle_degrees": feat.inception_angle_degrees,
        "voltage_sag_depth_pu": feat.voltage_sag_depth_pu,
        "voltage_sag_phase":    feat.voltage_sag_phase,
        "v_prefault_v":         feat.v_prefault_v,
        "v_fault_v":            feat.v_fault_v,
        "r_x_ratio":            feat.r_x_ratio,
        "z_magnitude_ohms":     feat.z_magnitude_ohms,
        "z_angle_degrees":      feat.z_angle_degrees,
        "station_name":         feat.station_name,
        "relay_model":          feat.relay_model,
        "voltage_kv":           feat.voltage_kv,
        "sampling_rate_hz":     feat.sampling_rate_hz,
        "record_duration_ms":   feat.record_duration_ms,
    }


def extract_soe_from_file(cfg_path: str) -> list:
    """Try to extract SOE from a COMTRADE file. Returns [] on any failure."""
    try:
        record = parse_comtrade(str(cfg_path))
        if record is None:
            return []
        fault = detect_fault(record)
        inception = fault.inception_time if fault else None
        return extract_soe(record, fault_inception_s=inception)
    except Exception:
        return []


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
