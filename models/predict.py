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
from core.protection_router import determine_protection, ProtectionType
from core.fault_detector import detect_fault, extract_soe
from core.feature_extractor import extract_distance_features
from core.transformer_channel_mapper import map_transformer_channels
from core.transformer_feature_extractor import extract_transformer_features, features_to_dict
from models.rules import apply_rules, RuleResult
from models.train import (
    FEATURE_COLS, encode_reclose,
    encode_trip_type, encode_zone, parse_phase_count,
)
from models.transformer_classifier import (
    classify_transformer_event,
    TransformerClassificationResult,
)

MODEL_PATH = Path(__file__).parent / "petir_tree.pkl"


def _compute_cause_scores(row: dict) -> dict:
    """
    Rule-based heuristic scores for each transient cause.
    Based on domain knowledge of power system protection behaviour.
    NOT a trained model — do not interpret as statistically rigorous.

    Signals used:
        fault_duration_ms, fault_count, i0_i1_ratio, peak_fault_current_a,
        reclose_successful, reclose_time_ms,
        di_dt_max, thd_percent, fault_type, inception_angle_degrees

    IMPORTANT: di_dt_max and peak_fault_current_a are only used as
    discriminators when the CT scaling is reliable (peak >= 200A).
    Below that threshold the values are secondary-side readings that
    cannot be compared across files without knowing the CT ratio.
    """
    dur        = float(row.get("fault_duration_ms", 80) or 80)
    fc         = int(row.get("fault_count", 1) or 1)
    i0i1       = float(row.get("i0_i1_ratio", 0) or 0)
    peak       = float(row.get("peak_fault_current_a", 0) or 0)
    reclose_ok = row.get("reclose_successful")
    rc_ms      = float(row.get("reclose_time_ms") or 0)
    di_dt      = float(row.get("di_dt_max", 0) or 0)
    thd        = float(row.get("thd_percent", 0) or 0)
    fault_type = str(row.get("fault_type", "") or "").upper()
    ang        = float(row.get("inception_angle_degrees", -1) or -1)

    # Only trust peak/di_dt when CT scaling looks primary (>=200A)
    scaled = peak >= 200.0

    # Base rates reflect real-world distribution (~87% PETIR in our labeled set).
    # Prior=3.5 is the calibrated balance point: 75% overall accuracy with 40%
    # non-PETIR recall (vs 62.5% / 20% with the original unweighted prior).
    scores = {"PETIR": 3.5, "Layang-Layang": 1.0, "Hewan": 1.0,
              "Benda Asing": 1.0, "Pohon": 0.5}

    # ── PETIR: steep wavefront, high current, single event, SLG ─────────────
    # Duration — lightning clears quickly (via AR), but FCT can be 80–150ms
    if dur < 60:    scores["PETIR"] *= 2.0
    elif dur < 150: scores["PETIR"] *= 1.3
    elif dur > 500: scores["PETIR"] *= 0.4

    # Fault count — lightning is typically a single impulse (or 1-pole reclose)
    if fc == 1:   scores["PETIR"] *= 1.3
    elif fc > 3:  scores["PETIR"] *= 0.5

    # Scaled signals — only when CT is reliable
    if scaled:
        if peak > 10000:     scores["PETIR"] *= 2.2
        elif peak > 5000:    scores["PETIR"] *= 1.6
        elif peak > 2000:    scores["PETIR"] *= 1.1
        elif peak < 500:     scores["PETIR"] *= 0.6

        if di_dt > 1_000_000:  scores["PETIR"] *= 1.8
        elif di_dt > 500_000:  scores["PETIR"] *= 1.3
        elif di_dt < 100_000:  scores["PETIR"] *= 0.6

    # THD — only meaningful when NOT near-zero (scaling issue gives flat waveforms)
    if thd > 60:   scores["PETIR"] *= 0.8
    elif thd < 10 and thd > 0: scores["PETIR"] *= 1.1

    # Inception angle near voltage peak (90°/270°) — highest arc breakdown probability
    if ang > 0:
        dist = min(abs(ang - 90), abs(ang - 270), abs(ang - 90 + 360), abs(ang - 270 + 360))
        if dist < 25:   scores["PETIR"] *= 1.3
        elif dist < 50: scores["PETIR"] *= 1.1

    # SLG is the most common fault type for lightning (single conductor flashover)
    if fault_type == "SLG":   scores["PETIR"] *= 1.2
    elif fault_type == "3PH": scores["PETIR"] *= 0.6

    # Short AR dead time = 1-pole scheme = typical lightning signature
    if rc_ms > 0 and rc_ms < 500: scores["PETIR"] *= 1.2

    # ── Layang-Layang: conductive string, SHORT duration, HIGH THD ───────────
    # Kite string acts as resistive contact → high harmonic distortion
    # Current is LIMITED by string resistance → LOW peak compared to lightning
    # Require COMBINATION of signals to overcome the high PETIR prior
    if thd > 50:   scores["Layang-Layang"] *= 2.5
    elif thd > 30: scores["Layang-Layang"] *= 1.5

    if 30 <= dur <= 120:    scores["Layang-Layang"] *= 1.8
    elif 120 < dur <= 350:  scores["Layang-Layang"] *= 1.2
    elif dur > 600:         scores["Layang-Layang"] *= 0.3

    if fc >= 3:   scores["Layang-Layang"] *= 1.8
    elif fc == 2: scores["Layang-Layang"] *= 1.4

    # Only use peak/di_dt discrimination when scaling is reliable
    if scaled:
        if peak < 2000:            scores["Layang-Layang"] *= 1.8
        elif peak > 10000:         scores["Layang-Layang"] *= 0.3

        if di_dt < 200_000:        scores["Layang-Layang"] *= 1.8
        elif di_dt > 1_000_000:    scores["Layang-Layang"] *= 0.5

    if fault_type == "SLG": scores["Layang-Layang"] *= 1.2

    # ── Hewan: brief single-phase contact ────────────────────────────────────
    if dur < 100 and i0i1 > 0.8: scores["Hewan"] *= 2.5
    elif i0i1 > 0.5:             scores["Hewan"] *= 1.4

    if fc == 1:   scores["Hewan"] *= 1.3
    elif fc == 2: scores["Hewan"] *= 1.2

    if scaled:
        if peak < 500:           scores["Hewan"] *= 2.0
        elif 500 <= peak < 4000: scores["Hewan"] *= 1.4
        elif peak > 10000:       scores["Hewan"] *= 0.2

        if di_dt < 50_000:       scores["Hewan"] *= 1.5

    # ── Benda Asing: foreign object — medium duration ────────────────────────
    if 80 <= dur <= 400:  scores["Benda Asing"] *= 1.8
    elif dur < 50:        scores["Benda Asing"] *= 0.5

    if fc >= 2:           scores["Benda Asing"] *= 1.8

    if 20 <= thd <= 60:   scores["Benda Asing"] *= 1.5
    elif thd > 60:        scores["Benda Asing"] *= 0.8

    if scaled:
        if 500 <= peak < 8000:  scores["Benda Asing"] *= 1.4
        elif peak < 200:        scores["Benda Asing"] *= 0.6

    # ── Pohon: branch contact — long duration, high fault count ──────────────
    if dur > 600:   scores["Pohon"] *= 4.0
    elif dur > 300: scores["Pohon"] *= 2.5
    elif dur < 100: scores["Pohon"] *= 0.3

    if fc > 3:   scores["Pohon"] *= 2.5
    elif fc > 2: scores["Pohon"] *= 1.8

    if reclose_ok is False or reclose_ok == "False":
        scores["Pohon"] *= 2.0

    if scaled:
        if 500 <= peak < 6000:  scores["Pohon"] *= 1.3
        elif peak > 8000:       scores["Pohon"] *= 0.3

    if fault_type in ("SLG", "DLG"): scores["Pohon"] *= 1.2

    return scores


def _transient_cause_likelihoods(row: dict) -> str:
    """Returns a formatted string like: PETIR 55% | Layang 25% | Hewan 12% | ..."""
    scores = _compute_cause_scores(row)
    total  = sum(scores.values())
    parts  = sorted(scores.items(), key=lambda x: -x[1])
    return "  |  ".join(f"{k} {v/total*100:.0f}%" for k, v in parts)


_CAUSE_RECOMMENDATIONS = {
    # Model class names (canonical)
    "PETIR": (
        "Rekaman dapat diarsipkan sebagai indikasi gangguan petir. "
        "Bandingkan dengan data cuaca atau rekaman penangkal petir di sekitar jalur untuk konfirmasi."
    ),
    "LAYANG": (
        "Periksa area ROW untuk aktivitas layang-layang. "
        "Koordinasikan sosialisasi larangan bermain layang-layang di bawah SUTT dengan masyarakat sekitar jalur."
    ),
    "POHON": (
        "Inspeksi vegetasi di sepanjang ROW pada zona gangguan. "
        "Jadwalkan pemangkasan jika terdapat pohon yang mendekati jarak aman konduktor."
    ),
    "HEWAN": (
        "Inspeksi isolator dan tower di zona gangguan untuk jejak kontak hewan. "
        "Pertimbangkan pemasangan pelindung hewan (bird/animal guard) pada tower yang rawan."
    ),
    "BENDA_ASING": (
        "Lakukan inspeksi visual tower dan konduktor di zona gangguan untuk benda asing "
        "(plastik, banner, terpal, tali layang, dll). Dokumentasikan untuk pemetaan titik rawan."
    ),
    "KONDUKTOR": (
        "Lakukan inspeksi mekanik segera pada tower dan konduktor di zona operasi rele. "
        "Periksa kondisi joint, klem, armor rod, dan struktur tower. Pertimbangkan patroli helikopter."
    ),
}


def _label_recommendation(model_class: str) -> str:
    """Return follow-up recommendation for a model class label."""
    return _CAUSE_RECOMMENDATIONS.get(
        model_class,
        "Kumpulkan data pendukung (cuaca, CCTV, laporan patroli) untuk menentukan penyebab gangguan."
    )


def _transient_recommendation(row: dict) -> str:
    """Return a context-aware follow-up recommendation based on the top heuristic cause."""
    scores = _compute_cause_scores(row)
    top    = max(scores, key=scores.get)
    # Map heuristic score keys to canonical model class names
    _heuristic_to_canonical = {
        "PETIR": "PETIR", "Layang-Layang": "LAYANG",
        "Hewan": "HEWAN", "Benda Asing": "BENDA_ASING", "Pohon": "POHON",
    }
    canonical = _heuristic_to_canonical.get(top, top)
    return _CAUSE_RECOMMENDATIONS.get(canonical, "Verifikasi penyebab melalui data pendukung.")


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
    # Phase 2 — transformer-specific fields
    event_type: str = "LINE"        # "LINE" or "TRANSFORMER"
    transformer_result: object = None  # TransformerClassificationResult if 87T
    transformer_features: dict = None  # flat dict of TransformerFeatures
    action_required: bool = False   # True when transformer has critical fault


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

    # Line 4: AI prediction / likely cause narrative
    conf_pct = result.confidence * 100
    label_upper = str(result.label or "").upper()
    cause_pcts = result.cause_pcts or []
    is_permanent = ("PERMANEN" in label_upper) or ("KONDUKTOR" in label_upper)

    if (not is_permanent) and ("TRANSIEN" in label_upper) and cause_pcts:
        top = max(cause_pcts, key=lambda x: float(x.get("pct", 0) or 0))
        top_name = str(top.get("name", "LAIN-LAIN"))
        top_pct = float(top.get("pct", 0) or 0)

        # Mention runner-up when the top cause is not dominant.
        others = sorted(cause_pcts, key=lambda x: -(float(x.get("pct", 0) or 0)))
        second = others[1] if len(others) > 1 else None
        second_name = str(second.get("name")) if second else ""
        second_pct = float(second.get("pct", 0) or 0) if second else 0.0

        if top_name.upper() == "PETIR" and top_pct >= 99.5:
            lines.append(
                f"AI mengidentifikasi gangguan ini sebagai {result.label}. "
                f"Penyebab paling mungkin adalah {top_name} dengan keyakinan heuristik {top_pct:.1f}% "
                f"(sangat dominan)."
            )
        elif top_pct >= 70:
            lines.append(
                f"AI mengidentifikasi gangguan ini sebagai {result.label}. "
                f"Penyebab paling mungkin adalah {top_name} ({top_pct:.1f}% heuristik)."
            )
        else:
            if second_name:
                lines.append(
                    f"AI mengidentifikasi gangguan ini sebagai {result.label}. "
                    f"Estimasi penyebab teratas: {top_name} ({top_pct:.1f}%), "
                    f"diikuti {second_name} ({second_pct:.1f}%)."
                )
            else:
                lines.append(
                    f"AI mengidentifikasi gangguan ini sebagai {result.label}. "
                    f"Estimasi penyebab teratas: {top_name} ({top_pct:.1f}%)."
                )
    else:
        lines.append(
            f"Berdasarkan analisis pola gelombang, AI mengklasifikasikan gangguan ini sebagai "
            f"{result.label} dengan tingkat keyakinan {conf_pct:.0f}%."
        )

    return "\n".join(lines)


def _classify_transformer_file(record, prot, fault, soe, cfg_path: str) -> "ClassificationResult":
    """
    Phase 2 classification pipeline for 87T transformer events.

    Steps:
      1. Map transformer channels (HV/LV/diff/restraint)
      2. Extract harmonic + differential features
      3. Apply knowledge-based transformer classifier
      4. Return ClassificationResult with transformer_result attached
    """
    try:
        ch_map = map_transformer_channels(record)
        tf = extract_transformer_features(record, ch_map, fault_event=fault)
        xfmr_result = classify_transformer_event(tf)

        # Map transformer event class → user-visible label (Indonesian)
        _label_map = {
            'INRUSH':         'INRUSH MAGNETISASI',
            'INTERNAL_FAULT': 'GANGGUAN INTERNAL TRAFO',
            'THROUGH_FAULT':  'GANGGUAN EKSTERNAL (THROUGH)',
            'OVEREXCITATION': 'TEGANGAN LEBIH / OVEREKSITASI',
            'MAL_OPERATE':    'KEMUNGKINAN MALOPERATE',
            'UNKNOWN':        'PERLU INVESTIGASI',
        }
        label = _label_map.get(xfmr_result.event_class, 'PERLU INVESTIGASI')

        # Build class probability list for UI bars
        cause_pcts = sorted(
            [{"name": _label_map.get(k, k), "pct": round(v * 100, 1)}
             for k, v in xfmr_result.class_probabilities.items()],
            key=lambda x: -x["pct"]
        )

        row = features_to_dict(tf)
        row['protection_type'] = '87T'
        row['station_name']    = getattr(record, 'station_name', '')
        row['relay_model']     = getattr(record, 'rec_dev_id', '')

        # Copy fault timing from FaultEvent so waveform viewer can auto-zoom to the fault window.
        # Without this, fault_inception_ms stays None and the Focus Fault button doesn't work.
        if fault is not None:
            row['fault_inception_ms'] = round(float(fault.inception_time) * 1000.0, 2)
            clearing = getattr(fault, 'clearing_time', None)
            if clearing is not None:
                row['fault_clearance_ms'] = round(float(clearing) * 1000.0, 2)
            elif fault.duration_ms and fault.duration_ms > 0:
                row['fault_clearance_ms'] = row['fault_inception_ms'] + fault.duration_ms

        result = ClassificationResult(
            label=label,
            confidence=xfmr_result.confidence,
            tier=1,   # knowledge-based = Tier 1; will become Tier 2 when ML model trained
            rule_name=xfmr_result.rule_name,
            evidence=xfmr_result.evidence,
            recommendation=xfmr_result.recommendation,
            features=row,
            soe=soe,
            cause_pcts=cause_pcts,
            event_type="TRANSFORMER",
            transformer_result=xfmr_result,
            transformer_features=row,
            action_required=xfmr_result.action_required,
        )

        # Generate description
        result.description = _generate_transformer_description(tf, xfmr_result)
        return result

    except Exception as exc:
        raise ValueError(f"Transformer classification failed for {cfg_path}: {exc}") from exc


def _generate_transformer_description(tf, xr: "TransformerClassificationResult") -> str:
    """Generate narrative description for transformer event result."""
    lines = []
    h2 = tf.h2_ratio_max_pct
    h5 = tf.h5_ratio_max_pct
    dc = tf.dc_offset_index_max
    slp = tf.slope_worst_pct

    lines.append(
        f"Rekaman proteksi diferensial transformator (87T) terdeteksi di gardu {tf.station_name}. "
        f"Rele family: {tf.relay_family}."
    )

    if h2 is not None:
        lines.append(f"Rasio harmonik ke-2 (indikator inrush): {h2:.1f}% (ambang: 15%).")
    if h5 is not None:
        lines.append(f"Rasio harmonik ke-5 (indikator overeksitasi): {h5:.1f}% (ambang: 10%).")
    if slp is not None:
        lines.append(f"Rasio diferensial/restraint (slope): {slp:.1f}% (ambang slope-1: 20%).")
    if dc is not None:
        lines.append(f"Indeks offset DC (bentuk gelombang): {dc:.3f} (ambang inrush: 0.35).")
    if tf.energisation_flag:
        lines.append("Terdeteksi kondisi energisasi: arus sisi LV ≈ 0 sebelum event.")

    lines.append(
        f"AI mengklasifikasikan event ini sebagai: {xr.event_class} "
        f"(keyakinan {xr.confidence:.0%})."
    )
    if xr.action_required:
        lines.append("TINDAKAN DIPERLUKAN — lihat rekomendasi untuk langkah selanjutnya.")

    return "\n".join(lines)


def _load_model() -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _build_feature_vector(row: dict) -> np.ndarray:
    """Build the numpy feature vector expected by the Tier 2 multi-class classifier."""
    di_dt  = float(row.get("di_dt_max", 0) or 0)
    peak_i = float(row.get("peak_fault_current_a", 0) or 0)

    vec = [
        float(row.get("fault_duration_ms", 0) or 0),           # fault_duration_ms
        float(row.get("fault_count", 1) or 1),                  # fault_count
        np.log1p(max(peak_i, 0)),                               # peak_fault_current_a (log)
        np.log1p(max(di_dt, 0)),                                # di_dt_max (log)
        float(row.get("i0_i1_ratio", 0) or 0),                 # i0_i1_ratio
        float(row.get("thd_percent", 0) or 0),                  # thd_percent
        float(row.get("inception_angle_degrees", 0) or 0),      # inception_angle_degrees
        float(row.get("voltage_sag_depth_pu", 0) or 0),        # voltage_sag_depth_pu
        encode_reclose(row.get("reclose_successful")),           # reclose_enc
        1 if str(row.get("is_ground_fault", "")).lower() == "true" else 0,  # is_ground_enc
        encode_trip_type(row.get("trip_type", "")),              # trip_type_enc
        parse_phase_count(row.get("faulted_phases", "")),        # phase_count
        encode_zone(row.get("zone_operated", "")),               # zone_enc
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

    # ── Phase 2: Route TRANSFORMER_DIFF (87T) to transformer classifier ─────────
    if prot.primary_protection == ProtectionType.TRANSFORMER_DIFF:
        return _classify_transformer_file(record, prot, fault, _soe, cfg_path)

    if prot.primary_protection.name == "UNKNOWN":
        # No protection type identified from status channels.
        # This is typical for standalone external DFRs (e.g. Qualitrol) that only capture
        # analog waveforms without relay trip/zone digital outputs.
        # Waveform physics (di/dt, THD, peak-I, inception angle) are still valid for
        # cause classification — zone info will be absent, but the ML model is trained
        # on waveform features only and does not require zone data.
        # Confidence is reduced slightly to reflect the missing relay context.
        _routing_caveat = (
            " | [DFR EKSTERNAL: sinyal trip rele tidak ditemukan — "
            "klasifikasi hanya dari analisis gelombang arus/tegangan. "
            "Untuk konfirmasi, gunakan rekaman rele jarak jika tersedia.]"
        )
    elif prot.primary_protection == ProtectionType.DIFFERENTIAL:
        # 87L line differential is a valid transmission-line event, but we do not yet
        # model the 87L logic itself. Fall back to waveform-based line analysis with a caveat.
        _routing_caveat = (
            " | [CATATAN: Rekaman menunjukkan proteksi diferensial saluran (87L) — "
            "analisis berikut dihitung dari pola gelombang arus/tegangan, "
            "bukan logika 87L khusus. Verifikasi manual diperlukan.]"
        )
    elif prot.primary_protection.name != "DISTANCE":
        raise ValueError(
            f"Protection type '{prot.primary_protection.name}' is not supported "
            f"(only DISTANCE relay files are classified)"
        )
    else:
        _routing_caveat = ""

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
            evidence=rule_result.evidence + _routing_caveat,
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

    # ── Step 7: Tier 2 ML classifier (multi-class) ───────────────────────────
    # Note: reclose success is now an input feature to the model, not a bypass.
    # The multi-class model directly outputs the physical cause label.
    _reclose_ok = row.get("reclose_successful")
    _peak_i     = float(row.get("peak_fault_current_a", 0) or 0)
    _reclose_confirmed = (_reclose_ok is True or _reclose_ok == "True") and _peak_i > 200

    model_bundle = _load_model()
    if model_bundle is not None:
        clf          = model_bundle["clf"]
        classes      = model_bundle.get("classes", [])
        model_type   = model_bundle.get("model_type", "binary")
        X            = _build_feature_vector(row)
        pred         = clf.predict(X)[0]
        proba        = clf.predict_proba(X)[0]

        # ── Multi-class path (new fault_classifier.pkl) ───────────────────
        proba_classes = list(getattr(clf, "classes_", classes))
        if model_type == "multiclass_random_forest" and proba_classes:
            confidence = float(proba.max())
            # cause_pcts uses "name" key for template compatibility
            _label_id_map = {
                "PETIR":       "PETIR",
                "LAYANG":      "LAYANG-LAYANG",
                "POHON":       "POHON / VEGETASI",
                "HEWAN":       "HEWAN / BINATANG",
                "BENDA_ASING": "BENDA ASING",
                "KONDUKTOR":   "KONDUKTOR / TOWER",
            }
            cause_pcts_ml = [
                {"name": _label_id_map.get(cls, cls), "pct": round(float(p) * 100, 1)}
                for cls, p in sorted(zip(proba_classes, proba), key=lambda x: -x[1])
            ]
            label_id = _label_id_map.get(pred, pred)
            reclose_note = "  AR berhasil — gangguan terkonfirmasi transien." if _reclose_confirmed else ""
            _r2_mc = ClassificationResult(
                label=label_id,
                confidence=confidence,
                tier=2,
                rule_name="multiclass_random_forest",
                evidence=(
                    f"Classifier ML: {pred} ({confidence:.0%}){reclose_note}  "
                    f"dur={row.get('fault_duration_ms', 0):.0f}ms  "
                    f"peak_i={_peak_i:.0f}A  "
                    f"i0/i1={row.get('i0_i1_ratio', 0):.2f}  "
                    f"zone={row.get('zone_operated', '?')}"
                    + (_routing_caveat or "")
                ),
                recommendation=_label_recommendation(pred),
                features=row,
                soe=_soe,
                cause_pcts=cause_pcts_ml,
            )
            _r2_mc.description = _generate_description(row, _r2_mc)
            return _r2_mc

        # ── Legacy binary path (old petir_tree.pkl) ───────────────────────
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
                    + _routing_caveat
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
                    f"Catatan: data latih non-PETIR terbatas — konfirmasi penyebab via"
                    f"data cuaca, CCTV, atau inspeksi lapangan."
                    + _routing_caveat
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
            + _routing_caveat
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
