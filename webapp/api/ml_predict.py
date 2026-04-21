"""Bridge between COMTRADE session JSON and the existing fault_classifier.pkl model.

Extracts the same 17-feature vector that models/train.py uses, then runs the
LightGBM multi-class classifier with the same calibration and confidence caps
as models/predict.py.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PIPELINE_DIR = Path(__file__).parent.parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "fault_classifier.pkl"

_LABEL_DISPLAY = {
    "PETIR":       "Petir / Lightning",
    "LAYANG":      "Layang-Layang / Kite",
    "POHON":       "Pohon / Vegetasi",
    "HEWAN":       "Hewan / Binatang",
    "BENDA_ASING": "Benda Asing",
    "KONDUKTOR":   "Konduktor / Tower",
    "PERALATAN":   "Peralatan / Proteksi",
}

_TRANSIENT = {"PETIR", "LAYANG", "HEWAN", "BENDA_ASING"}


def _load_model() -> Optional[dict]:
    if not _MODEL_PATH.exists():
        return None
    with open(_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _find_ch(channels: list, candidates: list[str]) -> Optional[np.ndarray]:
    for ch in channels:
        if ch.get("canonical_name") in candidates or ch.get("name", "").upper() in candidates:
            return np.array(ch["samples"], dtype=float)
    return None


def _rms_window(arr: np.ndarray, start: int, n: int) -> float:
    seg = arr[start: start + n]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0


def _thd_percent(arr: np.ndarray, sr: float, freq: float) -> float:
    """Estimate THD% of a one-cycle window using FFT."""
    cycle = max(4, int(sr / freq))
    if len(arr) < cycle:
        return 0.0
    seg = arr[:cycle]
    spectrum = np.abs(np.fft.rfft(seg, n=cycle))
    if spectrum[1] < 1e-9:
        return 0.0
    harmonics = np.sqrt(np.sum(spectrum[2:] ** 2))
    return float(harmonics / spectrum[1] * 100.0)


def _symmetrical_components(ia: np.ndarray, ib: np.ndarray, ic: np.ndarray):
    """Return (I0_rms, I1_rms, I2_rms) using one-sample approximation of seq."""
    a = np.exp(1j * 2 * np.pi / 3)
    i0 = (ia + ib + ic) / 3.0
    i1 = (ia + a * ib + (a ** 2) * ic) / 3.0
    i2 = (ia + (a ** 2) * ib + a * ic) / 3.0
    return float(np.sqrt(np.mean(np.abs(i0) ** 2))), float(np.sqrt(np.mean(np.abs(i1) ** 2))), float(np.sqrt(np.mean(np.abs(i2) ** 2)))


def extract_ml_features(payload: dict, relay_type: str = "21") -> dict:
    """Build the 17-feature dict from a stored COMTRADE session payload."""
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []), dtype=float)
    freq = float(payload.get("frequency", 50.0))
    status_channels = payload.get("status_channels", [])

    if len(time) < 4:
        return _empty_features()

    sr = 1.0 / (time[1] - time[0])
    cycle_n = max(4, int(sr / freq))

    # Current channels
    ia = _find_ch(channels, ["IA", "IL1", "I1"])
    ib = _find_ch(channels, ["IB", "IL2", "I2"])
    ic = _find_ch(channels, ["IC", "IL3", "I3"])
    i_primary = ia if ia is not None else (ib if ib is not None else ic)

    # Voltage channels
    va = _find_ch(channels, ["VA", "VAN", "UA"])
    vb = _find_ch(channels, ["VB", "VBN", "UB"])
    vc = _find_ch(channels, ["VC", "VCN", "UC"])

    if i_primary is None:
        return _empty_features()

    # --- Fault inception detection (same logic as relay_21._extract_features_from_payload) ---
    pre_end = min(2 * cycle_n, len(i_primary) // 4)
    pre_rms = float(np.sqrt(np.mean(i_primary[:pre_end] ** 2))) if pre_end > 1 else 0.0
    threshold = max(pre_rms * 2.0, np.max(np.abs(i_primary)) * 0.3, 0.05)
    inception_idx = next(
        (k for k in range(pre_end, len(i_primary)) if abs(i_primary[k]) > threshold),
        int(np.argmax(np.abs(i_primary))),
    )
    extinction_idx = len(i_primary) - 1
    for k in range(inception_idx + cycle_n, len(i_primary)):
        s = max(0, k - cycle_n // 2)
        if float(np.sqrt(np.mean(i_primary[s: k + 1] ** 2))) < threshold * 0.6:
            extinction_idx = k
            break

    fault_duration_ms = float((time[extinction_idx] - time[inception_idx]) * 1000)
    fault_window = i_primary[inception_idx: inception_idx + cycle_n]

    # Peak fault current (primary side)
    peak_fault_current_a = float(np.max(np.abs(i_primary[inception_idx: extinction_idx + 1]))) if inception_idx < extinction_idx else float(np.max(np.abs(i_primary)))

    # di/dt max in fault window
    if len(fault_window) > 1:
        di_dt_max = float(np.max(np.abs(np.diff(fault_window))) * sr)
    else:
        di_dt_max = 0.0

    # THD of current at fault
    thd_percent = _thd_percent(fault_window, sr, freq) if len(fault_window) >= 4 else 0.0

    # Fault inception angle (FIA)
    fia_deg = 0.0
    if va is not None and inception_idx < len(va):
        v_peak = float(np.max(np.abs(va[:inception_idx]))) if inception_idx > 0 else float(np.max(np.abs(va)))
        if v_peak > 0:
            ratio = float(np.clip(va[inception_idx] / v_peak, -1.0, 1.0))
            fia_deg = float(np.degrees(np.arcsin(ratio)))

    # Symmetrical components (zero-seq / pos-seq ratio)
    i0_i1_ratio = 0.0
    if ia is not None and ib is not None and ic is not None:
        seg_len = min(cycle_n, len(ia), len(ib), len(ic))
        s = inception_idx
        fa = ia[s: s + seg_len].astype(complex)
        fb = ib[s: s + seg_len].astype(complex)
        fc = ic[s: s + seg_len].astype(complex)
        if len(fa) >= 4:
            i0_rms, i1_rms, _ = _symmetrical_components(fa, fb, fc)
            i0_i1_ratio = float(i0_rms / i1_rms) if i1_rms > 0 else 0.0

    # Voltage sag features
    voltage_sag_depth_pu = 0.0
    voltage_phase_ratio_spread_pu = 0.0
    healthy_phase_voltage_ratio = 1.0
    v2_v1_ratio = 0.0
    voltage_thd_max_percent = 0.0

    v_channels = [(va, "A"), (vb, "B"), (vc, "C")]
    prefault_v_rms = []
    fault_v_rms = []
    for v_ch, _ in v_channels:
        if v_ch is None:
            continue
        pre_v = v_ch[:pre_end]
        fault_v = v_ch[inception_idx: inception_idx + cycle_n]
        if len(pre_v) > 1:
            prefault_v_rms.append(_rms_window(v_ch, 0, pre_end))
        if len(fault_v) > 1:
            fault_v_rms.append(_rms_window(v_ch, inception_idx, cycle_n))

    if prefault_v_rms and fault_v_rms:
        pre_mean = float(np.mean(prefault_v_rms))
        fault_min = float(np.min(fault_v_rms))
        if pre_mean > 0:
            voltage_sag_depth_pu = max(0.0, float((pre_mean - fault_min) / pre_mean))
        sag_ratios = [f / p if p > 0 else 1.0 for f, p in zip(fault_v_rms, prefault_v_rms)]
        voltage_phase_ratio_spread_pu = float(np.std(sag_ratios)) if len(sag_ratios) > 1 else 0.0
        healthy_phase_voltage_ratio = float(np.max(sag_ratios)) if sag_ratios else 1.0

    if va is not None and vb is not None and vc is not None:
        seg_len = min(cycle_n, len(va), len(vb), len(vc))
        s = inception_idx
        fva = va[s: s + seg_len].astype(complex)
        fvb = vb[s: s + seg_len].astype(complex)
        fvc = vc[s: s + seg_len].astype(complex)
        if len(fva) >= 4:
            _, v1_rms, v2_rms = _symmetrical_components(fva, fvb, fvc)
            v2_v1_ratio = float(v2_rms / v1_rms) if v1_rms > 0 else 0.0

    if fault_v_rms:
        v_thds = []
        for v_ch, _ in v_channels:
            if v_ch is None:
                continue
            seg = v_ch[inception_idx: inception_idx + cycle_n]
            v_thds.append(_thd_percent(seg, sr, freq))
        voltage_thd_max_percent = float(max(v_thds)) if v_thds else 0.0

    # AR result
    ar_result = None
    for sch in status_channels:
        name = sch.get("name", "").upper()
        if any(k in name for k in ("AR", "RECLOSE", "RECLUSE", "RECLOS")):
            if 1 in (sch.get("samples") or []):
                ar_result = True
            else:
                ar_result = False
            break

    # Ground fault detection (I0 > 20% of I1)
    is_ground = i0_i1_ratio > 0.2

    # Trip type from status channels
    trip_type_str = "unknown"
    for sch in status_channels:
        name = sch.get("name", "").upper()
        if "3PH" in name or "THREE" in name or "3P" in name or "THREE_POLE" in name:
            trip_type_str = "three_pole"
            break
        if "1PH" in name or "SINGLE" in name or "1P" in name or "SINGLE_POLE" in name:
            trip_type_str = "single_pole"
            break

    # Faulted phases (approximate from which phases exceeded threshold)
    faulted_phases = []
    for v_arr, phase in [(ia, "A"), (ib, "B"), (ic, "C")]:
        if v_arr is None:
            continue
        seg = v_arr[inception_idx: inception_idx + cycle_n]
        if len(seg) > 0 and float(np.max(np.abs(seg))) > threshold * 0.5:
            faulted_phases.append(phase)
    faulted_phases_str = "+".join(faulted_phases) if faulted_phases else "A"

    # Zone from status channels
    zone_str = ""
    for sch in status_channels:
        name = sch.get("name", "").upper()
        for z in ("ZONE 1", "ZONE 2", "ZONE 3", "Z1", "Z2", "Z3"):
            if z in name and 1 in (sch.get("samples") or []):
                zone_str = z.replace(" ", "")
                break

    return {
        "fault_duration_ms": round(max(fault_duration_ms, 0.0), 1),
        "fault_count": 1,
        "peak_fault_current_a": round(peak_fault_current_a, 2),
        "di_dt_max": round(di_dt_max, 2),
        "i0_i1_ratio": round(i0_i1_ratio, 3),
        "thd_percent": round(thd_percent, 2),
        "inception_angle_degrees": round(fia_deg, 1),
        "voltage_sag_depth_pu": round(voltage_sag_depth_pu, 3),
        "voltage_phase_ratio_spread_pu": round(voltage_phase_ratio_spread_pu, 3),
        "healthy_phase_voltage_ratio": round(healthy_phase_voltage_ratio, 3),
        "v2_v1_ratio": round(v2_v1_ratio, 3),
        "voltage_thd_max_percent": round(voltage_thd_max_percent, 2),
        "reclose_successful": ar_result,
        "is_ground_fault": is_ground,
        "trip_type": trip_type_str,
        "faulted_phases": faulted_phases_str,
        "zone_operated": zone_str,
    }


def _empty_features() -> dict:
    return {k: 0 for k in [
        "fault_duration_ms", "fault_count", "peak_fault_current_a", "di_dt_max",
        "i0_i1_ratio", "thd_percent", "inception_angle_degrees", "voltage_sag_depth_pu",
        "voltage_phase_ratio_spread_pu", "healthy_phase_voltage_ratio", "v2_v1_ratio",
        "voltage_thd_max_percent", "reclose_successful", "is_ground_fault",
        "trip_type", "faulted_phases", "zone_operated",
    ]}


def run_ml_prediction(payload: dict, relay_type: str = "21") -> dict:
    """
    Run the LightGBM fault classifier on session payload.
    Returns a dict matching the AIFaultResult schema.
    Heavy model imports are deferred to here so server startup never fails.
    """
    # Lazy imports — only executed when AI analysis is requested
    try:
        from models.train import FEATURE_COLS, encode_reclose, encode_trip_type, encode_zone, parse_phase_count  # noqa: F401
        from models.predict import (
            _calibrate_proba,
            _build_feature_vector,
            _apply_transient_ambiguity_confidence_cap,
            _apply_equipment_caution_cap,
        )
    except Exception as e:
        return {
            "fault_type": "transient",
            "cause_ranking": [],
            "overall_confidence": 0.0,
            "evidence": [f"Model imports gagal: {e}"],
        }

    model_bundle = _load_model()
    row = extract_ml_features(payload, relay_type)

    LABEL_MAP = {
        "PETIR":       "Petir / Lightning",
        "LAYANG":      "Layang-Layang / Kite",
        "POHON":       "Pohon / Vegetasi",
        "HEWAN":       "Hewan / Binatang",
        "BENDA_ASING": "Benda Asing",
        "KONDUKTOR":   "Konduktor / Tower",
        "PERALATAN":   "Peralatan / Proteksi",
    }

    if model_bundle is None:
        n = len(LABEL_MAP)
        ranking = [
            {"cause": k, "label": v, "confidence": round(1 / n, 3)}
            for k, v in LABEL_MAP.items()
        ]
        return {
            "fault_type": "transient",
            "cause_ranking": ranking,
            "overall_confidence": round(1 / n, 3),
            "evidence": ["Model fault_classifier.pkl tidak ditemukan — prediksi tidak tersedia."],
        }

    clf = model_bundle["clf"]
    classes = list(getattr(clf, "classes_", model_bundle.get("classes", [])))

    try:
        from models.train import FEATURE_COLS as _FC
    except Exception:
        _FC = model_bundle.get("feature_cols", [])

    X = _build_feature_vector(row, model_bundle.get("feature_cols", _FC))
    pred = str(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]

    proba = _calibrate_proba(proba, temperature=1.5)
    confidence = float(proba.max())
    if confidence > 0.92:
        confidence = 0.92

    sorted_p = np.sort(proba)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else 1.0

    confidence, _ = _apply_transient_ambiguity_confidence_cap(
        confidence=confidence,
        pred_label=pred,
        proba_classes=classes,
        proba=proba,
        margin=margin,
    )
    confidence, _ = _apply_equipment_caution_cap(
        pred_label=pred,
        confidence=confidence,
        class_counts=model_bundle.get("class_counts"),
        soe=None,
        protection_name="DISTANCE" if relay_type == "21" else "DIFFERENTIAL",
    )

    ranking = sorted(
        [
            {
                "cause": cls,
                "label": LABEL_MAP.get(cls, cls),
                "confidence": round(float(p), 3),
            }
            for cls, p in zip(classes, proba)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # Fault type: transient if top class is in transient set or AR succeeded
    ar_ok = row.get("reclose_successful")
    top_class = ranking[0]["cause"] if ranking else pred
    fault_type = "transient" if (top_class in _TRANSIENT or ar_ok is True) else "permanent"
    if ar_ok is False:
        fault_type = "permanent"

    # Evidence bullets
    evidence = []
    evidence.append(
        f"Model LightGBM multi-class (T=1.5 kalibrasi): {pred} ({confidence:.0%})"
    )
    if margin < 0.15 and len(ranking) >= 2:
        evidence.append(
            f"Keputusan tipis — selisih ke runner-up {ranking[1]['cause']} "
            f"hanya {margin * 100:.1f} pp. Verifikasi lapangan disarankan."
        )
    dur = row.get("fault_duration_ms", 0)
    if dur < 100:
        evidence.append(f"Durasi gangguan singkat ({dur:.0f} ms) — indikator gangguan transien.")
    elif dur > 400:
        evidence.append(f"Durasi gangguan panjang ({dur:.0f} ms) — indikator gangguan permanen.")
    if ar_ok is True:
        evidence.append("AR berhasil — gangguan terkonfirmasi transien.")
    elif ar_ok is False:
        evidence.append("AR gagal — indikator kuat gangguan permanen.")
    fia = row.get("inception_angle_degrees", 0)
    if abs(fia) > 60:
        evidence.append(f"FIA = {fia:.1f}° — gangguan dekat puncak tegangan, tipikal petir.")

    return {
        "fault_type": fault_type,
        "cause_ranking": ranking,
        "overall_confidence": confidence,
        "evidence": evidence,
    }
