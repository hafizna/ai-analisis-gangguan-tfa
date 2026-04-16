"""
Sistem Klasifikasi Penyebab Gangguan DFR — UIT JBT
====================================================
Unggah file COMTRADE (.cfg + .dat), dapatkan prediksi penyebab gangguan,
konfirmasi atau koreksi hasilnya, dan pantau riwayat analisis.

Menggunakan: Aturan deterministic (Tier 1) + Classifier ML PETIR (Tier 2)

Jalankan dari folder pipeline/:
    python webapp/app.py
Buka: http://localhost:5000
"""

import os
import sys
import csv
import json
import re
import uuid
import warnings
import traceback
import logging
from collections import Counter
from pathlib import Path
from datetime import datetime
import io as _io
import base64 as _b64

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
try:
    import psycopg
except Exception:
    psycopg = None

warnings.filterwarnings("ignore")

TRANSFORMER_RELAY_ROLES = [
    ("AUTO", "Auto detect dari file"),
    ("87T", "87T / Differential"),
    ("OCR_HV", "OCR HV"),
    ("OCR_LV", "OCR LV / Incomer"),
    ("REF", "REF"),
    ("GFR_HV", "GFR HV"),
    ("SBEF", "SBEF"),
    ("FEEDER", "Feeder / Outgoing"),
    ("OTHER", "Relay lain"),
]

TRANSFORMER_ANALYSIS_GOALS = [
    ("EVENT_MEANING", "Interpretasi event lokal"),
    ("FAULT_ORIGIN", "Menentukan origin fault"),
    ("PROTECTION_REVIEW", "Review selektivitas / proteksi"),
    ("ROOT_CAUSE", "Investigasi sebab akhir"),
]

MULTICLASS_MODEL_TYPES = {"multiclass_random_forest", "multiclass_lightgbm"}
MULTICLASS_LABEL_MAP = {
    "PETIR": "PETIR",
    "LAYANG": "LAYANG-LAYANG",
    "POHON": "POHON / VEGETASI",
    "HEWAN": "HEWAN / BINATANG",
    "BENDA_ASING": "BENDA ASING",
    "KONDUKTOR": "KONDUKTOR / TOWER",
    "PERALATAN": "PERALATAN / PROTEKSI",
}

# Bootstrap path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predict import (
    classify_file,
    extract_soe_from_file,
    _load_model,
    _build_feature_vector,
    _compute_cause_pcts,
    _transient_recommendation,
)
from models.rules import apply_rules
from core.comtrade_parser import parse_comtrade
from core.path_heuristics import (
    infer_path_kind,
    infer_path_tag,
    infer_status_data,
    infer_suspected_label,
)


def _f(val, default=0):
    """Cast numpy/float32 to plain Python float for JSON serialisation."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _s(val, default="-"):
    """Cast to plain str."""
    return str(val) if val is not None else default


def _json_safe(obj):
    """Recursively convert numpy/scalar objects to JSON-serialisable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # numpy scalar types expose .item() -> Python scalar
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


def _choice_label(options, value: str, fallback: str) -> str:
    for key, label in options:
        if key == value:
            return label
    return fallback


def _transformer_role_label(value: str) -> str:
    return _choice_label(TRANSFORMER_RELAY_ROLES, value, value or "Relay tidak diketahui")


def _transformer_goal_label(value: str) -> str:
    return _choice_label(TRANSFORMER_ANALYSIS_GOALS, value, value or "Tujuan tidak diketahui")


def _save_uploaded_pair(cfg_file, dat_file, ts: str, prefix: str = "") -> tuple[Path, Path]:
    slot = f"{prefix}_" if prefix else ""
    cfg_name = secure_filename(f"{ts}_{slot}{cfg_file.filename}")
    dat_name = secure_filename(f"{ts}_{slot}{dat_file.filename}")
    cfg_path = UPLOAD_DIR / cfg_name
    dat_path = UPLOAD_DIR / dat_name
    cfg_file.save(cfg_path)
    dat_file.save(dat_path)
    return cfg_path, dat_path


def _save_analysis_to_store(analysis: dict) -> str:
    """Persist analysis server-side and keep only a token in session."""
    token = uuid.uuid4().hex
    path = ANALYSIS_DIR / f"{token}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(analysis), f, ensure_ascii=False)
    session["analysis_token"] = token
    return token


def _load_analysis_from_store() -> dict | None:
    """Load analysis from the server-side store by token from session."""
    token = session.get("analysis_token")
    if not token:
        return None
    path = ANALYSIS_DIR / f"{token}.json"
    if not path.exists():
        session.pop("analysis_token", None)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        session.pop("analysis_token", None)
        return None


def _clear_analysis_store():
    """Delete stored analysis for current session token."""
    token = session.pop("analysis_token", None)
    if not token:
        return
    path = ANALYSIS_DIR / f"{token}.json"
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _ratio(primary, secondary) -> float:
    p = _f(primary, 1.0)
    s = _f(secondary, 1.0)
    return (p / s) if s > 0 else 1.0


def _to_bool_or_none(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def _normalize_label(val: str) -> str:
    return re.sub(r"\s+", " ", (val or "").strip().upper())


def _top_cause_name(result) -> str:
    cause_pcts = getattr(result, "cause_pcts", None) or []
    if isinstance(cause_pcts, list) and cause_pcts:
        try:
            top = max(cause_pcts, key=lambda item: _f(item.get("pct"), 0.0))
            return _s(top.get("name"), "")
        except Exception:
            return ""
    return ""


def _top_predicted_cause_from_analysis(analysis: dict) -> str:
    """
    Return top probable cause used for accuracy comparison.
    If cause_pcts exists, use highest-pct cause. Otherwise fallback to model label.
    """
    cause_pcts = analysis.get("cause_pcts") or []
    if isinstance(cause_pcts, list) and cause_pcts:
        try:
            top = max(cause_pcts, key=lambda x: _f(x.get("pct"), 0))
            name = _s(top.get("name"), "").strip()
            if name:
                return name
        except Exception:
            pass
    return _s(analysis.get("label"), "LAIN-LAIN")


def _row_correct_flag(row: dict):
    """
    Cause-based correctness: predicted top cause vs confirmed cause.
    Returns:
        True/False when comparable, None when prediction is non-causal bucket.
    """
    pred_cause = _s(row.get("predicted_cause_top"), "").strip()
    if not pred_cause:
        pred_cause = _s(row.get("predicted_label"), "").strip()
    conf = _s(row.get("confirmed_label"), "").strip()
    if not pred_cause or not conf:
        return None
    if _normalize_label(pred_cause) in {
        "GANGGUAN TRANSIEN", "GANGGUAN PERMANEN", "PERLU INVESTIGASI"
    }:
        return None
    return _normalize_label(pred_cause) == _normalize_label(conf)


def _ensure_history_schema():
    """Normalize history.csv to current HISTORY_FIELDS and repair mixed-width rows."""
    if not HISTORY_CSV.exists():
        return

    try:
        with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
            raw = list(csv.reader(f))
        if not raw:
            return
    except Exception:
        return

    header = raw[0]
    data_rows = raw[1:]

    # Fast path: already aligned.
    if header == HISTORY_FIELDS and all(len(r) == len(HISTORY_FIELDS) for r in data_rows):
        return

    normalized = []
    for r in data_rows:
        if not r:
            continue

        if len(r) == len(HISTORY_FIELDS):
            mapped = dict(zip(HISTORY_FIELDS, r))
        elif len(r) == len(header):
            mapped = dict(zip(header, r))
        elif len(r) == len(HISTORY_FIELDS) - 1:
            # Likely missing predicted_cause_top, insert it after predicted_label.
            rr = list(r)
            rr.insert(4, rr[3] if len(rr) > 3 else "")
            rr = rr[:len(HISTORY_FIELDS)] + [""] * max(0, len(HISTORY_FIELDS) - len(rr))
            mapped = dict(zip(HISTORY_FIELDS, rr))
        else:
            rr = list(r)[:len(HISTORY_FIELDS)]
            rr += [""] * max(0, len(HISTORY_FIELDS) - len(rr))
            mapped = dict(zip(HISTORY_FIELDS, rr))

        out = {k: _s(mapped.get(k), "") for k in HISTORY_FIELDS}
        if not out.get("predicted_cause_top"):
            out["predicted_cause_top"] = out.get("predicted_label", "")

        corr = _row_correct_flag(out)
        out["correct"] = "True" if corr is True else ("False" if corr is False else "")
        normalized.append(out)

    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


def _norm_ratio_num(x: float) -> float:
    """Normalize ratio display values: prefer integer-looking numbers."""
    xf = _f(x, 1.0)
    if abs(xf - round(xf)) < 1e-9:
        return int(round(xf))
    return round(xf, 6)


def _pick_representative_ratio(channels, measurement: str) -> tuple[float, float, bool]:
    """Pick most representative (primary, secondary) pair and whether ratio looks explicitly provided."""
    pairs = []
    for ch in channels:
        if getattr(ch, "measurement", "") != measurement:
            continue
        p = _norm_ratio_num(getattr(ch, "ct_primary", 1.0))
        s = _norm_ratio_num(getattr(ch, "ct_secondary", 1.0))
        if s == 0:
            s = 1
        pairs.append((p, s))
    if not pairs:
        return 1.0, 1.0, False

    # Prefer a non-1:1 ratio if present, otherwise take the most common ratio.
    non_ones = [x for x in pairs if not (x[0] == 1 and x[1] == 1)]
    pool = non_ones if non_ones else pairs
    chosen = Counter(pool).most_common(1)[0][0]
    return chosen[0], chosen[1], bool(non_ones)


def _pick_representative_scale_multiplier(channels, measurement: str) -> float:
    """Pick representative absolute COMTRADE 'a' scale multiplier for a measurement group."""
    vals = []
    for ch in channels:
        if getattr(ch, "measurement", "") != measurement:
            continue
        a = abs(_f(getattr(ch, "scale_a", 0.0), 0.0))
        if a > 0:
            vals.append(a)
    if not vals:
        return 0.0
    return Counter(round(v, 6) for v in vals).most_common(1)[0][0]


def _measurement_is_primary_native(channels, measurement: str) -> bool:
    """True when the COMTRADE channels for this measurement are already marked as primary-side values."""
    flags = [
        _s(getattr(ch, "pors", ""), "").upper()
        for ch in channels
        if getattr(ch, "measurement", "") == measurement
    ]
    flags = [flag for flag in flags if flag in {"P", "S"}]
    return bool(flags) and all(flag == "P" for flag in flags)


def _infer_ratio_from_parser_multiplier(
    measurement: str,
    mult: float,
    voltage_kv=None,
    peak_current_a=None,
) -> tuple[float, float, bool]:
    """
    Convert parser scale multiplier to a practical CT/VT P/S ratio for UI defaults.
    Used only when explicit cfg primary/secondary is unavailable (often 1/1).
    """
    m = _f(mult, 0.0)
    if m <= 0:
        return 1.0, 1.0, False

    v_sys = _nearest_supported_voltage_kv(_f(voltage_kv, 150.0))
    ct_steps_by_voltage = {
        30.0: [400, 600, 800, 1000, 1200],
        75.0: [600, 800, 1000, 1200, 1500, 2000],
        150.0: [800, 1000, 1200, 1600, 2000, 2500, 3000, 4000],
        275.0: [1000, 1200, 1600, 2000, 2500, 3000, 4000],
        500.0: [1600, 2000, 2500, 3000, 4000, 5000],
    }

    if measurement == "current":
        ct_steps = ct_steps_by_voltage.get(v_sys, ct_steps_by_voltage[150.0])
        best = None
        for sec in (1.0, 5.0):
            target_primary = m * sec
            prim = min(ct_steps, key=lambda x: abs(x - target_primary))
            eff = prim / sec
            rel_err = abs(eff - m) / max(m, 1e-9)
            # tiny preference to /1 only when equally good
            score = rel_err + (0.0001 if sec == 1.0 else 0.0)
            cand = (score, float(prim), float(sec))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None and best[0] <= 0.25:
            return best[1], best[2], True
        return 1.0, 1.0, False

    if measurement == "voltage":
        vt_primary = float(int(round(v_sys * 1000.0)))
        vt_secondaries = (100.0, 110.0, 115.0, 120.0, 125.0)
        best = None
        for sec in vt_secondaries:
            eff = vt_primary / sec
            rel_err = abs(eff - m) / max(m, 1e-9)
            cand = (rel_err, vt_primary, sec)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None and best[0] <= 0.25:
            return best[1], best[2], True
        return 1.0, 1.0, False

    return 1.0, 1.0, False


def _extract_cfg_ratios(cfg_path: str) -> dict:
    """Extract representative CT and VT ratios from COMTRADE metadata."""
    try:
        rec = parse_comtrade(cfg_path)
        if rec is None:
            return {
                "cfg_ct_primary": 1.0, "cfg_ct_secondary": 1.0,
                "cfg_vt_primary": 1.0, "cfg_vt_secondary": 1.0,
                "cfg_ct_known": False, "cfg_vt_known": False,
                "ct_primary_native": False, "vt_primary_native": False,
                "parser_ct_multiplier": 0.0, "parser_vt_multiplier": 0.0,
            }
        ctp, cts, ct_known = _pick_representative_ratio(rec.analog_channels, "current")
        vtp, vts, vt_known = _pick_representative_ratio(rec.analog_channels, "voltage")
        ct_primary_native = _measurement_is_primary_native(rec.analog_channels, "current")
        vt_primary_native = _measurement_is_primary_native(rec.analog_channels, "voltage")
        ct_mult = _pick_representative_scale_multiplier(rec.analog_channels, "current")
        vt_mult = _pick_representative_scale_multiplier(rec.analog_channels, "voltage")
        return {
            "cfg_ct_primary": ctp, "cfg_ct_secondary": cts,
            "cfg_vt_primary": vtp, "cfg_vt_secondary": vts,
            "cfg_ct_known": ct_known, "cfg_vt_known": vt_known,
            "ct_primary_native": ct_primary_native, "vt_primary_native": vt_primary_native,
            "parser_ct_multiplier": ct_mult, "parser_vt_multiplier": vt_mult,
        }
    except Exception:
        return {
            "cfg_ct_primary": 1.0, "cfg_ct_secondary": 1.0,
            "cfg_vt_primary": 1.0, "cfg_vt_secondary": 1.0,
            "cfg_ct_known": False, "cfg_vt_known": False,
            "ct_primary_native": False, "vt_primary_native": False,
            "parser_ct_multiplier": 0.0, "parser_vt_multiplier": 0.0,
        }


def _nearest_supported_voltage_kv(v_kv: float) -> float:
    levels = [30.0, 75.0, 150.0, 275.0, 500.0]
    v = _f(v_kv, 150.0)
    return min(levels, key=lambda x: abs(x - v))


def _assumed_line_ratios(voltage_kv, peak_current_a=None) -> dict:
    """
    Return user-facing PLN-style CT/VT defaults for transmission line bays.
    These are display/override defaults for operators, not raw COMTRADE scale factors.
    """
    v_sys = _nearest_supported_voltage_kv(_f(voltage_kv, 150.0))
    vt_primary_by_voltage = {
        30.0: 30000.0,
        75.0: 70000.0,
        150.0: 150000.0,
        275.0: 275000.0,
        500.0: 500000.0,
    }
    ct_candidates_by_voltage = {
        30.0: [600.0, 800.0, 1200.0, 1600.0, 2000.0],
        75.0: [600.0, 800.0, 1200.0, 1600.0, 2000.0],
        150.0: [1600.0, 2000.0, 3000.0, 4000.0],
        275.0: [2000.0, 3000.0, 4000.0],
        500.0: [3000.0, 4000.0, 5000.0],
    }

    candidates = ct_candidates_by_voltage.get(v_sys, ct_candidates_by_voltage[150.0])
    peak_i = _f(peak_current_a, 0.0)
    if peak_i > 0:
        # Transmission fault current often lands several times above CT nominal.
        # Map to the closest common bay CT by using a conservative quarter-current target.
        target = max(candidates[0], peak_i / 4.0)
        ct_primary = next((val for val in candidates if val >= target), candidates[-1])
    else:
        ct_primary = candidates[min(1, len(candidates) - 1)]

    return {
        "assumed_voltage_kv": v_sys,
        "assumed_ct_primary": float(ct_primary),
        "assumed_ct_secondary": 5.0,
        "assumed_vt_primary": float(vt_primary_by_voltage.get(v_sys, 150000.0)),
        "assumed_vt_secondary": 100.0,
    }


def _assumed_transformer_ratios(voltage_kv, peak_current_a, parser_ct_multiplier=0.0, parser_vt_multiplier=0.0) -> dict:
    """
    Return practical default CT/VT ratio assumptions for transmission systems.
    Supported nominal voltage levels: 30/75/150/275/500 kV.
    VT assumption uses common secondaries: 100/110/115/120/125 V.
    CT assumption uses common secondaries: 1 A or 5 A.
    If parser multipliers are available, they are used to infer the closest
    practical P/S pair so defaults follow the COMTRADE scaling behavior.
    """
    v_sys = _nearest_supported_voltage_kv(_f(voltage_kv, 150.0))
    vt_primary = int(round(v_sys * 1000.0))
    vt_secondary = 100

    p_i = _f(peak_current_a, 0.0)
    ct_steps_by_voltage = {
        30.0: [400, 600, 800, 1000, 1200],
        75.0: [600, 800, 1000, 1200, 1500, 2000],
        150.0: [800, 1000, 1200, 1600, 2000, 2500, 3000, 4000],
        275.0: [1000, 1200, 1600, 2000, 2500, 3000, 4000],
        500.0: [1600, 2000, 2500, 3000, 4000, 5000],
    }
    ct_steps = ct_steps_by_voltage.get(v_sys, ct_steps_by_voltage[150.0])
    if p_i <= 0:
        ct_primary = ct_steps[min(2, len(ct_steps) - 1)]
    else:
        # Use a light margin so suggestion stays close to measured current.
        target = max(ct_steps[0], p_i * 1.05)
        higher = [x for x in ct_steps if x >= target]
        ct_primary = higher[0] if higher else ct_steps[-1]
    ct_secondary = 1

    # Prefer parser-based inferred ratios when explicit cfg ratios are unavailable.
    inf_ct_p, inf_ct_s, inf_ct_ok = _infer_ratio_from_parser_multiplier(
        "current", parser_ct_multiplier, voltage_kv=voltage_kv, peak_current_a=peak_current_a
    )
    inf_vt_p, inf_vt_s, inf_vt_ok = _infer_ratio_from_parser_multiplier(
        "voltage", parser_vt_multiplier, voltage_kv=voltage_kv, peak_current_a=peak_current_a
    )
    if inf_ct_ok:
        ct_primary, ct_secondary = inf_ct_p, inf_ct_s
    if inf_vt_ok:
        vt_primary, vt_secondary = inf_vt_p, inf_vt_s

    return {
        "assumed_voltage_kv": v_sys,
        "assumed_ct_primary": float(ct_primary),
        "assumed_ct_secondary": float(ct_secondary),
        "assumed_vt_primary": float(vt_primary),
        "assumed_vt_secondary": float(vt_secondary),
    }


def _resolve_ratio_defaults(
    cfg_ratios: dict,
    voltage_kv=None,
    peak_current_a=None,
    event_type: str = "LINE",
) -> dict:
    """Resolve UI ratio fields while keeping a hidden comparison baseline for manual overrides."""
    event_kind = _s(event_type, "LINE").upper()

    ct_known = bool(cfg_ratios.get("cfg_ct_known"))
    vt_known = bool(cfg_ratios.get("cfg_vt_known"))

    if event_kind == "TRANSFORMER":
        assumed = _assumed_transformer_ratios(
            voltage_kv,
            peak_current_a,
            cfg_ratios.get("parser_ct_multiplier"),
            cfg_ratios.get("parser_vt_multiplier"),
        )
    else:
        assumed = _assumed_line_ratios(voltage_kv, peak_current_a)

    ct_base_p = _f(cfg_ratios.get("cfg_ct_primary"), 1.0) if ct_known else _f(assumed.get("assumed_ct_primary"), 1.0)
    ct_base_s = _f(cfg_ratios.get("cfg_ct_secondary"), 1.0) if ct_known else _f(assumed.get("assumed_ct_secondary"), 1.0)
    vt_base_p = _f(cfg_ratios.get("cfg_vt_primary"), 1.0) if vt_known else _f(assumed.get("assumed_vt_primary"), 1.0)
    vt_base_s = _f(cfg_ratios.get("cfg_vt_secondary"), 1.0) if vt_known else _f(assumed.get("assumed_vt_secondary"), 1.0)

    return {
        "ct_primary": ct_base_p if ct_known else None,
        "ct_secondary": ct_base_s if ct_known else None,
        "vt_primary": vt_base_p if vt_known else None,
        "vt_secondary": vt_base_s if vt_known else None,
        "ct_ratio_source": "cfg" if ct_known else (
            "meta_primary" if cfg_ratios.get("ct_primary_native") else "meta"
        ),
        "vt_ratio_source": "cfg" if vt_known else (
            "meta_primary" if cfg_ratios.get("vt_primary_native") else "meta"
        ),
        "ct_base_primary": ct_base_p,
        "ct_base_secondary": ct_base_s,
        "vt_base_primary": vt_base_p,
        "vt_base_secondary": vt_base_s,
        "ct_base_assumed": not ct_known,
        "vt_base_assumed": not vt_known,
    }


def _recalculate_analysis_with_ratio(analysis: dict, ct_p: float, ct_s: float, vt_p: float, vt_s: float) -> dict:
    """Scale electrical values using CT/VT P/S and re-run rule/ML classification."""
    updated = dict(analysis)

    # Preserve original analysis snapshot once (first recalculation only).
    if "baseline_label" not in updated:
        updated["baseline_label"] = analysis.get("label")
        updated["baseline_confidence"] = _f(analysis.get("confidence"))
        updated["baseline_tier"] = int(analysis.get("tier") or 0)
        updated["baseline_rule_name"] = analysis.get("rule_name")
        updated["baseline_evidence"] = analysis.get("evidence")
        updated["baseline_recommendation"] = analysis.get("recommendation")
        updated["baseline_description"] = analysis.get("description", "")
        updated["baseline_cause_pcts"] = analysis.get("cause_pcts", [])

    # Keep immutable numeric baseline so repeated overrides do not compound.
    # Also store the ORIGINAL file ratio so we can compute a relative scaling factor.
    if not isinstance(updated.get("ratio_base_values"), dict):
        updated["ratio_base_values"] = {
            "peak_fault_current_a": _f(analysis.get("peak_fault_current_a")),
            "i0_magnitude_a": _f(analysis.get("i0_magnitude_a")),
            "i1_magnitude_a": _f(analysis.get("i1_magnitude_a")),
            "i2_magnitude_a": _f(analysis.get("i2_magnitude_a")),
            "v_prefault_v": _f(analysis.get("v_prefault_v")),
            "v_fault_v": _f(analysis.get("v_fault_v")),
            "z_magnitude_ohms": _f(analysis.get("z_magnitude_ohms")),
            # Original file ratio (used to compute relative change)
            "orig_ct_p": _f(analysis.get("ratio_base_ct_primary"), _f(analysis.get("ct_primary"), 1.0)),
            "orig_ct_s": _f(analysis.get("ratio_base_ct_secondary"), _f(analysis.get("ct_secondary"), 1.0)),
            "orig_vt_p": _f(analysis.get("ratio_base_vt_primary"), _f(analysis.get("vt_primary"), 1.0)),
            "orig_vt_s": _f(analysis.get("ratio_base_vt_secondary"), _f(analysis.get("vt_secondary"), 1.0)),
        }
    base = updated["ratio_base_values"]

    # Relative scaling: how much does the new ratio differ from the original file ratio?
    # Base values are already in primary units (original ratio applied by parser).
    # If user enters same ratio as file → factor = 1.0 (no change).
    # If user enters different ratio → scale proportionally.
    orig_ct = _ratio(_f(base.get("orig_ct_p"), 1.0), _f(base.get("orig_ct_s"), 1.0))
    orig_vt = _ratio(_f(base.get("orig_vt_p"), 1.0), _f(base.get("orig_vt_s"), 1.0))
    ct_mul = (_ratio(ct_p, ct_s) / orig_ct) if orig_ct > 0 else _ratio(ct_p, ct_s)
    vt_mul = (_ratio(vt_p, vt_s) / orig_vt) if orig_vt > 0 else _ratio(vt_p, vt_s)

    if "ct_primary_default" not in updated:
        updated["ct_primary_default"] = analysis.get("ct_primary_default")
    if "ct_secondary_default" not in updated:
        updated["ct_secondary_default"] = analysis.get("ct_secondary_default")
    if "vt_primary_default" not in updated:
        updated["vt_primary_default"] = analysis.get("vt_primary_default")
    if "vt_secondary_default" not in updated:
        updated["vt_secondary_default"] = analysis.get("vt_secondary_default")

    def _same_num(left: float, right: float) -> bool:
        return abs(_f(left) - _f(right)) < 1e-9

    returning_to_base = (
        _same_num(ct_p, _f(base.get("orig_ct_p"), 1.0))
        and _same_num(ct_s, _f(base.get("orig_ct_s"), 1.0))
        and _same_num(vt_p, _f(base.get("orig_vt_p"), 1.0))
        and _same_num(vt_s, _f(base.get("orig_vt_s"), 1.0))
    )

    # Persist ratio inputs for the UI.
    if returning_to_base:
        updated["ct_primary"] = updated.get("ct_primary_default")
        updated["ct_secondary"] = updated.get("ct_secondary_default")
        updated["vt_primary"] = updated.get("vt_primary_default")
        updated["vt_secondary"] = updated.get("vt_secondary_default")
        updated["ratio_override_active"] = False
    else:
        updated["ct_primary"] = ct_p
        updated["ct_secondary"] = ct_s
        updated["vt_primary"] = vt_p
        updated["vt_secondary"] = vt_s
        updated["ratio_override_active"] = True
    updated["ct_multiplier"] = ct_mul
    updated["vt_multiplier"] = vt_mul

    # Scale from immutable baseline (absolute override), not from previous override result.
    for k in ("peak_fault_current_a", "i0_magnitude_a", "i1_magnitude_a", "i2_magnitude_a"):
        updated[k] = _f(base.get(k)) * ct_mul
    for k in ("v_prefault_v", "v_fault_v"):
        updated[k] = _f(base.get(k)) * vt_mul

    # Derived electrical fields
    i1 = _f(updated.get("i1_magnitude_a"))
    i0 = _f(updated.get("i0_magnitude_a"))
    i2 = _f(updated.get("i2_magnitude_a"))
    if i1 > 0:
        updated["i0_i1_ratio"] = i0 / i1
        updated["i2_i1_ratio"] = i2 / i1
    updated["scaling_ok"] = _f(updated.get("peak_fault_current_a")) >= 200.0

    # Z = V / I, so primary-side conversion scales by VT/CT
    z_scale = (vt_mul / ct_mul) if ct_mul > 0 else 1.0
    updated["z_magnitude_ohms"] = _f(base.get("z_magnitude_ohms")) * z_scale

    # Re-run classification logic on scaled features
    row = {
        "fault_duration_ms": _f(updated.get("fault_duration_ms")),
        "fault_count": int(updated.get("fault_count") or 1),
        "i0_i1_ratio": _f(updated.get("i0_i1_ratio")),
        "voltage_sag_depth_pu": _f(updated.get("voltage_sag_depth_pu")),
        "di_dt_max": _f(updated.get("di_dt_max")),
        "peak_fault_current_a": _f(updated.get("peak_fault_current_a")),
        "reclose_successful": updated.get("reclose_successful"),
        "trip_type": _s(updated.get("trip_type"), ""),
        "faulted_phases": _s(updated.get("faulted_phases"), ""),
        "fault_type": _s(updated.get("fault_type"), ""),
        "reclose_time_ms": _f(updated.get("reclose_time_ms")),
        "thd_percent": _f(updated.get("thd_percent")),
        "inception_angle_degrees": _f(updated.get("inception_angle_degrees"), -1),
    }

    rule_result = apply_rules(row)
    if rule_result is not None:
        updated["label"] = rule_result.label
        updated["confidence"] = _f(rule_result.confidence)
        updated["tier"] = 1
        updated["rule_name"] = rule_result.rule_name
        updated["evidence"] = f"{rule_result.evidence} | Analisa ulang dengan rasio CT/VT."
        updated["cause_pcts"] = _compute_cause_pcts(row)
        updated["recommendation"] = (
            "Lakukan inspeksi lapangan sesuai indikasi gangguan permanen atau kerusakan konduktor/tower."
            if "PERMANEN" in rule_result.label.upper() or "KONDUKTOR" in rule_result.label.upper()
            else updated.get("recommendation", "")
        )
        updated["description"] = (
            (updated.get("baseline_description") or updated.get("description", "")) + "\n\n"
            if (updated.get("baseline_description") or updated.get("description", ""))
            else ""
        ) + (
            f"Analisis dihitung ulang dengan rasio CT {ct_p:g}/{ct_s:g} dan VT {vt_p:g}/{vt_s:g}. "
            f"Hasil terbaru: {updated['label']} dengan keyakinan {updated['confidence']*100:.0f}%."
        )
        return _json_safe(updated)

    model_bundle = _load_model()
    if model_bundle is not None:
        clf        = model_bundle["clf"]
        classes    = model_bundle.get("classes", [])
        model_type = model_bundle.get("model_type", "binary")
        X          = _build_feature_vector(row)
        pred_raw   = clf.predict(X)[0]
        proba      = clf.predict_proba(X)[0]

        proba_classes = list(getattr(clf, "classes_", classes))
        if model_type in MULTICLASS_MODEL_TYPES and proba_classes:
            from models.predict import _label_recommendation
            label_id = MULTICLASS_LABEL_MAP.get(pred_raw, pred_raw)
            reclose_ok = _to_bool_or_none(row.get("reclose_successful"))
            reclose_note = "  AR berhasil — transien terkonfirmasi." if (
                reclose_ok is True and row["peak_fault_current_a"] > 200) else ""
            updated["label"] = label_id
            updated["confidence"] = float(proba.max())
            updated["tier"] = 2
            updated["rule_name"] = model_type
            updated["cause_pcts"] = [
                {"name": MULTICLASS_LABEL_MAP.get(c, c), "pct": round(float(p) * 100, 1)}
                for c, p in sorted(zip(proba_classes, proba), key=lambda x: -x[1])
            ]
            updated["recommendation"] = _label_recommendation(pred_raw)
            updated["evidence"] = f"ML multi-kelas: {pred_raw} ({updated['confidence']:.0%}).{reclose_note} Dihitung ulang dengan rasio CT/VT."
        else:
            pred = int(pred_raw)
            p_trans = float(proba[1])
            updated["label"] = "GANGGUAN TRANSIEN"
            updated["confidence"] = p_trans if pred == 1 else max(p_trans, 0.5)
            updated["tier"] = 2
            updated["rule_name"] = "petir_decision_tree" if pred == 1 else "petir_decision_tree_non_petir"
            updated["cause_pcts"] = _compute_cause_pcts(row)
            updated["recommendation"] = _transient_recommendation(row)
            updated["evidence"] = "Analisa ulang ML (binary) berbasis fitur yang sudah dikoreksi rasio CT/VT."
        updated["description"] = (
            (updated.get("baseline_description") or updated.get("description", "")) + "\n\n"
            if (updated.get("baseline_description") or updated.get("description", ""))
            else ""
        ) + (
            f"Analisis dihitung ulang dengan rasio CT {ct_p:g}/{ct_s:g} dan VT {vt_p:g}/{vt_s:g}. "
            f"Hasil terbaru: {updated['label']} dengan keyakinan {updated['confidence']*100:.0f}%."
        )
        return _json_safe(updated)

    updated["label"] = "PERLU INVESTIGASI"
    updated["confidence"] = 0.0
    updated["tier"] = 0
    updated["rule_name"] = "no_model"
    updated["cause_pcts"] = []
    updated["evidence"] = "Model ML tidak tersedia untuk perhitungan ulang."
    updated["description"] = (
        f"Analisis dihitung ulang dengan rasio CT {ct_p:g}/{ct_s:g} dan VT {vt_p:g}/{vt_s:g}, "
        "namun model ML belum tersedia."
    )
    return _json_safe(updated)

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
app.secret_key = "dfr-fault-classifier-2026"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["PROPAGATE_EXCEPTIONS"] = False
IS_LOCAL_DEV = not bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("VERCEL"))

if IS_LOCAL_DEV:
    app.config["PROPAGATE_EXCEPTIONS"] = True
    app.logger.setLevel(logging.INFO)

    @app.before_request
    def _log_request():
        print(f"[REQ] {request.method} {request.path}", flush=True)
        app.logger.info("REQ %s %s", request.method, request.path)

    @app.after_request
    def _log_response(resp):
        print(f"[RES] {request.method} {request.path} -> {resp.status_code}", flush=True)
        app.logger.info("RES %s %s -> %s", request.method, request.path, resp.status_code)
        return resp

import tempfile
UPLOAD_DIR  = Path(tempfile.gettempdir()) / "dfr_uploads"
ANALYSIS_DIR = Path(tempfile.gettempdir()) / "dfr_analysis"
HISTORY_CSV = Path(__file__).parent / "history.csv"
UPLOAD_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

HISTORY_FIELDS = [
    "timestamp",
    "filename",
    "station",
    "predicted_label",
    "predicted_cause_top",
    "predicted_conf",
    "tier",
    "rule_name",
    "confirmed_label",
    "correct",
    "notes",
    "zone",
    "phases",
    "duration_ms",
    "fault_count",
    "peak_current_a",
    "i0_i1_ratio",
    "reclose_ok",
]

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
# psycopg3 requires postgresql:// scheme; Railway sometimes provides postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]


def _db_enabled() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _db_connect():
    if not _db_enabled():
        return None
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5)


def _db_init():
    if not _db_enabled():
        app.logger.info("Postgres disabled (DATABASE_URL/psycopg unavailable), using CSV history.")
        return
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_feedback (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        timestamp_txt TEXT,
                        filename TEXT,
                        station TEXT,
                        predicted_label TEXT,
                        predicted_cause_top TEXT,
                        predicted_conf TEXT,
                        tier TEXT,
                        rule_name TEXT,
                        confirmed_label TEXT,
                        correct TEXT,
                        notes TEXT,
                        zone TEXT,
                        phases TEXT,
                        duration_ms TEXT,
                        fault_count TEXT,
                        peak_current_a TEXT,
                        i0_i1_ratio TEXT,
                        reclose_ok TEXT,
                        source_ip TEXT,
                        user_agent TEXT
                    )
                    """
                )
            # ── Phase 2: Transformer event table ─────────────────────────
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transformer_feedback (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    timestamp_txt TEXT,
                    filename TEXT,
                    station TEXT,
                    relay_family TEXT,
                    event_class TEXT,
                    predicted_label TEXT,
                    predicted_conf TEXT,
                    rule_name TEXT,
                    confirmed_label TEXT,
                    correct TEXT,
                    notes TEXT,
                    h2_ratio_max_pct TEXT,
                    h5_ratio_max_pct TEXT,
                    idiff_max_pu TEXT,
                    irstr_max_pu TEXT,
                    slope_worst_pct TEXT,
                    dc_offset_index_max TEXT,
                    energisation_flag TEXT,
                    fault_duration_ms TEXT,
                    action_required TEXT,
                    source_ip TEXT,
                    user_agent TEXT
                )
                """
            )
            # Migrate existing analysis_feedback: add event_type column if missing
            cur.execute(
                """
                ALTER TABLE analysis_feedback
                ADD COLUMN IF NOT EXISTS event_type TEXT DEFAULT 'LINE'
                """
            )
        app.logger.info("Postgres history backend is ready.")
    except Exception as e:
        app.logger.warning("Postgres init failed, fallback to CSV: %s", e)


def _db_insert_feedback(row: dict, source_ip: str, user_agent: str) -> bool:
    if not _db_enabled():
        return False
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_feedback (
                        timestamp_txt, filename, station, predicted_label, predicted_cause_top,
                        predicted_conf, tier, rule_name, confirmed_label, correct, notes,
                        zone, phases, duration_ms, fault_count, peak_current_a, i0_i1_ratio,
                        reclose_ok, source_ip, user_agent
                    ) VALUES (
                        %(timestamp)s, %(filename)s, %(station)s, %(predicted_label)s, %(predicted_cause_top)s,
                        %(predicted_conf)s, %(tier)s, %(rule_name)s, %(confirmed_label)s, %(correct)s, %(notes)s,
                        %(zone)s, %(phases)s, %(duration_ms)s, %(fault_count)s, %(peak_current_a)s, %(i0_i1_ratio)s,
                        %(reclose_ok)s, %(source_ip)s, %(user_agent)s
                    )
                    """,
                    {
                        **{k: _s(row.get(k), "") for k in HISTORY_FIELDS},
                        "source_ip": _s(source_ip, ""),
                        "user_agent": _s(user_agent, ""),
                    },
                )
        return True
    except Exception as e:
        app.logger.warning("Postgres insert failed, fallback to CSV: %s", e)
        return False


def _db_fetch_feedback(limit: int | None = None) -> list[dict]:
    if not _db_enabled():
        return []
    try:
        q = (
            "SELECT timestamp_txt, filename, station, predicted_label, predicted_cause_top, "
            "predicted_conf, tier, rule_name, confirmed_label, correct, notes, zone, phases, "
            "duration_ms, fault_count, peak_current_a, i0_i1_ratio, reclose_ok "
            "FROM analysis_feedback ORDER BY id DESC"
        )
        params = ()
        if limit is not None:
            q += " LIMIT %s"
            params = (int(limit),)

        rows = []
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(q, params)
                for r in cur.fetchall():
                    rows.append({
                        "timestamp": _s(r[0], ""),
                        "filename": _s(r[1], ""),
                        "station": _s(r[2], ""),
                        "predicted_label": _s(r[3], ""),
                        "predicted_cause_top": _s(r[4], ""),
                        "predicted_conf": _s(r[5], "0"),
                        "tier": _s(r[6], "0"),
                        "rule_name": _s(r[7], ""),
                        "confirmed_label": _s(r[8], ""),
                        "correct": _s(r[9], ""),
                        "notes": _s(r[10], ""),
                        "zone": _s(r[11], ""),
                        "phases": _s(r[12], ""),
                        "duration_ms": _s(r[13], "0"),
                        "fault_count": _s(r[14], "0"),
                        "peak_current_a": _s(r[15], "0"),
                        "i0_i1_ratio": _s(r[16], "0"),
                        "reclose_ok": _s(r[17], ""),
                    })
        return rows
    except Exception as e:
        app.logger.warning("Postgres fetch failed, fallback to CSV: %s", e)
        return []


def _load_history_rows(limit: int | None = None, newest_first: bool = True) -> list[dict]:
    rows = _db_fetch_feedback(limit if newest_first else None)
    if rows:
        if not newest_first:
            rows.reverse()
        return rows[:limit] if limit is not None else rows

    _ensure_history_schema()
    csv_rows = []
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    if newest_first:
        csv_rows.reverse()
    if limit is not None:
        csv_rows = csv_rows[:limit]
    return csv_rows

FAULT_CATEGORIES = [
    "PETIR",
    "LAYANG-LAYANG",
    "POHON",
    "HEWAN",
    "KONDUKTOR / TOWER",
    "PERALATAN / PROTEKSI",
    "BENDA ASING",
    "GANGGUAN PERMANEN",
    "LAIN-LAIN",
]

import threading as _threading
_threading.Thread(target=_db_init, daemon=True).start()


@app.errorhandler(Exception)
def _handle_unexpected_error(e):
    """
    Improve local debugging visibility while keeping hosted behavior safe.
    """
    if isinstance(e, HTTPException):
        return e

    app.logger.error("Unhandled exception: %s", e)
    app.logger.error(traceback.format_exc())
    if IS_LOCAL_DEV:
        print("[ERR] Unhandled exception:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)

    if IS_LOCAL_DEV:
        tb_tail = traceback.format_exc().splitlines()[-20:]
        return jsonify({
            "error": "internal_server_error",
            "message": str(e),
            "path": request.path,
            "traceback_tail": tb_tail,
        }), 500

    return (
        "Internal Server Error: application exception occurred. "
        "Check platform logs for traceback.",
        500,
    )


# ── routes ────────────────────────────────────────────────────────────────────

RAW_DATA = Path(__file__).parent.parent.parent / "raw_data"
SKIP_FRAGS = ["olah", "_extracted", "locus", "analisa"]

def _scan_recordings():
    """Return list of dicts describing discoverable CFGs in raw_data/."""
    events = []
    seen = set()
    if not RAW_DATA.exists():
        return events
    for cfg in sorted(RAW_DATA.rglob("*.cfg")) + sorted(RAW_DATA.rglob("*.CFG")):
        key = str(cfg.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        path_str = str(cfg)
        if any(f in path_str.lower() for f in SKIP_FRAGS):
            continue
        dat = cfg.with_suffix(".dat")
        if not dat.exists():
            dat = cfg.with_suffix(".DAT")
        if not dat.exists():
            continue
        # Extract UPT and year from path parts
        parts = cfg.parts
        upt  = next((p for p in parts if p.startswith("UPT ")), "-")
        year = next((p for p in parts if re.fullmatch(r"\d{4}", p)), "-")
        # Event folder = part after UPT/year/month (index 3), month = index 2
        try:
            rel   = cfg.relative_to(RAW_DATA)
            month = rel.parts[2] if len(rel.parts) > 2 else "-"
            gi    = rel.parts[3] if len(rel.parts) > 3 else rel.parts[-1]
        except Exception:
            month = "-"
            gi    = "-"
        status_data = infer_status_data(path_str)
        suspected_label = infer_suspected_label(path_str)
        path_tag = infer_path_tag(path_str)
        events.append({
            "upt": upt,
            "year": year,
            "month": month,
            "gi": gi,
            "filename": cfg.name,
            "cfg_path": str(cfg),
            "status_data": status_data,
            "suspected_label": suspected_label,
            "path_tag": path_tag,
            "kind": infer_path_kind(path_str),
            "search_text": (
                f"{path_str} {upt} {year} {month} {gi} {cfg.name} "
                f"{status_data} {suspected_label} {path_tag}"
            ),
        })
    return events


def _render_not_supported(filename: str, error_msg: str, soe: list = None, cfg_path: str = None):
    """Render a friendly notification page for unsupported file types."""
    msg = error_msg.lower()
    if "differential" in msg or "diferensial" in msg or "87l" in msg or "diff" in msg:
        title  = "Gangguan dideteksi dengan Rele Diferensial Penghantar"
        detail = ("Rekaman ini berasal dari rele diferensial (87L). "
                  "Model AI saat ini hanya mendukung analisis rele jarak (distance relay / 21). "
                  "Rele diferensial menggunakan arus diferensial sebagai besaran ukur, "
                  "sehingga memerlukan pendekatan fitur yang berbeda.")
        tips   = ["Proteksi diferensial umumnya digunakan pada penghantar transmisi jarak jauh",
                  "Data rekaman ini tetap berharga — simpan untuk pengembangan model berikutnya",
                  "Gunakan Pilih Rekaman untuk memilih file dari rele jarak"]
    elif "directional" in msg or "67n" in msg or "earth fault" in msg:
        title  = "Gangguan dideteksi dengan Rele Arah Hubung Tanah"
        detail = ("Rekaman ini berasal dari rele terarah hubung tanah (67N/EF). "
                  "Model AI saat ini hanya mendukung rele jarak (distance relay / 21).")
        tips   = ["Rele 67N digunakan untuk gangguan hubung tanah sensitif",
                  "Analisis manual tetap dapat dilakukan dari tampilan osilografi"]
    elif "no fault" in msg or "tidak" in msg.lower():
        title  = "Gangguan tidak terdeteksi dalam rekaman"
        detail = ("Pipeline tidak menemukan event gangguan yang jelas dalam file ini. "
                  "Kemungkinan file adalah rekaman normal (bukan gangguan), "
                  "atau sinyal terlalu kecil untuk dideteksi secara otomatis.")
        tips   = ["Periksa apakah file .cfg dan .dat sudah benar (nama sama)",
                  "Pastikan file adalah rekaman saat gangguan, bukan kondisi normal"]
    elif "parse" in msg or "load" in msg or "unpack" in msg:
        title  = "File COMTRADE tidak dapat dibaca"
        detail = ("Format file tidak dikenali atau file rusak. "
                  "Beberapa format COMTRADE lama atau file dari DFR internal "
                  "tidak kompatibel dengan parser yang digunakan.")
        tips   = ["Pastikan file .cfg dan .dat tidak terpisah atau rusak",
                  "File dari CSC101M, DFR internal, atau format non-standar belum didukung"]
    else:
        title  = "File tidak dapat dianalisis"
        detail = f"Alasan: {error_msg}"
        tips   = ["Coba file dari rele jarak (distance relay) dengan format COMTRADE standar"]

    # Attempt SOE extraction if not already provided
    if soe is None and cfg_path:
        try:
            soe = extract_soe_from_file(cfg_path)
        except Exception:
            soe = []

    return render_template("tidak_didukung.html",
                           filename=filename,
                           reason_title=title,
                           reason_detail=detail,
                           tips=tips,
                           soe=soe or [])


def _build_analysis_payload(result, original_filename: str, ts: str, cfg_path: str) -> dict:
    feats = result.features
    cfg_ratios = _extract_cfg_ratios(str(cfg_path))
    ratio_defaults = _resolve_ratio_defaults(
        cfg_ratios,
        feats.get("voltage_kv"),
        feats.get("peak_fault_current_a"),
        event_type=getattr(result, "event_type", "LINE"),
    )
    ct_ratio_src = ratio_defaults["ct_ratio_source"]
    vt_ratio_src = ratio_defaults["vt_ratio_source"]
    ct_p_display = ratio_defaults["ct_primary"]
    ct_s_display = ratio_defaults["ct_secondary"]
    vt_p_display = ratio_defaults["vt_primary"]
    vt_s_display = ratio_defaults["vt_secondary"]
    ct_p_base = ratio_defaults["ct_base_primary"]
    ct_s_base = ratio_defaults["ct_base_secondary"]
    vt_p_base = ratio_defaults["vt_base_primary"]
    vt_s_base = ratio_defaults["vt_base_secondary"]

    analysis_payload = {
        "original_filename": original_filename,
        "timestamp": ts,
        "label":          result.label,
        "confidence":     _f(result.confidence),
        "tier":           int(result.tier),
        "rule_name":      result.rule_name,
        "evidence":       result.evidence,
        "recommendation": result.recommendation,
        "description":    result.description or "",
        "cause_pcts":     result.cause_pcts or [],
        "soe":            result.soe or [],
        "di_dt_max":      _f(feats.get("di_dt_max")),
        "thd_percent":    _f(feats.get("thd_percent")),
        "inception_angle_degrees": _f(feats.get("inception_angle_degrees"), -1),
        "station_name":         _s(feats.get("station_name")),
        "relay_model":          _s(feats.get("relay_model")),
        "zone_operated":        _s(feats.get("zone_operated")),
        "trip_type":            _s(feats.get("trip_type")),
        "faulted_phases":       _s(feats.get("faulted_phases")),
        "fault_type":           _s(feats.get("fault_type")),
        "fault_duration_ms":    _f(feats.get("fault_duration_ms")),
        "fault_inception_ms":   _f(feats.get("fault_inception_ms")),
        "record_duration_ms":   _f(feats.get("record_duration_ms")),
        "fault_count":          int(feats.get("fault_count") or 0),
        "peak_fault_current_a": _f(feats.get("peak_fault_current_a")),
        "peak_fault_phase":     _s(feats.get("peak_fault_phase")),
        "i0_i1_ratio":          _f(feats.get("i0_i1_ratio")),
        "i0_magnitude_a":       _f(feats.get("i0_magnitude_a")),
        "i1_magnitude_a":       _f(feats.get("i1_magnitude_a")),
        "i2_magnitude_a":       _f(feats.get("i2_magnitude_a")),
        "voltage_sag_depth_pu": _f(feats.get("voltage_sag_depth_pu")),
        "voltage_sag_phase":    _s(feats.get("voltage_sag_phase")),
        "v_prefault_v":         _f(feats.get("v_prefault_v")),
        "v_fault_v":            _f(feats.get("v_fault_v")),
        "z_magnitude_ohms":     _f(feats.get("z_magnitude_ohms")),
        "z_angle_degrees":      _f(feats.get("z_angle_degrees")),
        "r_x_ratio":            _f(feats.get("r_x_ratio")),
        "reclose_successful":   _s(feats.get("reclose_successful")),
        "reclose_time_ms":      _f(feats.get("reclose_time_ms")),
        "voltage_kv":           str(int(_nearest_supported_voltage_kv(vt_p_base / 1000.0))) if (vt_ratio_src == "cfg" and vt_p_base > 1000.0) else (_s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui"),
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
        "ct_primary": ct_p_display,
        "ct_secondary": ct_s_display,
        "vt_primary": vt_p_display,
        "vt_secondary": vt_s_display,
        "ct_primary_default": ct_p_display,
        "ct_secondary_default": ct_s_display,
        "vt_primary_default": vt_p_display,
        "vt_secondary_default": vt_s_display,
        "ratio_base_ct_primary": ct_p_base,
        "ratio_base_ct_secondary": ct_s_base,
        "ratio_base_vt_primary": vt_p_base,
        "ratio_base_vt_secondary": vt_s_base,
        "ratio_override_active": False,
        "cfg_ct_primary": _f(cfg_ratios.get("cfg_ct_primary"), 1.0),
        "cfg_ct_secondary": _f(cfg_ratios.get("cfg_ct_secondary"), 1.0),
        "cfg_vt_primary": _f(cfg_ratios.get("cfg_vt_primary"), 1.0),
        "cfg_vt_secondary": _f(cfg_ratios.get("cfg_vt_secondary"), 1.0),
        "cfg_ct_known": bool(cfg_ratios.get("cfg_ct_known")),
        "cfg_vt_known": bool(cfg_ratios.get("cfg_vt_known")),
        "parser_ct_multiplier": _f(cfg_ratios.get("parser_ct_multiplier"), 0.0),
        "parser_vt_multiplier": _f(cfg_ratios.get("parser_vt_multiplier"), 0.0),
        "ct_ratio_source": ct_ratio_src,
        "vt_ratio_source": vt_ratio_src,
        "ct_ratio_known":  bool(cfg_ratios.get("cfg_ct_known")),
        "vt_ratio_known":  bool(cfg_ratios.get("cfg_vt_known")),
        "ratio_base_values": {
            "peak_fault_current_a": _f(feats.get("peak_fault_current_a")),
            "i0_magnitude_a": _f(feats.get("i0_magnitude_a")),
            "i1_magnitude_a": _f(feats.get("i1_magnitude_a")),
            "i2_magnitude_a": _f(feats.get("i2_magnitude_a")),
            "v_prefault_v": _f(feats.get("v_prefault_v")),
            "v_fault_v": _f(feats.get("v_fault_v")),
            "z_magnitude_ohms": _f(feats.get("z_magnitude_ohms")),
            "orig_ct_p": ct_p_base,
            "orig_ct_s": ct_s_base,
            "orig_vt_p": vt_p_base,
            "orig_vt_s": vt_s_base,
        },
    }
    _inject_transformer_fields(result, analysis_payload)
    analysis_payload["waveform_data"] = _build_waveform_payload(
        str(cfg_path),
        analysis_payload["fault_inception_ms"],
        analysis_payload["fault_duration_ms"],
    )
    analysis_payload["waveform_b64"] = _generate_waveform_plot(
        str(cfg_path),
        analysis_payload["fault_inception_ms"],
        analysis_payload["fault_duration_ms"],
    )
    return analysis_payload


def _apply_transformer_intake_context(payload: dict, relay_role: str, analysis_goal: str,
                                      intake_mode: str, supporting_relays: list[dict]) -> None:
    role_label = _transformer_role_label(relay_role)
    goal_label = _transformer_goal_label(analysis_goal)
    context_only = payload.get("event_type") != "TRANSFORMER"

    payload["intake_domain"] = "TRANSFORMER"
    payload["transformer_intake_mode"] = intake_mode
    payload["transformer_intake_mode_label"] = "Coordinated multi-relay" if intake_mode == "COORDINATED" else "Quick local analysis"
    payload["transformer_relay_role"] = relay_role
    payload["transformer_relay_role_label"] = role_label
    payload["transformer_analysis_goal"] = analysis_goal
    payload["transformer_analysis_goal_label"] = goal_label
    payload["supporting_relays"] = supporting_relays
    payload["transformer_context_only"] = context_only
    payload["transformer_context_title"] = f"ANALISIS RELAY {role_label.upper()}"
    payload["transformer_primary_result_label"] = payload.get("label")
    payload["transformer_primary_confidence"] = payload.get("confidence")

    role_notes = {
        "OCR_LV": (
            "Rekaman OCR LV terutama berguna untuk membaca operasi proteksi sisi distribusi atau incomer. "
            "File ini sendiri tidak cukup untuk membuktikan internal transformer fault."
        ),
        "OCR_HV": (
            "Rekaman OCR HV paling berguna untuk menilai backup operation, selektivitas, dan apakah sisi HV ikut trip secara wajar."
        ),
        "REF": (
            "REF lebih relevan untuk ground fault dekat zona trafo/NGR. Gunakan bersama GFR HV atau SBEF bila origin masih meragukan."
        ),
        "GFR_HV": (
            "GFR HV membantu membaca jalur gangguan tanah dan koordinasi sisi HV. Kesimpulan final tetap lebih kuat bila disandingkan dengan REF atau relay sisi LV."
        ),
        "SBEF": (
            "SBEF berguna untuk kasus ground fault sensitif dan isu NGR. Interpretasi terbaik memerlukan urutan trip dengan REF/GFR HV."
        ),
        "FEEDER": (
            "Relay feeder/outgoing biasanya paling kuat untuk mengonfirmasi apakah gangguan berasal dari downstream/distribusi."
        ),
        "87T": (
            "87T memberi bukti lokal paling kuat untuk membedakan inrush, through-fault, atau indikasi internal transformer fault."
        ),
    }
    payload["transformer_context_note"] = role_notes.get(
        relay_role,
        "File ini dibaca sebagai satu potong evidence dalam event trafo. Gunakan bersama relay lain bila tujuan analisis adalah origin fault atau review selektivitas."
    )

    next_files = {
        "OCR_LV": ["87T", "OCR HV", "REF / SBEF bila ground fault"],
        "OCR_HV": ["87T", "OCR LV / incomer", "Outgoing / feeder relay"],
        "REF": ["GFR HV", "SBEF", "87T"],
        "GFR_HV": ["REF", "SBEF", "OCR LV / feeder"],
        "SBEF": ["REF", "GFR HV", "87T"],
        "FEEDER": ["OCR LV / incomer", "87T bila trafo ikut trip", "OCR HV bila ada backup trip"],
        "87T": ["OCR HV", "OCR LV / incomer", "REF / SBEF bila earth fault"],
    }
    payload["transformer_next_files"] = next_files.get(relay_role, ["87T", "OCR HV", "OCR LV / incomer"])

    if context_only:
        payload["event_type"] = "TRANSFORMER_CONTEXT"


def _generate_waveform_plot(cfg_path: str, inception_ms: float, duration_ms: float) -> str:
    """
    Render voltage, current, and active digital channels from a COMTRADE file
    as a dark-themed matplotlib figure. Returns a base64-encoded PNG or '' on failure.
    Active digital = channels that changed state at least once in the recording.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D
        import numpy as np
    except ImportError:
        return ""
    try:
        record = parse_comtrade(cfg_path)
        if record is None or len(record.time) == 0:
            return ""

        t_ms = record.time * 1000   # seconds → milliseconds
        # Shift so fault inception = t=0 (consistent with interactive Plotly viewer)
        t0 = float(inception_ms) if inception_ms else 0.0
        if t0 == 0.0 and len(t_ms):
            t0 = float(t_ms[0])
        t_ms = t_ms - t0

        v_chs = [ch for ch in record.analog_channels
                 if ch.measurement == 'voltage' and len(ch.samples) == len(record.time)]
        i_chs = [ch for ch in record.analog_channels
                 if ch.measurement == 'current' and len(ch.samples) == len(record.time)]
        d_chs = [ch for ch in record.status_channels
                 if len(ch.samples) == len(record.time) and len(np.unique(ch.samples)) > 1]

        panels = [(chs, pt) for chs, pt in [(v_chs, 'v'), (i_chs, 'i'), (d_chs, 'd')] if chs]
        if not panels:
            return ""

        h_ratios = [max(2, min(len(c), 8)) if pt == 'd' else 3 for c, pt in panels]

        # Light/print-friendly palette
        BG      = '#ffffff'
        AX_BG   = '#f8fafc'
        GRID_C  = '#e2e8f0'
        MUTED   = '#64748b'
        BORDER  = '#cbd5e1'
        CV = ['#dc2626', '#2563eb', '#059669']          # Va Vb Vc — bold primaries
        CI = ['#ea580c', '#0891b2', '#7c3aed', '#ca8a04']  # Ia Ib Ic IN

        fig_h = max(4, sum(h_ratios) * 0.95 + 0.5)
        fig = plt.figure(figsize=(13, fig_h), facecolor=BG)
        gs  = gridspec.GridSpec(len(panels), 1, figure=fig,
                                height_ratios=h_ratios, hspace=0.10)
        axes = [fig.add_subplot(gs[i]) for i in range(len(panels))]

        # After shifting t_ms, inception is at t=0 and clearing at t=duration_ms
        t_in  = 0.0
        t_out = float(duration_ms)

        def _style(ax, ylabel):
            ax.set_facecolor(AX_BG)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
            ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
            for s in ax.spines.values():
                s.set_edgecolor(BORDER)
            ax.grid(True, alpha=0.6, color=GRID_C, linewidth=0.6)
            ax.set_xlim(t_ms[0], t_ms[-1])

        def _markers(ax):
            ax.axvline(t_in,  color='#d97706', lw=1.2, ls='--', alpha=0.9, zorder=5)
            ax.axvline(t_out, color='#dc2626', lw=1.2, ls='--', alpha=0.9, zorder=5)
            ax.axvspan(t_in, t_out, alpha=0.07, color='#f59e0b', zorder=1)

        leg_extra = [
            Line2D([0], [0], color='#d97706', lw=1.4, ls='--', label='Inception'),
            Line2D([0], [0], color='#dc2626', lw=1.4, ls='--', label='Clearing'),
        ]
        leg_kw = dict(fontsize=7.5, facecolor='white', edgecolor=BORDER,
                      labelcolor='#334155', framealpha=0.92)

        for ax_i, (chs, ptype) in enumerate(panels):
            ax = axes[ax_i]
            if ptype == 'v':
                _style(ax, 'Tegangan (kV)')
                for j, ch in enumerate(chs[:3]):
                    ax.plot(t_ms, ch.samples, color=CV[j % 3], lw=0.85,
                            label=ch.canonical_name or ch.name, alpha=0.95)
                _markers(ax)
                h, l = ax.get_legend_handles_labels()
                ax.legend(h + leg_extra, l + ['Inception', 'Clearing'],
                          loc='upper right', **leg_kw)

            elif ptype == 'i':
                _style(ax, 'Arus (A)')
                for j, ch in enumerate(chs[:4]):
                    ax.plot(t_ms, ch.samples, color=CI[j % 4], lw=0.85,
                            label=ch.canonical_name or ch.name, alpha=0.95)
                _markers(ax)
                ax.legend(loc='upper right', **leg_kw)

            elif ptype == 'd':
                _style(ax, 'Status Digital')
                max_d = min(len(chs), 8)
                for j, ch in enumerate(chs[:max_d]):
                    offset = (max_d - 1 - j) * 1.3
                    ax.step(t_ms, np.array(ch.samples, dtype=float) * 0.9 + offset,
                            where='post', color='#3b82f6', lw=1.0, alpha=0.9)
                    ax.text(t_ms[0], offset + 0.45, (ch.name or '')[:28],
                            color='#334155', fontsize=6.5, va='center', fontweight='600')
                ax.axvline(t_in,  color='#d97706', lw=1.2, ls='--', alpha=0.9, zorder=5)
                ax.axvline(t_out, color='#dc2626', lw=1.2, ls='--', alpha=0.9, zorder=5)
                ax.set_yticks([])

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel('Waktu (ms dari trigger)', color=MUTED, fontsize=8.5)

        plt.tight_layout(pad=0.5)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return _b64.b64encode(buf.read()).decode('ascii')

    except Exception as e:
        app.logger.warning("Waveform plot failed: %s", e)
        return ""


def _downsample_minmax(t_arr, y_arr, target_points: int = 4000):
    """
    Downsample using min/max per bucket to preserve peaks.
    Returns (t, y) numpy arrays.
    """
    import numpy as np

    n = len(t_arr)
    if n == 0:
        return np.array([]), np.array([])
    if n <= target_points:
        return t_arr, y_arr

    buckets = max(1, target_points // 2)
    bucket_size = int(np.ceil(n / buckets))
    xs, ys = [], []

    for start in range(0, n, bucket_size):
        end = min(n, start + bucket_size)
        seg_y = y_arr[start:end]
        if seg_y.size == 0:
            continue
        seg_t = t_arr[start:end]
        i_min = int(np.argmin(seg_y))
        i_max = int(np.argmax(seg_y))

        if seg_t[i_min] <= seg_t[i_max]:
            xs.extend([seg_t[i_min], seg_t[i_max]])
            ys.extend([seg_y[i_min], seg_y[i_max]])
        else:
            xs.extend([seg_t[i_max], seg_t[i_min]])
            ys.extend([seg_y[i_max], seg_y[i_min]])

    return np.array(xs), np.array(ys)


def _digital_change_points(t_arr, y_arr, target_points: int = 4000):
    """
    Reduce digital channels to change points so step plots stay light.
    """
    import numpy as np

    n = len(t_arr)
    if n == 0:
        return np.array([]), np.array([])

    changes = np.where(np.diff(y_arr) != 0)[0] + 1
    idx = np.concatenate(([0], changes, [n - 1]))
    idx = np.unique(idx)
    if len(idx) > target_points:
        step = max(1, int(np.ceil(len(idx) / target_points)))
        idx = idx[::step]

    return t_arr[idx], y_arr[idx]


def _waveform_raw_text(ch) -> str:
    return " ".join(
        _s(part, "")
        for part in (getattr(ch, "name", ""), getattr(ch, "id", ""), getattr(ch, "canonical_name", ""))
        if part
    ).upper()


def _infer_waveform_winding(ch) -> str | None:
    """Infer transformer winding tags from raw channel identifiers."""
    raw_text = _waveform_raw_text(ch)
    if re.search(r"\b(?:HVS|HV SIDE|HIGH VOLTAGE SIDE)\b", raw_text):
        return "HV/W1"
    if re.search(r"\b(?:LVS|LV SIDE|LOW VOLTAGE SIDE)\b", raw_text):
        return "LV/W2"
    if re.search(r"\b(?:TVS|TERTIARY SIDE)\b", raw_text):
        return "TV/W3"
    if re.search(r"\[[^\]]*-\s*1\]", raw_text):
        return "HV/W1"
    if re.search(r"\[[^\]]*-\s*2\]", raw_text):
        return "LV/W2"
    if re.search(r"\[[^\]]*-\s*3\]", raw_text):
        return "TV/W3"
    text = re.sub(r"[_:/\-\[\]\(\)]+", " ", raw_text)
    if re.search(r"\b(?:IA|IB|IC|IN|VA|VB|VC|VN|3I0|I0)\s*1\b", text):
        return "HV/W1"
    if re.search(r"\b(?:IA|IB|IC|IN|VA|VB|VC|VN|3I0|I0)\s*2\b", text):
        return "LV/W2"
    if re.search(r"\b(?:IA|IB|IC|IN|VA|VB|VC|VN|3I0|I0)\s*3\b", text):
        return "TV/W3"
    if re.search(r"\bS1\b", text):
        return "HV/W1"
    if re.search(r"\bS2\b", text):
        return "LV/W2"
    if re.search(r"\bS3\b", text):
        return "TV/W3"
    if re.search(r"\b(HV|W1|IW1|IW1[A-Z0-9]*|W1[A-Z0-9]*|PRIMARY|PRI|WIND1|WINDING1)\b", text):
        return "HV/W1"
    if re.search(r"\b(LV|W2|IW2|IW2[A-Z0-9]*|W2[A-Z0-9]*|SECONDARY|SEC|WIND2|WINDING2)\b", text):
        return "LV/W2"
    if re.search(r"\b(TV|W3|IW3|IW3[A-Z0-9]*|W3[A-Z0-9]*|TERTIARY|TERT|WIND3|WINDING3)\b", text):
        return "TV/W3"
    return None


def _infer_waveform_family(ch) -> str | None:
    """Infer special current families such as differential and bias."""
    text = re.sub(r"[_:/\-\[\]\(\)]+", " ", _waveform_raw_text(ch))
    if re.search(r"\b(RSTR|RESTRAINT|IRSTR|BIAS)\b", text):
        return "RESTRAINT"
    if re.search(r"\b(IREST|IRESTR|I RESTR|I RESTRAINT)\b", text):
        return "RESTRAINT"
    if re.search(r"\b(IDIFF|IDIF|I DIFF|I DIFF OPERATE|DIFF|87T)\b", text):
        return "DIFF"
    return None


def _infer_waveform_phase(ch) -> str | None:
    """Infer phase tags so traces can be rendered as IA, IB, and IC consistently."""
    sources = [
        _s(getattr(ch, "canonical_name", ""), "").upper(),
        _s(getattr(ch, "name", ""), "").upper(),
        _s(getattr(ch, "id", ""), "").upper(),
    ]
    patterns = (
        r"\b[IV]W[123]([ABCN])\b",
        r"\b(?:I|V)([ABCN])\b",
        r"\[\s*(?:I|V)?([ABCN])\s*-\s*\d+\s*\]",
        r"\b(?:DIFF|IDIFF|BIAS|RESTRAINT|RSTR)[ _-]*([ABCN])\b",
    )
    for src in sources:
        if not src:
            continue
        for pattern in patterns:
            match = re.search(pattern, src)
            if match:
                return match.group(1)
        fallback = re.search(r"([ABCN])$", src)
        if fallback:
            return fallback.group(1)
    return None


def _infer_waveform_side(ch) -> str | None:
    """Infer a user-facing side/family tag from raw channel identifiers."""
    family = _infer_waveform_family(ch)
    if family:
        return family
    return _infer_waveform_winding(ch)


def _waveform_side_short(side: str | None) -> str | None:
    return {
        "HV/W1": "HV",
        "LV/W2": "LV",
        "TV/W3": "TV",
    }.get(side)


def _waveform_color_and_dash(group: str, phase: str | None, winding: str | None) -> tuple[str, str]:
    phase_palette = {
        "A": "#ef4444",
        "B": "#2563eb",
        "C": "#16a34a",
        "N": "#eab308",
    }
    diff_side_palette = {
        "HV/W1": "#b91c1c",
        "LV/W2": "#1d4ed8",
        "TV/W3": "#15803d",
    }
    bias_side_palette = {
        "HV/W1": "#f97316",
        "LV/W2": "#06b6d4",
        "TV/W3": "#8b5cf6",
    }
    if group == "current_diff":
        return diff_side_palette.get(winding, phase_palette.get(phase, "#dc2626")), "solid"
    if group == "current_restraint":
        return bias_side_palette.get(winding, phase_palette.get(phase, "#0ea5e9")), "dash"
    return phase_palette.get(phase, "#94a3b8"), "solid"


def _infer_waveform_measurement(ch) -> str:
    measurement = _s(getattr(ch, "measurement", ""), "").lower()
    if measurement in {"current", "voltage"}:
        return measurement
    raw_text = _waveform_raw_text(ch)
    # Normalize separators so word-boundary regexes work on names like "87T.Ida_Hm2_Pcnt"
    norm_text = re.sub(r"[_\-\.:/\[\]\(\)]+", " ", raw_text)
    if re.search(r"\b(?:FREQ|FREQUENCY)\b", norm_text):
        return "other"
    if re.search(r"\b(?:IDIFF|IDIF|I[\s_\-]?DIFF|I[\s_\-]?RESTR(?:AINT)?|IREST|IRESTR|RSTR|RESTRAINT|BIAS)\b", norm_text):
        return "current"
    if re.search(r"\b87T\b", norm_text) and re.search(r"\bID[ABCN]\b", norm_text):
        return "current"
    if re.search(r"\b(?:I[ABCN]|IN|I0|3I0|IDIFF|IDIF|DIFF|BIAS|RSTR|REST)\b", norm_text):
        return "current"
    # Zero-sequence / residual / REF protection currents (e.g. HVS.64REF.3i0d, 3i0Ext)
    if re.search(r"\b(?:3I0|3IO|REF|64REF)\b", norm_text):
        return "current"
    if re.search(r"\b(?:V[ABCN]|VN|V0|VOLT|VX)\b", norm_text):
        return "voltage"
    return measurement or "other"


def _classify_waveform_group(ch) -> str:
    measurement = _infer_waveform_measurement(ch)
    family = _infer_waveform_family(ch)
    winding = _infer_waveform_winding(ch)
    if measurement == "current":
        if family == "DIFF":
            return "current_diff"
        if family == "RESTRAINT":
            return "current_restraint"
        if winding == "HV/W1":
            return "current_hv"
        if winding == "LV/W2":
            return "current_lv"
        if winding == "TV/W3":
            return "current_tv"
        return "current_other"
    if measurement == "voltage":
        if winding == "HV/W1":
            return "voltage_hv"
        if winding == "LV/W2":
            return "voltage_lv"
        if winding == "TV/W3":
            return "voltage_tv"
        return "voltage_other"
    return measurement or "other"


def _waveform_priority_key(ch) -> tuple:
    phase_order = {"A": 0, "B": 1, "C": 2, "N": 3}
    winding_order = {"HV/W1": 0, "LV/W2": 1, "TV/W3": 2}
    group_order = {
        "voltage_hv": 0,
        "voltage_lv": 1,
        "voltage_tv": 2,
        "voltage_other": 3,
        "current_hv": 4,
        "current_lv": 5,
        "current_tv": 6,
        "current_diff": 7,
        "current_restraint": 8,
        "current_other": 9,
    }
    canonical = _s(getattr(ch, "canonical_name", ""), "").upper()
    name = _s(getattr(ch, "name", ""), "").upper()
    group = _classify_waveform_group(ch)
    phase = _infer_waveform_phase(ch)
    winding = _infer_waveform_winding(ch)
    return (
        group_order.get(group, 99),
        winding_order.get(winding, 99),
        phase_order.get(phase, 99),
        canonical or name,
        name,
    )


def _build_waveform_display_name(ch, duplicate_canonicals: Counter) -> str:
    """Keep transformer current labels explicit as IA HV, IA LV, Idiff HV, and Bias LV."""
    canonical = _s(getattr(ch, "canonical_name", ""), "").strip()
    raw_name = _s(getattr(ch, "name", ""), "").strip()
    base = canonical or raw_name or _s(getattr(ch, "id", ""), "").strip() or "CHANNEL"
    measurement = _infer_waveform_measurement(ch)
    group = _classify_waveform_group(ch)
    winding = _infer_waveform_winding(ch)
    winding_short = _waveform_side_short(winding)
    phase = _infer_waveform_phase(ch)

    if measurement == "current":
        if group in {"current_hv", "current_lv", "current_tv"} and phase:
            return f"I{phase} {winding_short or ''}".strip()
        if group == "current_diff":
            if phase:
                return f"Idiff {phase}"
            if winding_short:
                return f"Idiff {winding_short}"
            return "Idiff"
        if group == "current_restraint":
            if phase:
                return f"Bias {phase}"
            if winding_short:
                return f"Bias {winding_short}"
            return "Bias"

    dup_key = (measurement, canonical.upper()) if canonical else None
    is_duplicate = bool(dup_key and duplicate_canonicals.get(dup_key, 0) > 1)
    is_generic = bool(re.fullmatch(r"(?:V[ABCN]|I[ABCN]|IN|I0|3I0|VN|V0)", base.upper()))

    if winding_short and (is_duplicate or is_generic):
        return f"{base} {winding_short}"
    if is_duplicate and raw_name and raw_name.upper() != base.upper():
        return f"{base} [{raw_name}]"
    return base


def _waveform_should_skip_channel(ch) -> bool:
    raw_text = _waveform_raw_text(ch)
    if re.search(r"\b(?:FREQ|FREQUENCY|SPARE)\b", raw_text):
        return True
    return False


def _extract_waveform_line_tag(name: str) -> str | None:
    text = re.sub(r"[_:/\\\-\[\]\(\)]+", " ", _s(name, "")).upper()
    if not text:
        return None
    match = re.search(r"(?:LINE|BAY|JEPARA|SIRKIT|CCT|CIRCUIT)\s*#?\s*([0-9A-Z]+)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-Z]{3,}\s+([0-9])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9])\b", text)
    if match and not re.search(r"\b(?:HV|LV|TV|W[123]|S[123])\b", text):
        return match.group(1)
    return None


def _select_waveform_line_tag(analog_channels, status_channels, raw_t_ms, inception_ms) -> str | None:
    import numpy as np

    structured_groups = {
        "current_hv", "current_lv", "current_tv",
        "current_diff", "current_restraint",
        "voltage_hv", "voltage_lv", "voltage_tv",
    }
    if any(_classify_waveform_group(ch) in structured_groups for ch in analog_channels):
        return None

    phase_currents = [
        ch for ch in analog_channels
        if _infer_waveform_measurement(ch) == "current"
        and _s(getattr(ch, "canonical_name", ""), "").upper() in {"IA", "IB", "IC"}
    ]
    distinct_tags = sorted({
        tag for ch in phase_currents
        if (tag := _extract_waveform_line_tag(getattr(ch, "name", "")))
    })
    if len(distinct_tags) < 2:
        return None

    inception_idx = None
    if len(raw_t_ms):
        try:
            inception_idx = int(abs(raw_t_ms - float(inception_ms or 0.0)).argmin())
        except Exception:
            inception_idx = None

    if inception_idx is not None:
        status_scores = {}
        status_kw = ("TRIP", "OPRT", "OPERATE", "PICKUP", "DIST", "DIS.", "START")
        for ch in status_channels:
            name = _s(getattr(ch, "name", ""), "").upper()
            if not any(keyword in name for keyword in status_kw):
                continue
            tag = _extract_waveform_line_tag(name)
            if tag not in distinct_tags:
                continue
            samples = getattr(ch, "samples", [])
            if len(samples) < 2:
                continue
            t0 = max(0, inception_idx - 100)
            t1 = min(len(samples), inception_idx + 1000)
            if t1 <= t0 + 1:
                continue
            rises = (np.diff(np.asarray(samples[t0:t1], dtype=int)) > 0).sum()
            if rises:
                status_scores[tag] = status_scores.get(tag, 0.0) + float(rises)
        if status_scores:
            ranked = sorted(status_scores.items(), key=lambda item: item[1], reverse=True)
            best_tag, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if best_score > 0 and (len(ranked) == 1 or best_score >= second_score * 1.5):
                return best_tag

    activity_scores = {}
    for ch in phase_currents:
        tag = _extract_waveform_line_tag(getattr(ch, "name", ""))
        if tag not in distinct_tags:
            continue
        samples = np.asarray(getattr(ch, "samples", []), dtype=float)
        if not samples.size:
            continue
        if inception_idx is None:
            segment = samples
        else:
            ws = max(0, inception_idx - 50)
            we = min(samples.size, inception_idx + 400)
            segment = samples[ws:we] if we > ws else samples
        if segment.size:
            activity_scores[tag] = activity_scores.get(tag, 0.0) + float(np.max(np.abs(segment)))
    if activity_scores:
        ranked = sorted(activity_scores.items(), key=lambda item: item[1], reverse=True)
        best_tag, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score > 0 and (len(ranked) == 1 or best_score >= second_score * 1.25):
            return best_tag
    return None


def _build_waveform_payload(cfg_path: str, inception_ms: float, duration_ms: float) -> dict:
    """
    Prepare a lightweight, JSON-safe payload for interactive waveform viewing.
    Downsamples to keep the UI responsive while preserving peaks.
    """
    try:
        import numpy as np
    except Exception:
        return {}

    record = parse_comtrade(cfg_path)
    if record is None or len(record.time) == 0:
        return {}

    raw_t_ms = np.array(record.time, dtype=float) * 1000.0
    total_samples = len(raw_t_ms)
    # Shift time so fault inception sits at t=0 (more intuitive for analysis)
    t0 = float(inception_ms or 0.0)
    if t0 == 0.0 and total_samples:
        t0 = float(raw_t_ms[0])
    t_ms = raw_t_ms - t0

    max_analog = 24
    max_digital = 16

    analog_all = [
        ch for ch in record.analog_channels
        if len(ch.samples) == total_samples and not _waveform_should_skip_channel(ch)
    ]
    digital_all = [ch for ch in record.status_channels if len(ch.samples) == total_samples]
    digital_active_unfiltered = [ch for ch in digital_all if len(np.unique(ch.samples)) > 1]

    selected_line_tag = None
    candidate_line_tag = _select_waveform_line_tag(
        analog_all,
        getattr(record, "status_channels", []),
        raw_t_ms,
        float(inception_ms or 0.0),
    )
    should_focus_single_tag = (
        candidate_line_tag is not None
        and (len(analog_all) > max_analog or len(digital_active_unfiltered) > max_digital)
    )
    if should_focus_single_tag:
        selected_line_tag = candidate_line_tag
        analog_all = [
            ch for ch in analog_all
            if (tag := _extract_waveform_line_tag(getattr(ch, "name", ""))) in {None, selected_line_tag}
        ]
        digital_all = [
            ch for ch in digital_all
            if (tag := _extract_waveform_line_tag(getattr(ch, "name", ""))) in {None, selected_line_tag}
        ]
    duplicate_canonicals = Counter(
        (_s(ch.measurement, "").lower(), _s(ch.canonical_name, "").upper())
        for ch in analog_all
        if _s(ch.canonical_name, "").strip()
    )
    analog_sorted = sorted(analog_all, key=_waveform_priority_key)
    analog_chs = analog_sorted[:max_analog]
    analog_truncated = len(analog_sorted) > max_analog

    digital_active = [ch for ch in digital_all if len(np.unique(ch.samples)) > 1]
    digital_chs = digital_active[:max_digital]
    digital_truncated = len(digital_active) > max_digital

    channels = []
    for ch in analog_chs:
        display_name = _build_waveform_display_name(ch, duplicate_canonicals)
        measurement = _infer_waveform_measurement(ch)
        group = _classify_waveform_group(ch)
        side = _infer_waveform_side(ch)
        winding = _infer_waveform_winding(ch)
        phase = _infer_waveform_phase(ch)
        color, line_dash = _waveform_color_and_dash(group, phase, winding)

        y = np.array(ch.samples, dtype=float)
        def _safe_stat(val):
            return float(val) if np.isfinite(val) else 0.0
        def _nan_safe_list(arr):
            """Convert numpy array to list, replacing NaN/Inf with None (JSON null)."""
            return [None if not np.isfinite(v) else float(v) for v in arr]
        x_ds, y_ds = _downsample_minmax(t_ms, y, target_points=4000)

        channels.append({
            "id": ch.id,
            "name": ch.name,
            "canonical": ch.canonical_name,
            "display_name": display_name,
            "measurement": measurement,
            "group": group,
            "side": side,
            "winding": winding,
            "phase": phase,
            "unit": ch.unit,
            "color": color,
            "line_dash": line_dash,
            "x": _nan_safe_list(x_ds),
            "y": _nan_safe_list(y_ds),
            "min": _safe_stat(np.min(y)) if y.size else 0.0,
            "max": _safe_stat(np.max(y)) if y.size else 0.0,
            "rms": _safe_stat(np.sqrt(np.mean(y ** 2))) if y.size else 0.0,
        })

    digital = []
    for ch in digital_chs:
        y = np.array(ch.samples, dtype=int)
        x_ds, y_ds = _digital_change_points(t_ms, y, target_points=4000)
        digital.append({
            "id": ch.id,
            "name": ch.name,
            "x": _nan_safe_list(x_ds),
            "y": [int(v) for v in y_ds],
            "events": int(len(np.where(np.diff(y) != 0)[0])) if y.size else 0,
        })

    # Estimate sampling rate from time axis
    if total_samples > 1:
        dt = np.mean(np.diff(t_ms)) / 1000.0
        sampling_hz = float(1.0 / dt) if (dt > 0 and np.isfinite(dt)) else 0.0
    else:
        sampling_hz = 0.0

    dur_ms = float(t_ms[-1] - t_ms[0]) if total_samples else 0.0
    if not np.isfinite(dur_ms):
        dur_ms = 0.0

    return {
        "meta": {
            "total_samples": total_samples,
            "sampling_hz": sampling_hz,
            "duration_ms": dur_ms,
            "analog_total": len(analog_all),
            "digital_total": len(digital_all),
            "analog_truncated": analog_truncated,
            "digital_truncated": digital_truncated,
            "selected_line_tag": selected_line_tag,
        },
        "markers": {
            "inception_ms": 0.0,
            "clearing_ms": float(duration_ms or 0.0),
        },
        "channels": channels,
        "digital": digital,
    }

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/line/upload")
def line_upload():
    return render_template("line_upload.html", fault_categories=FAULT_CATEGORIES)


@app.route("/upload", methods=["POST"])
def upload_files():
    cfg_file = request.files.get("cfg_file")
    dat_file = request.files.get("dat_file")

    if not cfg_file or not dat_file:
        return jsonify({"error": "Both .cfg and .dat files are required"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path, _ = _save_uploaded_pair(cfg_file, dat_file, ts)

    try:
        result = classify_file(str(cfg_path))
    except ValueError as e:
        return _render_not_supported(cfg_file.filename, str(e), cfg_path=str(cfg_path))
    except Exception as e:
        return _render_not_supported(cfg_file.filename, f"Pipeline error: {e}", cfg_path=str(cfg_path))

    analysis_payload = _build_analysis_payload(result, cfg_file.filename, ts, cfg_path)
    analysis_payload["intake_domain"] = "LINE"
    _clear_analysis_store()
    _save_analysis_to_store(analysis_payload)

    return redirect(url_for("results"))


@app.route("/transformer/upload", methods=["GET", "POST"])
def transformer_upload():
    if request.method == "GET":
        return render_template(
            "transformer_upload.html",
            relay_roles=TRANSFORMER_RELAY_ROLES,
            analysis_goals=TRANSFORMER_ANALYSIS_GOALS,
            form_error="",
        )

    cfg_file = request.files.get("cfg_file")
    dat_file = request.files.get("dat_file")
    intake_mode = (request.form.get("intake_mode") or "QUICK").upper()
    relay_role = (request.form.get("relay_role") or "AUTO").upper()
    analysis_goal = (request.form.get("analysis_goal") or "EVENT_MEANING").upper()

    if not cfg_file or not dat_file:
        return render_template(
            "transformer_upload.html",
            relay_roles=TRANSFORMER_RELAY_ROLES,
            analysis_goals=TRANSFORMER_ANALYSIS_GOALS,
            form_error="File primer .cfg dan .dat wajib diisi.",
        ), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path, _ = _save_uploaded_pair(cfg_file, dat_file, ts, prefix="primary")

    supporting_relays = []
    for idx in range(1, 4):
        support_cfg = request.files.get(f"support_cfg_{idx}")
        support_dat = request.files.get(f"support_dat_{idx}")
        support_role = (request.form.get(f"support_role_{idx}") or "").upper()
        has_cfg = bool(support_cfg and support_cfg.filename)
        has_dat = bool(support_dat and support_dat.filename)
        if not (has_cfg or has_dat or support_role):
            continue
        if has_cfg != has_dat:
            return render_template(
                "transformer_upload.html",
                relay_roles=TRANSFORMER_RELAY_ROLES,
                analysis_goals=TRANSFORMER_ANALYSIS_GOALS,
                form_error=f"Supporting relay #{idx} harus berisi pasangan .cfg dan .dat lengkap.",
            ), 400
        support_cfg_path, _ = _save_uploaded_pair(support_cfg, support_dat, ts, prefix=f"support{idx}")
        supporting_relays.append({
            "role": support_role or "OTHER",
            "role_label": _transformer_role_label(support_role or "OTHER"),
            "cfg_name": support_cfg.filename,
            "cfg_path": str(support_cfg_path),
        })

    try:
        result = classify_file(str(cfg_path))
    except ValueError as e:
        return _render_not_supported(cfg_file.filename, str(e), cfg_path=str(cfg_path))
    except Exception as e:
        return _render_not_supported(cfg_file.filename, f"Pipeline error: {e}", cfg_path=str(cfg_path))

    analysis_payload = _build_analysis_payload(result, cfg_file.filename, ts, cfg_path)
    _apply_transformer_intake_context(
        analysis_payload,
        relay_role=relay_role,
        analysis_goal=analysis_goal,
        intake_mode=intake_mode,
        supporting_relays=supporting_relays,
    )

    # ── COORDINATED mode: analyse supporting relays and fuse evidence ────────
    if intake_mode == "COORDINATED" and supporting_relays:
        support_evidence = []
        for sr in supporting_relays:
            ev = _analyse_supporting_relay(sr["cfg_path"], sr["role"])
            # carry the display name through
            ev["cfg_name"] = sr["cfg_name"]
            support_evidence.append(ev)

        fused = _fuse_multi_relay_evidence(analysis_payload, support_evidence)

        # Overwrite the primary payload's layered fields with fused values
        analysis_payload["fault_origin"]                  = fused["fault_origin"]
        analysis_payload["fault_origin_reason"]           = fused["fault_origin_reason"]
        analysis_payload["protection_assessment"]         = fused["protection_assessment"]
        analysis_payload["protection_assessment_reason"]  = fused["protection_assessment_reason"]
        analysis_payload["fusion_summary"]                = fused["fusion_summary"]
        analysis_payload["fusion_conflicts"]              = fused["fusion_conflicts"]
        analysis_payload["is_coordinated"]                = True
    else:
        analysis_payload["fusion_summary"]   = []
        analysis_payload["fusion_conflicts"] = []
        analysis_payload["is_coordinated"]   = False

    _clear_analysis_store()
    _save_analysis_to_store(analysis_payload)
    return redirect(url_for("results"))


@app.route("/results")
def results():
    analysis = _load_analysis_from_store()
    if not analysis:
        return redirect(url_for("index"))
    # Phase 2: route transformer events to dedicated template
    if analysis.get("event_type") in {"TRANSFORMER", "TRANSFORMER_CONTEXT"}:
        return render_template(
            "transformer_results.html",
            analysis=analysis,
            fault_categories=XFMR_FAULT_CATEGORIES,
            predicted_event_class_label=_xfmr_event_class_label(analysis.get("event_class")),
        )
    return render_template("results.html",
                           analysis=analysis,
                           fault_categories=FAULT_CATEGORIES)


@app.route("/browse")
def browse():
    if not RAW_DATA.exists():
        return render_template("browse.html", events=[], upts=[], status_groups=[],
                               status_counts={},
                               offline_mode=True, db_enabled=_db_enabled())
    events = _scan_recordings()
    status_order = {"TRANSIENT": 0, "TRANSFORMER": 1, "UNKNOWN": 2}
    events.sort(key=lambda e: (
        status_order.get(e.get("status_data", "UNKNOWN"), 99),
        e.get("upt", ""),
        e.get("year", ""),
        e.get("month", ""),
        e.get("gi", ""),
        e.get("filename", ""),
    ))
    upts   = sorted(set(e["upt"]   for e in events))
    status_groups = sorted(set(e["status_data"] for e in events))
    status_counts = dict(Counter(e["status_data"] for e in events))
    return render_template(
        "browse.html",
        events=events,
        upts=upts,
        status_groups=status_groups,
        status_counts=status_counts,
        offline_mode=False,
        db_enabled=_db_enabled()
    )


@app.route("/analyze-from-browse", methods=["POST"])
def analyze_from_browse():
    cfg_path = request.form.get("cfg_path", "")
    if not cfg_path:
        return jsonify({"error": "File tidak ditemukan"}), 404

    try:
        result = classify_file(cfg_path)
    except ValueError as e:
        return _render_not_supported(Path(cfg_path).name, str(e), cfg_path=cfg_path)
    except Exception as e:
        return _render_not_supported(Path(cfg_path).name, f"Pipeline error: {e}", cfg_path=cfg_path)

    feats = result.features
    cfg_ratios = _extract_cfg_ratios(cfg_path)
    ratio_defaults = _resolve_ratio_defaults(
        cfg_ratios,
        feats.get("voltage_kv"),
        feats.get("peak_fault_current_a"),
        event_type=getattr(result, "event_type", "LINE"),
    )
    ct_ratio_src = ratio_defaults["ct_ratio_source"]
    vt_ratio_src = ratio_defaults["vt_ratio_source"]
    ct_p_display = ratio_defaults["ct_primary"]
    ct_s_display = ratio_defaults["ct_secondary"]
    vt_p_display = ratio_defaults["vt_primary"]
    vt_s_display = ratio_defaults["vt_secondary"]
    ct_p_base = ratio_defaults["ct_base_primary"]
    ct_s_base = ratio_defaults["ct_base_secondary"]
    vt_p_base = ratio_defaults["vt_base_primary"]
    vt_s_base = ratio_defaults["vt_base_secondary"]
    analysis_payload = {
        "original_filename": Path(cfg_path).name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "label":          result.label,
        "confidence":     _f(result.confidence),
        "tier":           int(result.tier),
        "rule_name":      result.rule_name,
        "evidence":       result.evidence,
        "recommendation": result.recommendation,
        "description":    result.description or "",
        "cause_pcts":     result.cause_pcts or [],
        "soe":            result.soe or [],
        "di_dt_max":      _f(feats.get("di_dt_max")),
        "thd_percent":    _f(feats.get("thd_percent")),
        "inception_angle_degrees": _f(feats.get("inception_angle_degrees"), -1),
        "station_name":         _s(feats.get("station_name")),
        "relay_model":          _s(feats.get("relay_model")),
        "zone_operated":        _s(feats.get("zone_operated")),
        "trip_type":            _s(feats.get("trip_type")),
        "faulted_phases":       _s(feats.get("faulted_phases")),
        "fault_type":           _s(feats.get("fault_type")),
        "fault_duration_ms":    _f(feats.get("fault_duration_ms")),
        "fault_inception_ms":   _f(feats.get("fault_inception_ms")),
        "record_duration_ms":   _f(feats.get("record_duration_ms")),
        "fault_count":          int(feats.get("fault_count") or 0),
        "peak_fault_current_a": _f(feats.get("peak_fault_current_a")),
        "peak_fault_phase":     _s(feats.get("peak_fault_phase")),
        "i0_i1_ratio":          _f(feats.get("i0_i1_ratio")),
        "i0_magnitude_a":       _f(feats.get("i0_magnitude_a")),
        "i1_magnitude_a":       _f(feats.get("i1_magnitude_a")),
        "i2_magnitude_a":       _f(feats.get("i2_magnitude_a")),
        "voltage_sag_depth_pu": _f(feats.get("voltage_sag_depth_pu")),
        "voltage_sag_phase":    _s(feats.get("voltage_sag_phase")),
        "v_prefault_v":         _f(feats.get("v_prefault_v")),
        "v_fault_v":            _f(feats.get("v_fault_v")),
        "z_magnitude_ohms":     _f(feats.get("z_magnitude_ohms")),
        "z_angle_degrees":      _f(feats.get("z_angle_degrees")),
        "r_x_ratio":            _f(feats.get("r_x_ratio")),
        "reclose_successful":   _s(feats.get("reclose_successful")),
        "reclose_time_ms":      _f(feats.get("reclose_time_ms")),
        "voltage_kv":           str(int(_nearest_supported_voltage_kv(vt_p_base / 1000.0))) if (vt_ratio_src == "cfg" and vt_p_base > 1000.0) else (_s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui"),
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
        "ct_primary": ct_p_display,
        "ct_secondary": ct_s_display,
        "vt_primary": vt_p_display,
        "vt_secondary": vt_s_display,
        "ct_primary_default": ct_p_display,
        "ct_secondary_default": ct_s_display,
        "vt_primary_default": vt_p_display,
        "vt_secondary_default": vt_s_display,
        "ratio_base_ct_primary": ct_p_base,
        "ratio_base_ct_secondary": ct_s_base,
        "ratio_base_vt_primary": vt_p_base,
        "ratio_base_vt_secondary": vt_s_base,
        "ratio_override_active": False,
        "cfg_ct_primary": _f(cfg_ratios.get("cfg_ct_primary"), 1.0),
        "cfg_ct_secondary": _f(cfg_ratios.get("cfg_ct_secondary"), 1.0),
        "cfg_vt_primary": _f(cfg_ratios.get("cfg_vt_primary"), 1.0),
        "cfg_vt_secondary": _f(cfg_ratios.get("cfg_vt_secondary"), 1.0),
        "cfg_ct_known": bool(cfg_ratios.get("cfg_ct_known")),
        "cfg_vt_known": bool(cfg_ratios.get("cfg_vt_known")),
        "parser_ct_multiplier": _f(cfg_ratios.get("parser_ct_multiplier"), 0.0),
        "parser_vt_multiplier": _f(cfg_ratios.get("parser_vt_multiplier"), 0.0),
        "ct_ratio_source": ct_ratio_src,
        "vt_ratio_source": vt_ratio_src,
        "ct_ratio_known":  bool(cfg_ratios.get("cfg_ct_known")),
        "vt_ratio_known":  bool(cfg_ratios.get("cfg_vt_known")),
        "ratio_base_values": {
            "peak_fault_current_a": _f(feats.get("peak_fault_current_a")),
            "i0_magnitude_a": _f(feats.get("i0_magnitude_a")),
            "i1_magnitude_a": _f(feats.get("i1_magnitude_a")),
            "i2_magnitude_a": _f(feats.get("i2_magnitude_a")),
            "v_prefault_v": _f(feats.get("v_prefault_v")),
            "v_fault_v": _f(feats.get("v_fault_v")),
            "z_magnitude_ohms": _f(feats.get("z_magnitude_ohms")),
            "orig_ct_p": ct_p_base,
            "orig_ct_s": ct_s_base,
            "orig_vt_p": vt_p_base,
            "orig_vt_s": vt_s_base,
        },
    }
    # Phase 2: add transformer-specific fields if applicable
    _inject_transformer_fields(result, analysis_payload)

    analysis_payload["waveform_data"] = _build_waveform_payload(
        str(cfg_path),
        analysis_payload["fault_inception_ms"],
        analysis_payload["fault_duration_ms"],
    )
    analysis_payload["waveform_b64"] = _generate_waveform_plot(
        str(cfg_path),
        analysis_payload["fault_inception_ms"],
        analysis_payload["fault_duration_ms"],
    )
    _clear_analysis_store()
    _save_analysis_to_store(analysis_payload)
    return redirect(url_for("results"))


@app.route("/confirm", methods=["POST"])
def confirm_prediction():
    analysis = _load_analysis_from_store()
    if not analysis:
        return jsonify({"error": "No analysis in session"}), 400
    _ensure_history_schema()

    confirmed = request.form.get("fault_cause", "LAIN-LAIN")
    notes     = request.form.get("notes", "")
    predicted_cause_top = _top_predicted_cause_from_analysis(analysis)
    correct = _normalize_label(confirmed) == _normalize_label(predicted_cause_top)

    row = {
        "timestamp":          analysis["timestamp"],
        "filename":           analysis["original_filename"],
        "station":            analysis["station_name"],
        "predicted_label":    analysis["label"],
        "predicted_cause_top": predicted_cause_top,
        "predicted_conf":     f"{analysis['confidence']:.2f}",
        "tier":               analysis["tier"],
        "rule_name":          analysis["rule_name"],
        "confirmed_label":    confirmed,
        "correct":            correct,
        "notes":              notes,
        "zone":               analysis["zone_operated"],
        "phases":             analysis["faulted_phases"],
        "duration_ms":        analysis["fault_duration_ms"],
        "fault_count":        analysis["fault_count"],
        "peak_current_a":     analysis["peak_fault_current_a"],
        "i0_i1_ratio":        analysis["i0_i1_ratio"],
        "reclose_ok":         analysis["reclose_successful"],
    }

    # Persist to Postgres first (Railway), fallback to CSV if DB unavailable.
    saved_to_db = _db_insert_feedback(
        row,
        source_ip=request.headers.get("X-Forwarded-For", request.remote_addr or ""),
        user_agent=request.headers.get("User-Agent", ""),
    )
    if not saved_to_db:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    _clear_analysis_store()
    return redirect(url_for("success"))


@app.route("/success")
def success():
    return render_template("success.html")


@app.route("/trends")
def trends():
    rows = _load_history_rows(newest_first=False)

    if not rows:
        return render_template("trends.html", rows=[], stats={}, cause_dist=[],
                               station_stats=[], recurring=[], accuracy_by_cause=[], monthly=[])

    from collections import Counter, defaultdict

    total   = len(rows)
    comparable = [r for r in rows if _row_correct_flag(r) is not None]
    correct = sum(1 for r in comparable if _row_correct_flag(r) is True)
    stations = len(set(r.get("station", "-") for r in rows if r.get("station")))
    permanent = sum(1 for r in rows if "PERMANEN" in r.get("confirmed_label", "").upper()
                    or "KONDUKTOR" in r.get("confirmed_label", "").upper())

    stats = {
        "total":     total,
        "correct":   correct,
        "accuracy":  round(correct / len(comparable) * 100, 1) if comparable else 0,
        "stations":  stations,
        "permanent": permanent,
    }

    # Cause distribution (by confirmed label)
    cause_counter = Counter(r.get("confirmed_label", "LAIN-LAIN") for r in rows)
    cause_dist = cause_counter.most_common()

    # Per-station stats
    by_station = defaultdict(list)
    for r in rows:
        by_station[r.get("station", "-")].append(r)

    station_stats = []
    for st, st_rows in sorted(by_station.items(), key=lambda x: -len(x[1])):
        st_total   = len(st_rows)
        st_comp = [r for r in st_rows if _row_correct_flag(r) is not None]
        st_correct = sum(1 for r in st_comp if _row_correct_flag(r) is True)
        top_cause  = Counter(r.get("confirmed_label", "") for r in st_rows).most_common(1)[0][0]
        station_stats.append({
            "station":   st,
            "count":     st_total,
            "top_cause": top_cause,
            "accuracy":  round(st_correct / len(st_comp) * 100, 0) if st_comp else None,
        })

    # Recurring faults (stations with ≥ 3 events)
    recurring = [s for s in station_stats if s["count"] >= 3]

    # Accuracy per cause
    by_cause = defaultdict(list)
    for r in rows:
        by_cause[r.get("confirmed_label", "LAIN-LAIN")].append(r)

    accuracy_by_cause = []
    for cause, c_rows in sorted(by_cause.items(), key=lambda x: -len(x[1])):
        c_total   = len(c_rows)
        c_comp = [r for r in c_rows if _row_correct_flag(r) is not None]
        c_correct = sum(1 for r in c_comp if _row_correct_flag(r) is True)
        accuracy_by_cause.append({
            "cause":    cause,
            "total":    c_total,
            "correct":  c_correct,
            "accuracy": round(c_correct / len(c_comp) * 100, 0) if c_comp else 0,
        })

    # Monthly breakdown — parse timestamp YYYYMMDD_HHMMSS
    by_month = defaultdict(list)
    for r in rows:
        ts = r.get("timestamp", "")
        month = ts[:6] if len(ts) >= 6 else "??????"
        try:
            month_label = f"{month[4:6]}/{month[:4]}"
        except Exception:
            month_label = month
        by_month[month_label].append(r)

    monthly = []
    for month_lbl in sorted(by_month.keys(), reverse=True)[:12]:
        m_rows  = by_month[month_lbl]
        m_total = len(m_rows)
        m_comp  = [r for r in m_rows if _row_correct_flag(r) is not None]
        m_corr  = sum(1 for r in m_comp if _row_correct_flag(r) is True)
        m_causes = Counter(r.get("confirmed_label", "") for r in m_rows)
        monthly.append({
            "month":    month_lbl,
            "total":    m_total,
            "causes":   dict(m_causes),
            "accuracy": round(m_corr / len(m_comp) * 100, 0) if m_comp else None,
        })

    return render_template("trends.html", rows=rows, stats=stats,
                           cause_dist=cause_dist, station_stats=station_stats,
                           recurring=recurring, accuracy_by_cause=accuracy_by_cause,
                           monthly=monthly)


@app.route("/report")
def report():
    analysis = _load_analysis_from_store()
    if not analysis:
        return redirect(url_for("index"))
    return render_template("report.html", analysis=analysis)


@app.route("/recalculate-with-ratio", methods=["POST"])
def recalculate_with_ratio():
    analysis = _load_analysis_from_store()
    if not analysis:
        return redirect(url_for("index"))

    ct_p = _f(request.form.get("ct_primary"), 1.0)
    ct_s = _f(request.form.get("ct_secondary"), 1.0)
    vt_p = _f(request.form.get("vt_primary"), 1.0)
    vt_s = _f(request.form.get("vt_secondary"), 1.0)

    updated = _recalculate_analysis_with_ratio(analysis, ct_p, ct_s, vt_p, vt_s)
    _clear_analysis_store()
    _save_analysis_to_store(updated)
    return redirect(url_for("results"))


@app.route("/history")
def history():
    rows = _load_history_rows(newest_first=True)

    for r in rows:
        r["_correct_eval"] = _row_correct_flag(r)

    # Simple accuracy stats
    total = len(rows)
    comparable = [r for r in rows if r.get("_correct_eval") is not None]
    correct = sum(1 for r in comparable if r.get("_correct_eval") is True)
    accuracy = round(correct / len(comparable) * 100, 1) if comparable else 0

    return render_template("history.html", rows=rows,
                           total=total, correct=correct, accuracy=accuracy,
                           db_enabled=_db_enabled())


# ── API endpoints ─────────────────────────────────────────────────────────────
# All API routes are prefixed with /api and return JSON.
# No session required — stateless, suitable for integration with other apps.

def _build_analysis_json(result, original_filename: str, ts: str) -> dict:
    """Build a serialisable dict from a ClassificationResult."""
    feats = result.features
    return {
        "meta": {
            "filename":   original_filename,
            "timestamp":  ts,
            "station":    _s(feats.get("station_name")),
            "relay":      _s(feats.get("relay_model")),
            "voltage_kv": _f(feats.get("voltage_kv")) or None,
            "sampling_hz": _f(feats.get("sampling_rate_hz")),
        },
        "classification": {
            "label":       result.label,
            "confidence":  round(_f(result.confidence), 4),
            "tier":        int(result.tier),
            "rule":        result.rule_name,
            "description": result.description or "",
        },
        "cause_likelihoods": result.cause_pcts or [],
        "fault": {
            "type":          _s(feats.get("fault_type")),
            "phases":        _s(feats.get("faulted_phases")),
            "zone":          _s(feats.get("zone_operated")),
            "trip_type":     _s(feats.get("trip_type")),
            "duration_ms":   _f(feats.get("fault_duration_ms")),
            "inception_ms":  _f(feats.get("fault_inception_ms")),
            "record_ms":     _f(feats.get("record_duration_ms")),
            "fault_count":   int(feats.get("fault_count") or 1),
        },
        "electrical": {
            "peak_current_a":      _f(feats.get("peak_fault_current_a")),
            "peak_phase":          _s(feats.get("peak_fault_phase")),
            "i0_magnitude_a":      _f(feats.get("i0_magnitude_a")),
            "i1_magnitude_a":      _f(feats.get("i1_magnitude_a")),
            "i2_magnitude_a":      _f(feats.get("i2_magnitude_a")),
            "i0_i1_ratio":         round(_f(feats.get("i0_i1_ratio")), 4),
            "voltage_sag_pu":      round(_f(feats.get("voltage_sag_depth_pu")), 4),
            "voltage_sag_phase":   _s(feats.get("voltage_sag_phase")),
            "v_prefault_v":        _f(feats.get("v_prefault_v")),
            "v_fault_v":           _f(feats.get("v_fault_v")),
            "z_magnitude_ohm":     _f(feats.get("z_magnitude_ohms")),
            "z_angle_deg":         _f(feats.get("z_angle_degrees")),
            "r_x_ratio":           _f(feats.get("r_x_ratio")),
            "di_dt_max":           _f(feats.get("di_dt_max")),
            "thd_percent":         _f(feats.get("thd_percent")),
        },
        "reclose": {
            "attempted":   bool(feats.get("reclose_attempted")),
            "successful":  feats.get("reclose_successful"),  # True/False/None
            "dead_time_ms": _f(feats.get("reclose_time_ms")),
        },
        "soe": result.soe or [],
        "evidence": result.evidence,
        "recommendation": result.recommendation,
    }


@app.route("/api/status")
def api_status():
    """Health-check endpoint."""
    return jsonify({
        "status": "ok",
        "app": "DFR Fault Classifier — TFA",
        "version": "2.0",
        "db_enabled": _db_enabled(),
        "endpoints": [
            "POST /api/classify  — upload .cfg + .dat, returns JSON analysis",
            "GET  /api/history   — returns list of confirmed analyses",
            "GET  /api/status    — this endpoint",
        ]
    })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    Classify a COMTRADE file pair and return JSON.

    Accepts multipart/form-data with fields:
        cfg_file  — the .cfg file
        dat_file  — the .dat file (must share the same base name)

    Returns JSON with full classification result, electrical parameters,
    symmetrical components, impedance, SOE, and cause likelihoods.

    On error returns:
        { "error": "...", "code": "..." }  with HTTP 4xx/5xx
    """
    cfg_file = request.files.get("cfg_file")
    dat_file = request.files.get("dat_file")

    if not cfg_file or not dat_file:
        return jsonify({"error": "Both cfg_file and dat_file are required", "code": "missing_files"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_saved = UPLOAD_DIR / secure_filename(f"{ts}_{cfg_file.filename}")
    dat_saved = UPLOAD_DIR / secure_filename(f"{ts}_{dat_file.filename}")
    cfg_file.save(cfg_saved)
    dat_file.save(dat_saved)

    try:
        result = classify_file(str(cfg_saved))
    except ValueError as e:
        # Try to still extract SOE for unsupported relay types
        soe = extract_soe_from_file(str(cfg_saved))
        return jsonify({
            "error": str(e),
            "code": "unsupported_or_no_fault",
            "soe": soe,
        }), 422
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {e}", "code": "pipeline_error"}), 500

    payload = _json_safe(_build_analysis_json(result, cfg_file.filename, ts))
    return jsonify(payload), 200


@app.route("/api/history")
def api_history():
    """
    Return the analysis history as JSON.

    Optional query params:
        limit  — max number of rows (default 100, newest first)
        format — "full" returns all CSV fields; default returns summary only
    """
    limit = min(int(request.args.get("limit", 100)), 1000)
    full  = request.args.get("format") == "full"

    rows = _load_history_rows(limit=limit, newest_first=True)

    total   = len(rows)
    comparable = [r for r in rows if _row_correct_flag(r) is not None]
    correct = sum(1 for r in comparable if _row_correct_flag(r) is True)

    summary = {
        "total":    total,
        "correct":  correct,
        "accuracy": round(correct / len(comparable) * 100, 1) if comparable else 0,
    }

    if full:
        return jsonify({"summary": summary, "rows": rows})

    slim = [{
        "timestamp":       r.get("timestamp"),
        "filename":        r.get("filename"),
        "station":         r.get("station"),
        "predicted_label": r.get("predicted_label"),
        "predicted_cause_top": r.get("predicted_cause_top"),
        "confirmed_label": r.get("confirmed_label"),
        "correct":         _row_correct_flag(r),
        "confidence":      r.get("predicted_conf"),
        "tier":            r.get("tier"),
        "zone":            r.get("zone"),
        "phases":          r.get("phases"),
        "duration_ms":     r.get("duration_ms"),
        "reclose_ok":      r.get("reclose_ok"),
    } for r in rows]

    return jsonify({"summary": summary, "rows": slim})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Transformer helpers
# ─────────────────────────────────────────────────────────────────────────────

XFMR_HISTORY_CSV = Path(__file__).parent / "transformer_history.csv"
XFMR_HISTORY_FIELDS = [
    "timestamp", "filename", "station", "relay_family",
    "event_class", "predicted_label", "predicted_conf", "rule_name",
    "confirmed_label", "correct", "notes",
    "h2_ratio_max_pct", "h5_ratio_max_pct", "idiff_max_pu", "irstr_max_pu",
    "slope_worst_pct", "dc_offset_index_max", "energisation_flag",
    "fault_duration_ms", "action_required",
]

XFMR_FAULT_CATEGORIES = [
    "INRUSH MAGNETISASI",
    "GANGGUAN INTERNAL TRAFO",
    "GANGGUAN EKSTERNAL (THROUGH)",
    "TEGANGAN LEBIH / OVEREKSITASI",
    "KEMUNGKINAN MALOPERATE",
    "LAIN-LAIN",
]

XFMR_EVENT_CLASS_LABELS = {
    "INRUSH": "INRUSH MAGNETISASI",
    "INTERNAL_FAULT": "GANGGUAN INTERNAL TRAFO",
    "THROUGH_FAULT": "GANGGUAN EKSTERNAL (THROUGH)",
    "OVEREXCITATION": "TEGANGAN LEBIH / OVEREKSITASI",
    "MAL_OPERATE": "KEMUNGKINAN MALOPERATE",
    "LAIN-LAIN": "LAIN-LAIN",
}

XFMR_LABEL_TO_EVENT_CLASS = {
    _normalize_label(label): code
    for code, label in XFMR_EVENT_CLASS_LABELS.items()
}


def _xfmr_confirmed_to_event_class(value: str) -> str:
    norm = _normalize_label(value)
    return XFMR_LABEL_TO_EVENT_CLASS.get(norm, norm)


def _xfmr_event_class_label(value: str) -> str:
    code = _xfmr_confirmed_to_event_class(value)
    return XFMR_EVENT_CLASS_LABELS.get(code, value or "-")


def _xfmr_row_correct_flag(row: dict):
    pred = _xfmr_confirmed_to_event_class(_s(row.get("event_class"), "").strip())
    conf = _xfmr_confirmed_to_event_class(_s(row.get("confirmed_label"), "").strip())
    if not pred or not conf:
        return None
    return pred == conf


def _inject_transformer_fields(result, payload: dict) -> None:
    """
    If result is a transformer event, add all Phase 2 fields to the analysis payload dict.
    Called after the standard payload is assembled in upload / analyze-from-browse routes.
    """
    event_type = getattr(result, 'event_type', 'LINE')
    payload['event_type'] = event_type

    if event_type != 'TRANSFORMER':
        return

    xr = getattr(result, 'transformer_result', None)
    tf = getattr(result, 'transformer_features', None) or {}
    class_probs = getattr(xr, 'class_probabilities', {}) if xr else {}
    if not isinstance(class_probs, dict):
        class_probs = {}

    evidence = getattr(xr, 'evidence', '') if xr else ''
    if isinstance(evidence, (list, tuple)):
        evidence_text = '; '.join(_s(x, '') for x in evidence if _s(x, '').strip())
    else:
        evidence_text = _s(evidence, '')

    payload['event_class']           = xr.event_class if xr else 'UNKNOWN'
    payload['action_required']       = bool(xr.action_required) if xr else False
    payload['limited_data']          = bool(xr.limited_data) if xr else False
    payload['relay_family']          = _s(tf.get('relay_family'), 'UNKNOWN')

    # Layered output (README: Arah Berikutnya #1)
    payload['fault_origin']              = xr.fault_origin if xr else 'UNKNOWN'
    payload['fault_origin_reason']       = xr.fault_origin_reason if xr else ''
    payload['protection_assessment']     = xr.protection_assessment if xr else 'EVIDENCE_KURANG'
    payload['protection_assessment_reason'] = xr.protection_assessment_reason if xr else ''

    # Harmonic features
    payload['h2_ratio_a_pct']        = _f(tf.get('h2_ratio_a_pct'))
    payload['h2_ratio_b_pct']        = _f(tf.get('h2_ratio_b_pct'))
    payload['h2_ratio_c_pct']        = _f(tf.get('h2_ratio_c_pct'))
    payload['h2_ratio_max_pct']      = _f(tf.get('h2_ratio_max_pct'))
    payload['h5_ratio_a_pct']        = _f(tf.get('h5_ratio_a_pct'))
    payload['h5_ratio_b_pct']        = _f(tf.get('h5_ratio_b_pct'))
    payload['h5_ratio_c_pct']        = _f(tf.get('h5_ratio_c_pct'))
    payload['h5_ratio_max_pct']      = _f(tf.get('h5_ratio_max_pct'))
    payload['thd_diff_pct']          = _f(tf.get('thd_diff_pct'))

    # Differential / restraint
    payload['idiff_max_pu']          = _f(tf.get('idiff_max_pu'))
    payload['irstr_max_pu']          = _f(tf.get('irstr_max_pu'))
    payload['slope_worst_pct']       = _f(tf.get('slope_worst_pct'))
    payload['above_slope1']          = bool(tf.get('above_slope1') or False)
    payload['above_slope2']          = bool(tf.get('above_slope2') or False)

    # Waveform shape
    payload['dc_offset_index_max']   = _f(tf.get('dc_offset_index_max'))
    payload['pp_asymmetry_a']        = _f(tf.get('pp_asymmetry_a'))
    payload['zc_interval_variance']  = _f(tf.get('zc_interval_variance'))

    # Balance
    payload['hv_lv_ratio_a']         = _f(tf.get('hv_lv_ratio_a'))
    payload['hv_lv_phase_diff_a_deg']= _f(tf.get('hv_lv_phase_diff_a_deg'))

    # Context
    payload['energisation_flag']     = bool(tf.get('energisation_flag') or False)
    payload['lv_prefault_irms_pu']   = _f(tf.get('lv_prefault_irms_pu'))
    payload['inception_angle_deg']   = _f(tf.get('inception_angle_deg'))
    payload['peak_idiff_a']          = _f(tf.get('peak_idiff_a'))

    # MVA / nominal current (for HV/LV current level display per IEEE C37.91)
    payload['estimated_mva']         = _f(tf.get('estimated_mva'))
    payload['hv_base_current_a']     = _f(tf.get('hv_base_current_a'))  # I_rated HV
    payload['lv_base_current_a']     = _f(tf.get('lv_base_current_a'))  # I_rated LV
    payload['fault_duration_ms']     = _f(tf.get('fault_duration_ms'))

    # Channel availability (propagated from TransformerFeatures.channels_available)
    raw_avail = tf.get('channels_available') or {}
    payload['channels_available'] = {
        key: bool(raw_avail.get(key, False))
        for key in ['i_hv_a','i_hv_b','i_hv_c','i_lv_a','i_lv_b','i_lv_c',
                    'i_diff_a','i_diff_b','i_diff_c','i_rstr_a','i_rstr_b','i_rstr_c']
    }

    # Classifier rule detail
    payload['evidence']              = evidence_text
    payload['rule_name']             = xr.rule_name if xr else ''
    payload['recommendation']        = xr.recommendation if xr else ''
    payload['cause_pcts'] = [
        {'name': k.replace('_', ' ').title(), 'pct': round(v * 100, 1)}
        for k, v in class_probs.items()
    ]

    # Warnings
    payload['warnings'] = list(getattr(xr, 'warnings', [])) if xr else []


# ─────────────────────────────────────────────────────────────────────────────
# Multi-relay evidence fusion (COORDINATED mode)
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for fault_origin — higher index = stronger evidence
_ORIGIN_STRENGTH = {
    'UNKNOWN':             0,
    'EVIDENCE_KURANG':     0,   # alias used by protection_assessment
    'PROTEKSI_PERALATAN':  1,
    'EXTERNAL_TRANSMISI':  2,
    'EXTERNAL_DISTRIBUSI': 3,
    'INTERNAL_TRAFO':      4,
}

# Priority order for protection_assessment
_PA_STRENGTH = {
    'EVIDENCE_KURANG': 0,
    'BENAR':           1,
    'TIDAK_SELEKTIF':  2,
}

# Which relay roles are most authoritative for each question
_ORIGIN_AUTHORITY = {
    # role → weight for determining fault_origin (0–3)
    '87T':     3,
    'REF':     2,
    'GFR_HV':  2,
    'SBEF':    2,
    'OCR_HV':  1,
    'OCR_LV':  2,  # strong for external-distribution claim
    'FEEDER':  3,  # feeder trip = strongest external-distribution signal
    'OTHER':   0,
    'AUTO':    1,
}

_PA_AUTHORITY = {
    '87T':     3,
    'OCR_HV':  2,
    'OCR_LV':  2,
    'FEEDER':  2,
    'REF':     1,
    'GFR_HV':  1,
    'SBEF':    1,
    'OTHER':   0,
    'AUTO':    1,
}


def _analyse_supporting_relay(cfg_path: str, role: str) -> dict:
    """
    Run classify_file on a supporting relay file and return a slim evidence dict:
      {role, role_label, event_class, fault_origin, protection_assessment,
       confidence, limited_data, evidence, fault_origin_reason,
       protection_assessment_reason}
    Returns a fallback dict if classification fails.
    """
    role_label = _transformer_role_label(role)
    fallback = {
        'role': role, 'role_label': role_label,
        'event_class': 'UNKNOWN', 'confidence': 0.0,
        'fault_origin': 'UNKNOWN', 'fault_origin_reason': '',
        'protection_assessment': 'EVIDENCE_KURANG',
        'protection_assessment_reason': 'File tidak dapat dianalisis.',
        'limited_data': True, 'evidence': '',
    }
    try:
        result = classify_file(cfg_path)
    except Exception as e:
        fallback['protection_assessment_reason'] = f'Error: {e}'
        return fallback

    xr = getattr(result, 'transformer_result', None)

    if xr is None:
        return _derive_supporting_line_evidence(result, role, role_label)

    return {
        'role':  role,
        'role_label': role_label,
        'event_class': xr.event_class,
        'confidence':  xr.confidence,
        'fault_origin': xr.fault_origin,
        'fault_origin_reason': xr.fault_origin_reason,
        'protection_assessment': xr.protection_assessment,
        'protection_assessment_reason': xr.protection_assessment_reason,
        'limited_data': xr.limited_data,
        'evidence': xr.evidence if isinstance(xr.evidence, str) else '',
    }


def _derive_supporting_line_evidence(result, role: str, role_label: str) -> dict:
    """Convert a non-87T classification result into conservative transformer-side evidence."""
    label = _s(getattr(result, "label", ""), "")
    event_class = _top_cause_name(result) or label or "UNKNOWN"
    label_norm = _normalize_label(label)
    event_norm = _normalize_label(event_class)
    features = getattr(result, "features", None) or {}
    confidence = max(_f(getattr(result, "confidence", 0.0), 0.0), 0.0)
    fault_type = _normalize_label(_s(features.get("fault_type"), ""))
    ground_indicated = (
        fault_type in {"SLG", "DLG"} or
        _f(features.get("i0_i1_ratio"), 0.0) >= 0.25 or
        _f(features.get("i0_magnitude_a"), 0.0) > 0.0
    )

    evidence_bits = []
    if label:
        evidence_bits.append(f"Prediksi lokal {label}")
    if role in {"OCR_LV", "FEEDER"} and ground_indicated:
        evidence_bits.append("indikasi earth fault/downstream terlihat pada relay sisi LV")
    elif role in {"REF", "GFR_HV", "SBEF"} and ground_indicated:
        evidence_bits.append("relay earth-fault ikut merekam event")
    elif role == "OCR_HV":
        evidence_bits.append("relay OCR HV ikut melihat gangguan sebagai evidence koordinasi")
    evidence = ". ".join(evidence_bits).strip()

    if "PERALATAN" in label_norm or "PERALATAN" in event_norm:
        return {
            'role': role,
            'role_label': role_label,
            'event_class': event_class,
            'confidence': confidence,
            'fault_origin': 'PROTEKSI_PERALATAN',
            'fault_origin_reason': (
                f"[{role_label}] Rekaman ini lebih mengarah ke isu peralatan/proteksi "
                f"daripada external fault murni."
            ),
            'protection_assessment': 'TIDAK_SELEKTIF',
            'protection_assessment_reason': (
                f"[{role_label}] Indikasi {event_class} membuat operasi relay perlu "
                "direview sebagai potensi gangguan peralatan/proteksi."
            ),
            'limited_data': False,
            'evidence': evidence,
        }

    if role in {'FEEDER', 'OCR_LV'}:
        return {
            'role': role,
            'role_label': role_label,
            'event_class': event_class,
            'confidence': confidence,
            'fault_origin': 'EXTERNAL_DISTRIBUSI',
            'fault_origin_reason': (
                f"[{role_label}] Relay sisi hilir ikut melihat dan mengisolasi gangguan, "
                "lebih konsisten dengan fault distribusi/downstream daripada internal trafo."
            ),
            'protection_assessment': 'BENAR',
            'protection_assessment_reason': (
                f"[{role_label}] Operasi relay hilir menjadi bukti kuat bahwa proteksi lokal "
                "di sisi distribusi bekerja sebagai first-clearing layer."
            ),
            'limited_data': False,
            'evidence': evidence,
        }

    if role in {'REF', 'GFR_HV', 'SBEF'}:
        return {
            'role': role,
            'role_label': role_label,
            'event_class': event_class,
            'confidence': confidence,
            'fault_origin': 'UNKNOWN',
            'fault_origin_reason': (
                f"[{role_label}] Relay ini memberi konteks jalur gangguan tanah, "
                "tetapi origin final tetap perlu disandingkan dengan 87T atau relay pasangan."
            ),
            'protection_assessment': 'BENAR' if ground_indicated else 'EVIDENCE_KURANG',
            'protection_assessment_reason': (
                f"[{role_label}] Event terekam pada elemen earth-fault; evidence ini relevan "
                "untuk koordinasi, tetapi belum cukup untuk memutuskan internal vs external sendirian."
            ),
            'limited_data': False,
            'evidence': evidence,
        }

    if role == 'OCR_HV':
        return {
            'role': role,
            'role_label': role_label,
            'event_class': event_class,
            'confidence': confidence,
            'fault_origin': 'UNKNOWN',
            'fault_origin_reason': (
                f"[{role_label}] OCR HV memberi evidence koordinasi sisi atas, "
                "namun satu file ini saja belum cukup untuk menegaskan origin akhir."
            ),
            'protection_assessment': 'EVIDENCE_KURANG',
            'protection_assessment_reason': (
                f"[{role_label}] Bandingkan dengan OCR LV/feeder atau 87T untuk menilai "
                "apakah trip sisi HV memang diperlukan atau hanya backup."
            ),
            'limited_data': False,
            'evidence': evidence,
        }

    return {
        'role': role,
        'role_label': role_label,
        'event_class': event_class,
        'confidence': confidence,
        'fault_origin': 'UNKNOWN',
        'fault_origin_reason': (
            f"[{role_label}] File berhasil dibaca sebagai evidence lokal, tetapi peran relay ini "
            "belum cukup spesifik untuk menentukan origin tanpa pasangan relay lain."
        ),
        'protection_assessment': 'EVIDENCE_KURANG',
        'protection_assessment_reason': (
            f"[{role_label}] Jadikan file ini sebagai supporting evidence, "
            "bukan penentu final internal vs external sendirian."
        ),
        'limited_data': False,
        'evidence': evidence,
    }


def _fuse_multi_relay_evidence(primary_payload: dict,
                                support_evidence: list[dict]) -> dict:
    """
    Fuse evidence from the primary relay and all supporting relays.

    Returns a dict of fused fields that overwrite the primary payload:
      fault_origin, fault_origin_reason,
      protection_assessment, protection_assessment_reason,
      fusion_summary   (list of per-relay rows for the results table)
      fusion_conflicts (list of conflict strings shown as warnings)
    """
    primary_role = (primary_payload.get('transformer_relay_role') or 'AUTO').upper()

    # Build unified evidence list: primary first, then supporting
    all_evidence = [{
        'role':  primary_role,
        'role_label': _transformer_role_label(primary_role),
        'event_class': primary_payload.get('event_class', 'UNKNOWN'),
        'confidence':  primary_payload.get('confidence', 0.0),
        'fault_origin': primary_payload.get('fault_origin', 'UNKNOWN'),
        'fault_origin_reason': primary_payload.get('fault_origin_reason', ''),
        'protection_assessment': primary_payload.get('protection_assessment', 'EVIDENCE_KURANG'),
        'protection_assessment_reason': primary_payload.get('protection_assessment_reason', ''),
        'limited_data': primary_payload.get('limited_data', False),
        'evidence': primary_payload.get('evidence', ''),
        'is_primary': True,
    }] + [{**e, 'is_primary': False} for e in support_evidence]

    # ── Fault origin: pick highest-authority non-UNKNOWN origin ──────────────
    best_origin = 'UNKNOWN'
    best_origin_reason = 'Tidak ada relay yang memberikan indikasi origin yang cukup kuat.'
    best_origin_weight = -1

    for ev in all_evidence:
        origin = ev.get('fault_origin', 'UNKNOWN')
        if origin in ('UNKNOWN', ''):
            continue
        role_w = _ORIGIN_AUTHORITY.get(ev['role'], 0)
        origin_s = _ORIGIN_STRENGTH.get(origin, 0)
        weight = role_w * origin_s * max(ev.get('confidence', 0.0), 0.1)
        if weight > best_origin_weight:
            best_origin_weight = weight
            best_origin = origin
            best_origin_reason = (
                f"[{ev['role_label']}] {ev.get('fault_origin_reason', '')} "
                f"(confidence {ev['confidence']:.0%})"
            )

    # ── Override: if FEEDER or OCR_LV fired and is non-UNKNOWN,
    #    that's strong external-distribution evidence regardless of primary ────
    for ev in all_evidence:
        if ev['role'] in ('FEEDER', 'OCR_LV') and \
                ev.get('fault_origin') == 'EXTERNAL_DISTRIBUSI' and \
                ev.get('confidence', 0) >= 0.40:
            best_origin = 'EXTERNAL_DISTRIBUSI'
            best_origin_reason = (
                f"[{ev['role_label']}] Trip konfirmasikan gangguan sisi distribusi — "
                f"override dari relay {ev['role_label']}. "
                f"{ev.get('fault_origin_reason','')}"
            )
            break

    # ── Protection assessment: use highest-authority non-EVIDENCE_KURANG ─────
    best_pa = 'EVIDENCE_KURANG'
    best_pa_reason = 'Belum ada relay yang memberikan penilaian selektivitas yang kuat.'
    best_pa_weight = -1

    for ev in all_evidence:
        pa = ev.get('protection_assessment', 'EVIDENCE_KURANG')
        if pa == 'EVIDENCE_KURANG':
            continue
        role_w = _PA_AUTHORITY.get(ev['role'], 0)
        pa_s = _PA_STRENGTH.get(pa, 0)
        weight = role_w * pa_s * max(ev.get('confidence', 0.0), 0.1)
        if weight > best_pa_weight:
            best_pa_weight = weight
            best_pa = pa
            best_pa_reason = (
                f"[{ev['role_label']}] {ev.get('protection_assessment_reason', '')} "
                f"(confidence {ev['confidence']:.0%})"
            )

    # ── Detect conflicts ──────────────────────────────────────────────────────
    conflicts = []
    origins_seen = set(ev.get('fault_origin', 'UNKNOWN')
                       for ev in all_evidence
                       if ev.get('fault_origin', 'UNKNOWN') not in ('UNKNOWN', ''))
    if 'INTERNAL_TRAFO' in origins_seen and \
            ('EXTERNAL_DISTRIBUSI' in origins_seen or 'EXTERNAL_TRANSMISI' in origins_seen):
        conflicts.append(
            "Konflik: satu relay menunjukkan gangguan INTERNAL, relay lain menunjukkan "
            "EKSTERNAL. Perlu review manual sebelum kesimpulan final."
        )

    pa_seen = set(ev.get('protection_assessment', 'EVIDENCE_KURANG')
                  for ev in all_evidence
                  if ev.get('protection_assessment') not in ('EVIDENCE_KURANG', ''))
    if 'BENAR' in pa_seen and 'TIDAK_SELEKTIF' in pa_seen:
        conflicts.append(
            "Konflik selektivitas: sebagian relay menunjukkan operasi benar, "
            "sebagian lagi menunjukkan tidak selektif. Periksa urutan trip."
        )

    return {
        'fault_origin': best_origin,
        'fault_origin_reason': best_origin_reason.strip(),
        'protection_assessment': best_pa,
        'protection_assessment_reason': best_pa_reason.strip(),
        'fusion_summary': all_evidence,
        'fusion_conflicts': conflicts,
    }


def _save_xfmr_to_csv(row: dict) -> None:
    """Append a transformer event row to transformer_history.csv."""
    write_header = not XFMR_HISTORY_CSV.exists()
    with open(XFMR_HISTORY_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=XFMR_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: _s(row.get(k), '') for k in XFMR_HISTORY_FIELDS})


def _db_insert_xfmr_feedback(row: dict, source_ip: str, user_agent: str) -> bool:
    """Insert transformer feedback into PostgreSQL transformer_feedback table."""
    if not _db_enabled():
        return False
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO transformer_feedback (
                        timestamp_txt, filename, station, relay_family,
                        event_class, predicted_label, predicted_conf, rule_name,
                        confirmed_label, correct, notes,
                        h2_ratio_max_pct, h5_ratio_max_pct, idiff_max_pu, irstr_max_pu,
                        slope_worst_pct, dc_offset_index_max, energisation_flag,
                        fault_duration_ms, action_required, source_ip, user_agent
                    ) VALUES (
                        %(timestamp)s, %(filename)s, %(station)s, %(relay_family)s,
                        %(event_class)s, %(predicted_label)s, %(predicted_conf)s, %(rule_name)s,
                        %(confirmed_label)s, %(correct)s, %(notes)s,
                        %(h2_ratio_max_pct)s, %(h5_ratio_max_pct)s, %(idiff_max_pu)s, %(irstr_max_pu)s,
                        %(slope_worst_pct)s, %(dc_offset_index_max)s, %(energisation_flag)s,
                        %(fault_duration_ms)s, %(action_required)s, %(source_ip)s, %(user_agent)s
                    )
                    """,
                    {
                        **{k: _s(row.get(k), '') for k in XFMR_HISTORY_FIELDS},
                        'source_ip': source_ip,
                        'user_agent': user_agent,
                    },
                )
        return True
    except Exception as e:
        app.logger.warning("Postgres transformer insert failed: %s", e)
        return False


def _load_xfmr_history(limit: int | None = None) -> list[dict]:
    """Load transformer history from DB or CSV."""
    if _db_enabled():
        try:
            q = (
                "SELECT timestamp_txt, filename, station, relay_family, event_class, "
                "predicted_label, predicted_conf, rule_name, confirmed_label, correct, notes, "
                "h2_ratio_max_pct, h5_ratio_max_pct, idiff_max_pu, irstr_max_pu, "
                "slope_worst_pct, dc_offset_index_max, energisation_flag, "
                "fault_duration_ms, action_required "
                "FROM transformer_feedback ORDER BY id DESC"
            )
            params = ()
            if limit:
                q += " LIMIT %s"
                params = (limit,)
            rows = []
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(q, params)
                    cols = [d[0] for d in cur.description]
                    for r in cur.fetchall():
                        rows.append(dict(zip(cols, [_s(v, '') for v in r])))
            return rows
        except Exception as e:
            app.logger.warning("Postgres transformer fetch failed: %s", e)

    # CSV fallback
    if not XFMR_HISTORY_CSV.exists():
        return []
    with open(XFMR_HISTORY_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows.reverse()
    return rows[:limit] if limit else rows


# ── Transformer routes ────────────────────────────────────────────────────────

@app.route("/confirm-transformer", methods=["POST"])
def confirm_transformer():
    """Save operator confirmation for a transformer event."""
    analysis = _load_analysis_from_store()
    if not analysis:
        return redirect(url_for("index"))

    confirmed_input = request.form.get("confirmed_label", "").strip()
    confirmed_label = _xfmr_confirmed_to_event_class(confirmed_input)
    notes           = request.form.get("notes", "").strip()

    pred = _xfmr_confirmed_to_event_class(_s(analysis.get("event_class"), "UNKNOWN"))
    corr = (
        "True"  if confirmed_label and confirmed_label == pred else
        "False" if confirmed_input else ""
    )

    row = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename":           _s(analysis.get("original_filename")),
        "station":            _s(analysis.get("station_name")),
        "relay_family":       _s(analysis.get("relay_family")),
        "event_class":        pred,
        "predicted_label":    _s(analysis.get("label")),
        "predicted_conf":     _s(round(_f(analysis.get("confidence")) * 100, 1)),
        "rule_name":          _s(analysis.get("rule_name")),
        "confirmed_label":    confirmed_label or confirmed_input,
        "correct":            corr,
        "notes":              notes,
        "h2_ratio_max_pct":   _s(round(_f(analysis.get("h2_ratio_max_pct")), 2)),
        "h5_ratio_max_pct":   _s(round(_f(analysis.get("h5_ratio_max_pct")), 2)),
        "idiff_max_pu":       _s(round(_f(analysis.get("idiff_max_pu")), 4)),
        "irstr_max_pu":       _s(round(_f(analysis.get("irstr_max_pu")), 4)),
        "slope_worst_pct":    _s(round(_f(analysis.get("slope_worst_pct")), 2)),
        "dc_offset_index_max":_s(round(_f(analysis.get("dc_offset_index_max")), 4)),
        "energisation_flag":  _s(analysis.get("energisation_flag")),
        "fault_duration_ms":  _s(round(_f(analysis.get("fault_duration_ms")), 1)),
        "action_required":    _s(analysis.get("action_required")),
    }

    source_ip  = request.remote_addr or ""
    user_agent = request.headers.get("User-Agent", "")[:200]

    if not _db_insert_xfmr_feedback(row, source_ip, user_agent):
        _save_xfmr_to_csv(row)

    _clear_analysis_store()
    return redirect(url_for("success"))


@app.route("/transformer/history")
def transformer_history():
    """History page for transformer 87T events."""
    rows = _load_xfmr_history(limit=200)
    total = len(rows)
    for r in rows:
        eval_flag = _xfmr_row_correct_flag(r)
        r["correct"] = "True" if eval_flag is True else ("False" if eval_flag is False else "")
        r["confirmed_label_display"] = _xfmr_event_class_label(r.get("confirmed_label"))
        r["event_class_display"] = _xfmr_event_class_label(r.get("event_class"))
    confirmed = [r for r in rows if _xfmr_confirmed_to_event_class(r.get("confirmed_label", ""))]
    correct   = [r for r in confirmed if r.get("correct") == "True"]
    accuracy  = round(len(correct) / len(confirmed) * 100, 1) if confirmed else 0

    # Class distribution
    from collections import Counter
    class_dist = Counter(r.get("event_class", "UNKNOWN") for r in rows)

    return render_template("transformer_history.html",
                           rows=rows,
                           total=total,
                           accuracy=accuracy,
                           confirmed_count=len(confirmed),
                           class_dist=dict(class_dist),
                           db_enabled=_db_enabled())


@app.route("/transformer/trends")
def transformer_trends():
    """Trend analysis for transformer events."""
    rows = _load_xfmr_history()
    from collections import Counter, defaultdict

    class_counts = Counter(r.get("event_class", "UNKNOWN") for r in rows)
    station_counts = Counter(r.get("station", "-") for r in rows)
    action_count = sum(1 for r in rows if r.get("action_required") == "True")

    # Monthly trend
    monthly: dict = defaultdict(Counter)
    for r in rows:
        ts = r.get("timestamp", "")[:7]   # "YYYY-MM"
        ec = r.get("event_class", "UNKNOWN")
        if ts:
            monthly[ts][ec] += 1
    monthly_sorted = sorted(monthly.items())

    return render_template("transformer_trends.html",
                           class_counts=dict(class_counts),
                           station_counts=dict(station_counts.most_common(10)),
                           action_count=action_count,
                           total=len(rows),
                           monthly=monthly_sorted,
                           db_enabled=_db_enabled())


@app.route("/transformer/data-status")
def transformer_data_status():
    """Phase 2 data readiness dashboard — shows how many labeled events per class."""
    rows = _load_xfmr_history()
    confirmed = [r for r in rows if _xfmr_confirmed_to_event_class(r.get("confirmed_label", ""))]

    from collections import Counter
    confirmed_dist = Counter(
        _xfmr_confirmed_to_event_class(r.get("confirmed_label", ""))
        for r in confirmed
        if _xfmr_confirmed_to_event_class(r.get("confirmed_label", ""))
    )
    synthetic_dir  = Path(__file__).parent.parent / "data" / "synthetic" / "transformer"

    # Count synthetic files per class
    synthetic_counts: dict = {}
    for cls in ["INRUSH", "INTERNAL_FAULT", "THROUGH_FAULT", "OVEREXCITATION", "MAL_OPERATE"]:
        d = synthetic_dir / cls
        synthetic_counts[cls] = len(list(d.glob("*.cfg"))) if d.exists() else 0

    min_for_training = 30
    readiness = {
        cls: {
            "real": confirmed_dist.get(cls, 0),
            "synthetic": synthetic_counts.get(cls, 0),
            "ready": (confirmed_dist.get(cls, 0) >= min_for_training),
        }
        for cls in ["INRUSH", "INTERNAL_FAULT", "THROUGH_FAULT", "OVEREXCITATION", "MAL_OPERATE"]
    }

    total_real = sum(r["real"] for r in readiness.values())
    classes_ready = sum(1 for r in readiness.values() if r["ready"])

    return render_template("transformer_data_status.html",
                           readiness=readiness,
                           total_real=total_real,
                           classes_ready=classes_ready,
                           min_for_training=min_for_training,
                           synthetic_dir=str(synthetic_dir),
                           db_enabled=_db_enabled())


@app.route("/api/transformer/status")
def api_transformer_status():
    """JSON API: transformer data readiness summary."""
    rows = _load_xfmr_history()
    confirmed = [r for r in rows if r.get("confirmed_label")]
    from collections import Counter
    dist = Counter(r.get("confirmed_label", "") for r in confirmed)
    return jsonify({
        "total_events": len(rows),
        "confirmed":    len(confirmed),
        "class_distribution": dict(dist),
        "phase2_ready": len(confirmed) >= 30,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  DFR Fault Classifier - Web App")
    print("  http://localhost:5000")
    print("")
    print("  API endpoints:")
    print("    GET  /api/status")
    print("    POST /api/classify  (cfg_file + dat_file)")
    print("    GET  /api/history")
    print("=" * 60)
    app.run(debug=IS_LOCAL_DEV, use_reloader=False, host="0.0.0.0", port=5000)
