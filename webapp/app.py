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

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
try:
    import psycopg
except Exception:
    psycopg = None

warnings.filterwarnings("ignore")

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


def _infer_ratio_from_parser_multiplier(measurement: str, mult: float) -> tuple[float, float, bool]:
    """
    Convert parser scale multiplier to a practical P/S ratio for UI default.
    This is only used when explicit cfg primary/secondary is unavailable (often 1/1).
    """
    m = _f(mult, 0.0)
    if measurement == "current":
        if 5 <= m <= 10000:
            return float(round(m)), 1.0, True
        return 1.0, 1.0, False
    if measurement == "voltage":
        if 10 <= m <= 10000:
            return float(round(m * 100)), 100.0, True
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
                "parser_ct_multiplier": 0.0, "parser_vt_multiplier": 0.0,
            }
        ctp, cts, ct_known = _pick_representative_ratio(rec.analog_channels, "current")
        vtp, vts, vt_known = _pick_representative_ratio(rec.analog_channels, "voltage")
        ct_mult = _pick_representative_scale_multiplier(rec.analog_channels, "current")
        vt_mult = _pick_representative_scale_multiplier(rec.analog_channels, "voltage")
        return {
            "cfg_ct_primary": ctp, "cfg_ct_secondary": cts,
            "cfg_vt_primary": vtp, "cfg_vt_secondary": vts,
            "cfg_ct_known": ct_known, "cfg_vt_known": vt_known,
            "parser_ct_multiplier": ct_mult, "parser_vt_multiplier": vt_mult,
        }
    except Exception:
        return {
            "cfg_ct_primary": 1.0, "cfg_ct_secondary": 1.0,
            "cfg_vt_primary": 1.0, "cfg_vt_secondary": 1.0,
            "cfg_ct_known": False, "cfg_vt_known": False,
            "parser_ct_multiplier": 0.0, "parser_vt_multiplier": 0.0,
        }


def _nearest_supported_voltage_kv(v_kv: float) -> float:
    levels = [30.0, 70.0, 150.0, 275.0, 500.0]
    v = _f(v_kv, 150.0)
    return min(levels, key=lambda x: abs(x - v))


def _assumed_transformer_ratios(voltage_kv, peak_current_a) -> dict:
    """
    Return practical default CT/VT ratio assumptions for transmission systems.
    Supported nominal voltage levels: 30/75/150/275/500 kV.
    VT assumption uses common 100V secondary (LL basis) -> ratio ~= Vll/100.
    CT assumption uses common primary steps with 1A secondary and is nudged by
    measured peak current when available.
    """
    v_sys = _nearest_supported_voltage_kv(_f(voltage_kv, 150.0))
    vt_primary = int(round(v_sys * 10)) * 100  # 30kV->30000, 150kV->150000
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

    return {
        "assumed_voltage_kv": v_sys,
        "assumed_ct_primary": float(ct_primary),
        "assumed_ct_secondary": float(ct_secondary),
        "assumed_vt_primary": float(vt_primary),
        "assumed_vt_secondary": float(vt_secondary),
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
            "orig_ct_p": _f(analysis.get("ct_primary"), 1.0),
            "orig_ct_s": _f(analysis.get("ct_secondary"), 1.0),
            "orig_vt_p": _f(analysis.get("vt_primary"), 1.0),
            "orig_vt_s": _f(analysis.get("vt_secondary"), 1.0),
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

    # Persist ratio inputs for the UI
    updated["ct_primary"] = ct_p
    updated["ct_secondary"] = ct_s
    updated["vt_primary"] = vt_p
    updated["vt_secondary"] = vt_s
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
            "Lakukan inspeksi lapangan sesuai indikasi gangguan permanen/peralatan."
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

    reclose_ok = _to_bool_or_none(row.get("reclose_successful"))
    if reclose_ok is True and row["peak_fault_current_a"] > 200:
        updated["label"] = "GANGGUAN TRANSIEN"
        updated["confidence"] = 0.95
        updated["tier"] = 1
        updated["rule_name"] = "reclose_confirmed_transient"
        updated["cause_pcts"] = _compute_cause_pcts(row)
        updated["recommendation"] = _transient_recommendation(row)
        updated["evidence"] = "AR berhasil + arus primer memadai setelah koreksi rasio CT/VT."
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
        clf = model_bundle["clf"]
        X = _build_feature_vector(row)
        pred = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]
        p_trans = float(proba[1])
        updated["label"] = "GANGGUAN TRANSIEN"
        updated["confidence"] = p_trans if pred == 1 else max(p_trans, 0.5)
        updated["tier"] = 2
        updated["rule_name"] = "petir_decision_tree" if pred == 1 else "petir_decision_tree_non_petir"
        updated["cause_pcts"] = _compute_cause_pcts(row)
        updated["recommendation"] = _transient_recommendation(row)
        updated["evidence"] = "Analisa ulang ML berbasis fitur yang sudah dikoreksi rasio CT/VT."
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


def _db_enabled() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _db_connect():
    if not _db_enabled():
        return None
    return psycopg.connect(DATABASE_URL, autocommit=True)


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
    "KONDUKTOR / KERUSAKAN PERALATAN",
    "BENDA ASING",
    "GANGGUAN PERMANEN",
    "LAIN-LAIN",
]

_db_init()


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

LABEL_MAP = [
    ("petir", "PETIR"), ("layang", "LAYANG-LAYANG"),
    ("pohon", "POHON"), ("hewan", "HEWAN"), ("ular", "HEWAN"),
    ("babi", "HEWAN"), ("tower roboh", "KONDUKTOR"),
    ("konduktor", "KONDUKTOR"), ("benda asing", "BENDA ASING"),
]
SKIP_FRAGS = ["olah", "_extracted", "locus", "analisa"]

def _infer_label(path_str):
    low = path_str.lower()
    for frag, lbl in LABEL_MAP:
        if frag in low:
            return lbl
    return ""

def _scan_recordings():
    """Return list of dicts describing every labeled CFG in raw_data/."""
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
        label = _infer_label(path_str)
        if not label:
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
        events.append({
            "upt": upt, "year": year, "month": month, "label": label,
            "gi": gi, "filename": cfg.name, "cfg_path": str(cfg),
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


@app.route("/")
def index():
    return render_template("index.html", fault_categories=FAULT_CATEGORIES)


@app.route("/upload", methods=["POST"])
def upload_files():
    cfg_file = request.files.get("cfg_file")
    dat_file = request.files.get("dat_file")

    if not cfg_file or not dat_file:
        return jsonify({"error": "Both .cfg and .dat files are required"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_name = secure_filename(f"{ts}_{cfg_file.filename}")
    dat_name = secure_filename(f"{ts}_{dat_file.filename}")

    cfg_path = UPLOAD_DIR / cfg_name
    dat_path = UPLOAD_DIR / dat_name
    cfg_file.save(cfg_path)
    dat_file.save(dat_path)

    try:
        result = classify_file(str(cfg_path))
    except ValueError as e:
        return _render_not_supported(cfg_file.filename, str(e), cfg_path=str(cfg_path))
    except Exception as e:
        return _render_not_supported(cfg_file.filename, f"Pipeline error: {e}", cfg_path=str(cfg_path))

    feats = result.features
    cfg_ratios = _extract_cfg_ratios(str(cfg_path))
    assumed_ratios = _assumed_transformer_ratios(feats.get("voltage_kv"), feats.get("peak_fault_current_a"))
    ct_ratio_src = "cfg" if cfg_ratios.get("cfg_ct_known") else "assumed"
    vt_ratio_src = "cfg" if cfg_ratios.get("cfg_vt_known") else "assumed"
    if cfg_ratios.get("cfg_ct_known"):
        ct_p_default = _f(cfg_ratios.get("cfg_ct_primary"), 1.0)
        ct_s_default = _f(cfg_ratios.get("cfg_ct_secondary"), 1.0)
    else:
        ct_p_default = _f(assumed_ratios.get("assumed_ct_primary"), 1.0)
        ct_s_default = _f(assumed_ratios.get("assumed_ct_secondary"), 1.0)
    if cfg_ratios.get("cfg_vt_known"):
        vt_p_default = _f(cfg_ratios.get("cfg_vt_primary"), 1.0)
        vt_s_default = _f(cfg_ratios.get("cfg_vt_secondary"), 1.0)
    else:
        vt_p_default = _f(assumed_ratios.get("assumed_vt_primary"), 1.0)
        vt_s_default = _f(assumed_ratios.get("assumed_vt_secondary"), 1.0)

    analysis_payload = {
        "original_filename": cfg_file.filename,
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
        # key features for display
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
        "voltage_kv":           str(int(_nearest_supported_voltage_kv(vt_p_default / 1000.0))) if (vt_ratio_src == "cfg" and vt_p_default > 1000.0) else (_s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui"),
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
        "ct_primary": ct_p_default,
        "ct_secondary": ct_s_default,
        "vt_primary": vt_p_default,
        "vt_secondary": vt_s_default,
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
        "assumed_voltage_kv": _f(assumed_ratios.get("assumed_voltage_kv"), 150.0),
        "assumed_ct_primary": _f(assumed_ratios.get("assumed_ct_primary"), 2000.0),
        "assumed_ct_secondary": _f(assumed_ratios.get("assumed_ct_secondary"), 1.0),
        "assumed_vt_primary": _f(assumed_ratios.get("assumed_vt_primary"), 150000.0),
        "assumed_vt_secondary": _f(assumed_ratios.get("assumed_vt_secondary"), 100.0),
    }
    _clear_analysis_store()
    _save_analysis_to_store(analysis_payload)

    return redirect(url_for("results"))


@app.route("/results")
def results():
    analysis = _load_analysis_from_store()
    if not analysis:
        return redirect(url_for("index"))
    return render_template("results.html",
                           analysis=analysis,
                           fault_categories=FAULT_CATEGORIES)


@app.route("/browse")
def browse():
    if not RAW_DATA.exists():
        return render_template("browse.html", events=[], upts=[], labels=[],
                               offline_mode=True)
    events = _scan_recordings()
    upts   = sorted(set(e["upt"]   for e in events))
    labels = sorted(set(e["label"] for e in events))
    return render_template("browse.html", events=events, upts=upts, labels=labels,
                           offline_mode=False)


@app.route("/analyze-from-browse", methods=["POST"])
def analyze_from_browse():
    cfg_path = request.form.get("cfg_path", "")
    if not cfg_path or not Path(cfg_path).exists():
        return jsonify({"error": "File tidak ditemukan"}), 404

    try:
        result = classify_file(cfg_path)
    except ValueError as e:
        return _render_not_supported(Path(cfg_path).name, str(e), cfg_path=cfg_path)
    except Exception as e:
        return _render_not_supported(Path(cfg_path).name, f"Pipeline error: {e}", cfg_path=cfg_path)

    feats = result.features
    cfg_ratios = _extract_cfg_ratios(cfg_path)
    assumed_ratios = _assumed_transformer_ratios(feats.get("voltage_kv"), feats.get("peak_fault_current_a"))
    ct_ratio_src = "cfg" if cfg_ratios.get("cfg_ct_known") else "assumed"
    vt_ratio_src = "cfg" if cfg_ratios.get("cfg_vt_known") else "assumed"
    if cfg_ratios.get("cfg_ct_known"):
        ct_p_default = _f(cfg_ratios.get("cfg_ct_primary"), 1.0)
        ct_s_default = _f(cfg_ratios.get("cfg_ct_secondary"), 1.0)
    else:
        ct_p_default = _f(assumed_ratios.get("assumed_ct_primary"), 1.0)
        ct_s_default = _f(assumed_ratios.get("assumed_ct_secondary"), 1.0)
    if cfg_ratios.get("cfg_vt_known"):
        vt_p_default = _f(cfg_ratios.get("cfg_vt_primary"), 1.0)
        vt_s_default = _f(cfg_ratios.get("cfg_vt_secondary"), 1.0)
    else:
        vt_p_default = _f(assumed_ratios.get("assumed_vt_primary"), 1.0)
        vt_s_default = _f(assumed_ratios.get("assumed_vt_secondary"), 1.0)
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
        "voltage_kv":           str(int(_nearest_supported_voltage_kv(vt_p_default / 1000.0))) if (vt_ratio_src == "cfg" and vt_p_default > 1000.0) else (_s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui"),
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
        "ct_primary": ct_p_default,
        "ct_secondary": ct_s_default,
        "vt_primary": vt_p_default,
        "vt_secondary": vt_s_default,
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
        "assumed_voltage_kv": _f(assumed_ratios.get("assumed_voltage_kv"), 150.0),
        "assumed_ct_primary": _f(assumed_ratios.get("assumed_ct_primary"), 2000.0),
        "assumed_ct_secondary": _f(assumed_ratios.get("assumed_ct_secondary"), 1.0),
        "assumed_vt_primary": _f(assumed_ratios.get("assumed_vt_primary"), 150000.0),
        "assumed_vt_secondary": _f(assumed_ratios.get("assumed_vt_secondary"), 100.0),
    }
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
                           total=total, correct=correct, accuracy=accuracy)


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
