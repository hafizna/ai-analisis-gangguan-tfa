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

import sys
import csv
import re
import warnings
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# Bootstrap path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predict import classify_file, extract_soe_from_file


def _f(val, default=0):
    """Cast numpy/float32 to plain Python float for JSON serialisation."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _s(val, default="-"):
    """Cast to plain str."""
    return str(val) if val is not None else default

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
app.secret_key = "dfr-fault-classifier-2026"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

import tempfile
UPLOAD_DIR  = Path(tempfile.gettempdir()) / "dfr_uploads"
HISTORY_CSV = Path(__file__).parent / "history.csv"
UPLOAD_DIR.mkdir(exist_ok=True)

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

    session["analysis"] = {
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
        "voltage_kv":           _s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui",
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
    }

    return redirect(url_for("results"))


@app.route("/results")
def results():
    analysis = session.get("analysis")
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
    session["analysis"] = {
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
        "voltage_kv":           _s(feats.get("voltage_kv")) if feats.get("voltage_kv") else "Tidak diketahui",
        "scaling_ok":           _f(feats.get("peak_fault_current_a", 0)) >= 200.0,
    }
    return redirect(url_for("results"))


@app.route("/confirm", methods=["POST"])
def confirm_prediction():
    analysis = session.get("analysis")
    if not analysis:
        return jsonify({"error": "No analysis in session"}), 400

    confirmed = request.form.get("fault_cause", "LAIN-LAIN")
    notes     = request.form.get("notes", "")
    correct   = confirmed == analysis["label"]

    row = {
        "timestamp":          analysis["timestamp"],
        "filename":           analysis["original_filename"],
        "station":            analysis["station_name"],
        "predicted_label":    analysis["label"],
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

    # Append to history CSV
    write_header = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    session.pop("analysis", None)
    return redirect(url_for("success"))


@app.route("/success")
def success():
    return render_template("success.html")


@app.route("/history")
def history():
    rows = []
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.reverse()   # newest first

    # Simple accuracy stats
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct") == "True")
    accuracy = round(correct / total * 100, 1) if total else 0

    return render_template("history.html", rows=rows,
                           total=total, correct=correct, accuracy=accuracy)


if __name__ == "__main__":
    print("=" * 60)
    print("  DFR Fault Classifier - Web App")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
