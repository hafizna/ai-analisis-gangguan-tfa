"""
Ekstraksi Fitur Batch
=====================
Mencari semua file COMTRADE berlabel di raw_data/, menjalankan pipeline
analisis gangguan, dan menulis fitur ke data/features/labeled_features.csv.

Label is inferred from the folder name (PETIR, LAYANG, POHON, HEWAN,
KONDUKTOR, BENDA_ASING).  Files in 'olah' or '_extracted' sub-folders
are skipped (processed copies).

Run from the pipeline/ directory:
    python batch_extract.py
"""

import os
import sys
import csv
import warnings
import traceback
from pathlib import Path
from dataclasses import asdict

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from core.comtrade_parser import parse_comtrade
from core.protection_router import determine_protection
from core.fault_detector import detect_fault
from core.feature_extractor import extract_distance_features

RAW_DATA = Path(__file__).parent.parent / "raw_data"
OUT_DIR  = Path(__file__).parent / "data" / "features"
OUT_CSV  = OUT_DIR / "labeled_features.csv"
ERR_CSV  = OUT_DIR / "extraction_errors.csv"

# Folder name → label mapping (case-insensitive substring match)
LABEL_MAP = [
    ("petir",        "PETIR"),
    ("layang",       "LAYANG"),
    ("pohon",        "POHON"),
    ("hewan",        "HEWAN"),
    ("ular",         "HEWAN"),
    ("babi",         "HEWAN"),
    ("tower roboh",  "KONDUKTOR"),
    ("konduktor",    "KONDUKTOR"),
    ("benda asing",  "BENDA_ASING"),
]

# Sub-folder fragments to skip (processed copies, analysis outputs)
SKIP_FRAGMENTS = ["olah", "_extracted", "locus z", "locus_z", "locus\\", "/locus/", "analisa"]


def infer_label(path_str: str) -> str:
    low = path_str.lower()
    for fragment, label in LABEL_MAP:
        if fragment in low:
            return label
    return ""


def should_skip(path_str: str) -> bool:
    low = path_str.lower()
    return any(frag in low for frag in SKIP_FRAGMENTS)


def find_labeled_cfgs(root: Path):
    """Yield (cfg_path, label) for every valid labeled CFG file."""
    seen = set()
    for cfg_path in sorted(root.rglob("*.cfg")) + sorted(root.rglob("*.CFG")):
        # Deduplicate: Windows filesystem is case-insensitive so *.cfg and *.CFG
        # can return the same physical file twice.
        resolved = cfg_path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)

        path_str = str(cfg_path)
        if should_skip(path_str.lower()):
            continue
        label = infer_label(path_str)
        if not label:
            continue
        # Check matching DAT exists
        dat = cfg_path.with_suffix(".dat")
        if not dat.exists():
            dat = cfg_path.with_suffix(".DAT")
        if not dat.exists():
            continue
        yield cfg_path, label


def flatten_features(feat, label, cfg_path, prot, fault):
    """Convert DistanceFeatures to a flat dict for CSV output."""
    d = {}
    d["label"]        = label
    d["cfg_path"]     = str(cfg_path)
    d["station_name"] = feat.station_name
    d["relay_model"]  = feat.relay_model
    d["voltage_kv"]   = feat.voltage_kv

    # Protection context
    d["protection_type"]  = prot.primary_protection.name
    d["zone_operated"]    = feat.zone_operated
    d["trip_type"]        = feat.trip_type
    d["faulted_phases"]   = "+".join(feat.faulted_phases) if feat.faulted_phases else ""
    d["fault_type"]       = feat.fault_type
    d["is_ground_fault"]  = feat.is_ground_fault

    # Reclose
    d["reclose_attempted"]  = feat.reclose_attempted
    d["reclose_successful"] = feat.reclose_successful
    d["reclose_time_ms"]    = feat.reclose_time_ms
    d["fault_count"]        = feat.fault_count

    # Fault duration (from fault detector)
    d["fault_duration_ms"]  = fault.duration_ms
    d["fault_inception_ms"] = round(fault.inception_time * 1000, 2)

    # Waveform features
    d["di_dt_max"]              = feat.di_dt_max
    d["di_dt_phase"]            = feat.di_dt_phase
    d["peak_fault_current_a"]   = feat.peak_fault_current_a
    d["peak_fault_phase"]       = feat.peak_fault_phase
    d["i0_i1_ratio"]            = feat.i0_i1_ratio
    d["thd_percent"]            = feat.thd_percent
    d["inception_angle_degrees"]= feat.inception_angle_degrees
    d["voltage_sag_depth_pu"]   = feat.voltage_sag_depth_pu
    d["voltage_sag_phase"]      = feat.voltage_sag_phase

    # Impedance
    d["r_x_ratio"]        = feat.r_x_ratio
    d["z_magnitude_ohms"] = feat.z_magnitude_ohms
    d["z_angle_degrees"]  = feat.z_angle_degrees

    # Metadata
    d["sampling_rate_hz"]   = feat.sampling_rate_hz
    d["record_duration_ms"] = feat.record_duration_ms
    d["teleprotection_rx"]  = feat.teleprotection_received
    d["comms_failure"]      = feat.comms_failure

    # Quality flags — rows failing these should be excluded from training
    # peak < 200A primary means secondary-scaling issue (no CT ratio in file)
    # duration < 5ms means false detection (contact bounce / noise)
    d["scaling_ok"]  = feat.peak_fault_current_a >= 200.0
    d["duration_ok"] = fault.duration_ms >= 5.0

    return d


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_files = list(find_labeled_cfgs(RAW_DATA))
    print(f"Found {len(cfg_files)} labeled CFG files")

    from collections import Counter
    label_counts = Counter(label for _, label in cfg_files)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()

    rows = []
    errors = []

    for i, (cfg_path, label) in enumerate(cfg_files):
        short = str(cfg_path).replace(str(RAW_DATA), "")
        print(f"[{i+1:3d}/{len(cfg_files)}] {label:<12} {short[-70:]}", end="  ")

        try:
            record = parse_comtrade(str(cfg_path))
            if record is None:
                print("SKIP (parse failed)")
                errors.append({"cfg": str(cfg_path), "label": label, "reason": "parse returned None"})
                continue

            prot  = determine_protection(record)
            fault = detect_fault(record)

            if fault is None:
                print("SKIP (no fault detected)")
                errors.append({"cfg": str(cfg_path), "label": label, "reason": "no fault detected"})
                continue

            if prot.primary_protection.name != "DISTANCE":
                print(f"SKIP (prot={prot.primary_protection.name})")
                errors.append({"cfg": str(cfg_path), "label": label,
                                "reason": f"protection={prot.primary_protection.name}"})
                continue

            feat = extract_distance_features(record, fault, prot)
            if feat is None:
                print("SKIP (feature extraction failed)")
                errors.append({"cfg": str(cfg_path), "label": label, "reason": "extract_distance_features returned None"})
                continue

            row = flatten_features(feat, label, cfg_path, prot, fault)
            rows.append(row)
            print(f"OK  z={row['zone_operated']} ph={row['faulted_phases']} "
                  f"dur={row['fault_duration_ms']:.0f}ms "
                  f"ar={row['reclose_successful']}")

        except Exception as e:
            msg = str(e)[:120]
            print(f"ERROR: {msg}")
            errors.append({"cfg": str(cfg_path), "label": label, "reason": msg,
                           "traceback": traceback.format_exc()[-300:]})

    # Write features CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows → {OUT_CSV}")

    # Write errors CSV
    if errors:
        with open(ERR_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cfg", "label", "reason", "traceback"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(errors)
        print(f"Wrote {len(errors)} errors → {ERR_CSV}")

    # Summary by label
    from collections import Counter
    success_labels = Counter(r["label"] for r in rows)
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"{'Label':<15} {'Success':>8} {'Input':>8} {'Rate':>8}")
    print("-" * 45)
    for label in sorted(label_counts):
        s = success_labels.get(label, 0)
        t = label_counts[label]
        print(f"{label:<15} {s:>8} {t:>8} {s/t*100:>7.0f}%")
    print(f"\nTotal: {len(rows)} extracted / {len(cfg_files)} attempted")


if __name__ == "__main__":
    main()
