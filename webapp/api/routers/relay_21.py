"""Relay 21 (Distance) — impedance locus + AI fault analysis."""

import sys
import asyncio
from pathlib import Path
from functools import partial
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    LocusAnalysisRequest, LocusResponse, LocusPoint,
    AIFaultFeatures, AIFaultResult,
)
from ..storage import load_analysis

router = APIRouter(prefix="/api/analyze/21", tags=["relay-21"])

# Phase-to-channel mappings for each loop
LOOP_CHANNELS = {
    "ZA":  {"v": ["VA", "VAN", "UA"], "i": ["IA"]},
    "ZB":  {"v": ["VB", "VBN", "UB"], "i": ["IB"]},
    "ZC":  {"v": ["VC", "VCN", "UC"], "i": ["IC"]},
    "ZAB": {"v": ["VAB", "UAB"], "i": ["IA", "IB"], "diff": True},
    "ZBC": {"v": ["VBC", "UBC"], "i": ["IB", "IC"], "diff": True},
    "ZCA": {"v": ["VCA", "UCA"], "i": ["IC", "IA"], "diff": True},
}


def _find_channel(channels, candidates: list[str]) -> Optional[np.ndarray]:
    """Return samples for the first matching canonical name."""
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"])
    return None


def _compute_locus(comtrade_data: dict, loop: str) -> list[dict]:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    mapping = LOOP_CHANNELS.get(loop, LOOP_CHANNELS["ZA"])

    v = _find_channel(channels, mapping["v"])
    if v is None:
        raise HTTPException(status_code=422, detail=f"Could not find voltage channel for loop {loop}")

    i_channels = [_find_channel(channels, [c]) for c in mapping["i"]]
    i_channels = [c for c in i_channels if c is not None]
    if not i_channels:
        raise HTTPException(status_code=422, detail=f"Could not find current channel(s) for loop {loop}")

    if mapping.get("diff") and len(i_channels) == 2:
        i = i_channels[0] - i_channels[1]
    else:
        i = i_channels[0]

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(i) > 0.01, v / i, np.nan + 1j * np.nan)

    r = np.real(z) if np.iscomplexobj(z) else np.zeros_like(v)
    x = np.imag(z) if np.iscomplexobj(z) else np.zeros_like(v)

    # If signals are real (not complex analytic), compute instantaneous R/X via DFT window
    if not np.iscomplexobj(z):
        freq = comtrade_data.get("frequency", 50.0)
        sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
        win = max(1, int(sr / freq))  # one cycle window
        r_list, x_list = [], []
        for k in range(len(time)):
            s = max(0, k - win + 1)
            v_w = v[s:k+1]
            i_w = i[s:k+1]
            if len(v_w) < 2 or np.max(np.abs(i_w)) < 0.01:
                r_list.append(float("nan"))
                x_list.append(float("nan"))
            else:
                # Least-squares: V = R*I + X*I_90  (I_90 = Hilbert approx)
                i_90 = np.gradient(i_w) / (2 * np.pi * freq / sr)
                A = np.column_stack([i_w, i_90])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, v_w, rcond=None)
                    r_list.append(float(coeffs[0]))
                    x_list.append(float(coeffs[1]))
                except Exception:
                    r_list.append(float("nan"))
                    x_list.append(float("nan"))
        r = np.array(r_list)
        x = np.array(x_list)

    points = []
    for k in range(len(time)):
        rv, xv = float(r[k]), float(x[k])
        if np.isnan(rv) or np.isnan(xv):
            continue
        if abs(rv) > 500 or abs(xv) > 500:
            continue
        points.append({"t": float(time[k]), "r": rv, "x": xv})

    return points


@router.post("/locus", response_model=LocusResponse)
async def compute_locus(body: LocusAnalysisRequest):
    """Compute impedance locus (R-X trajectory) for the selected loop."""
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    loop = asyncio.get_event_loop()
    points = await loop.run_in_executor(
        None, partial(_compute_locus, payload, body.loop)
    )
    return LocusResponse(
        loop=body.loop,
        points=[LocusPoint(**p) for p in points],
        zones=body.zones,
        fault_inception_idx=None,
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis(features: AIFaultFeatures):
    """Run AI fault cause analysis for relay 21 (distance protection)."""
    # Feature-based heuristic rules (ML model integration point)
    scores: dict[str, float] = {
        "PETIR": 0.0,
        "LAYANG": 0.0,
        "POHON": 0.0,
        "HEWAN": 0.0,
        "BENDA_ASING": 0.0,
        "KONDUKTOR": 0.0,
    }
    evidence: list[str] = []

    fia = features.fault_inception_angle_deg
    dur = features.fault_duration_ms
    asym = features.waveform_asymmetry
    dc = features.dc_offset

    # High DC offset + fault near voltage zero-crossing → lightning signature
    if abs(dc) > 0.3 and (abs(fia) < 30 or abs(fia) > 150):
        scores["PETIR"] += 0.4
        evidence.append(f"High DC offset ({dc:.2f}) with fault near voltage zero-crossing (FIA={fia:.1f}°) — lightning signature")

    # Very short fault duration → likely transient (lightning, kite)
    if dur < 100:
        scores["PETIR"] += 0.2
        scores["LAYANG"] += 0.15
        evidence.append(f"Short fault duration ({dur:.0f} ms) — consistent with transient cause")

    # High waveform asymmetry → conductor damage or tree
    if asym > 0.5:
        scores["POHON"] += 0.25
        scores["KONDUKTOR"] += 0.2
        evidence.append(f"High waveform asymmetry ({asym:.2f}) — possible conductor/vegetation contact")

    # Long fault → permanent (tree, conductor)
    if dur > 500:
        scores["POHON"] += 0.3
        scores["KONDUKTOR"] += 0.25
        fault_type = "permanent"
        evidence.append(f"Long fault duration ({dur:.0f} ms) — permanent fault indicator")
    else:
        fault_type = "transient"

    # AR result
    if features.ar_result == "successful":
        scores["PETIR"] += 0.15
        scores["LAYANG"] += 0.15
        evidence.append("Successful auto-reclose — supports transient fault classification")
    elif features.ar_result == "failed":
        scores["POHON"] += 0.2
        scores["KONDUKTOR"] += 0.2
        fault_type = "permanent"
        evidence.append("Failed auto-reclose — supports permanent fault classification")

    # Normalise
    total = sum(scores.values()) or 1.0
    ranking = sorted(
        [
            {
                "cause": k,
                "label_id": k,
                "label": {
                    "PETIR": "Petir / Lightning",
                    "LAYANG": "Layang-Layang / Kite",
                    "POHON": "Pohon / Vegetasi",
                    "HEWAN": "Hewan / Binatang",
                    "BENDA_ASING": "Benda Asing",
                    "KONDUKTOR": "Konduktor / Tower",
                }.get(k, k),
                "confidence": round(v / total, 3),
            }
            for k, v in scores.items()
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    overall_confidence = ranking[0]["confidence"] if ranking else 0.0

    return AIFaultResult(
        cause_ranking=ranking,
        fault_type=fault_type,
        overall_confidence=overall_confidence,
        evidence=evidence,
    )
