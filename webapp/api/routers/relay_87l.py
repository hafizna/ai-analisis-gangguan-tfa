"""Relay 87L (Differential Line) — diff/restraint plot + AI analysis."""

import sys
import asyncio
from pathlib import Path
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    DiffRestraintRequest, DiffRestraintResponse, DiffRestraintSample,
    CharacteristicParams, AIFaultResult,
)

router = APIRouter(prefix="/api/analyze/87l", tags=["relay-87l"])

PHASE_CURRENT_MAP = {
    "L1": ["IA", "IL1", "I1"],
    "L2": ["IB", "IL2", "I2"],
    "L3": ["IC", "IL3", "I3"],
}


def _find_ch(channels, candidates):
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"])
    return None


def _compute_diff_restraint(comtrade_data: dict, params: dict) -> dict:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    samples = []
    operated_phases = []

    idiff_pickup = params["idiff_pickup"]
    idiff_fast = params["idiff_fast"]
    slope1 = params["slope1"]
    intersection1 = params["intersection1"]
    slope2 = params["slope2"]
    intersection2 = params["intersection2"]

    for phase, candidates in PHASE_CURRENT_MAP.items():
        # For 87L, diff = local I - remote I. Since we only have one set of CTs,
        # compute a proxy: i_diff = |i_local| - |i_remote| using half the channels.
        # Real implementation would use both ends; here we split the total measured current.
        i = _find_ch(channels, candidates)
        if i is None:
            continue

        # Proxy differential calculation for single-ended COMTRADE:
        # I_diff ≈ |I| (single-ended — requires both-end data for true 87L)
        # I_rest ≈ |I| (restraint)
        i_rms_window = []
        sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
        freq = comtrade_data.get("frequency", 50.0)
        win = max(1, int(sr / freq))

        for k in range(0, len(time), max(1, win // 4)):
            s = max(0, k - win + 1)
            i_w = i[s:k+1]
            i_rms = float(np.sqrt(np.mean(i_w**2))) if len(i_w) > 0 else 0.0
            i_diff = abs(i_rms)       # proxy: use RMS as differential magnitude
            i_rest = abs(i_rms)       # proxy: use RMS as restraint
            samples.append({
                "t": float(time[k]),
                "i_diff": i_diff,
                "i_rest": i_rest,
                "phase": phase,
            })

        # Check if any sample crossed the operate region
        phase_operated = False
        for s in samples:
            if s["phase"] != phase:
                continue
            id_, ir = s["i_diff"], s["i_rest"]
            char = _characteristic_threshold(ir, idiff_pickup, slope1, intersection1, slope2, intersection2)
            if id_ >= char:
                phase_operated = True
                break
        if phase_operated:
            operated_phases.append(phase)

    # Determine operated status
    if any(
        s["i_diff"] >= idiff_fast
        for s in samples
    ):
        status = "IDIFF_FAST_OPERATED"
    elif operated_phases:
        status = "IDIFF_OPERATED"
    else:
        status = "NOT_OPERATED"

    return {"samples": samples, "operated_status": status, "operated_phases": operated_phases}


def _characteristic_threshold(i_rest, pickup, slope1, int1, slope2, int2):
    """Return the I-DIFF operate threshold for a given I_rest value."""
    if i_rest <= int1:
        return pickup
    elif i_rest <= int2:
        return pickup + slope1 * (i_rest - int1)
    else:
        return pickup + slope1 * (int2 - int1) + slope2 * (i_rest - int2)


@router.post("/diff-restraint", response_model=DiffRestraintResponse)
async def diff_restraint(body: DiffRestraintRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_compute_diff_restraint, body.comtrade.model_dump(), body.params.model_dump()),
    )
    return DiffRestraintResponse(
        samples=[DiffRestraintSample(**s) for s in result["samples"]],
        params=body.params,
        operated_status=result["operated_status"],
        operated_phases=result["operated_phases"],
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis_87l(body: DiffRestraintRequest):
    """AI fault analysis for 87L — internal vs external fault discrimination."""
    result = _compute_diff_restraint(body.comtrade.model_dump(), body.params.model_dump())
    status = result["operated_status"]
    samples = result["samples"]

    evidence = []
    scores = {
        "INTERNAL_FAULT": 0.0,
        "EXTERNAL_FAULT": 0.0,
        "INRUSH": 0.0,
        "INSULATION_BREAKDOWN": 0.0,
        "CONDUCTOR_CONTACT": 0.0,
    }

    if status in ("IDIFF_OPERATED", "IDIFF_FAST_OPERATED"):
        scores["INTERNAL_FAULT"] += 0.5
        scores["INSULATION_BREAKDOWN"] += 0.3
        evidence.append(f"Differential relay operated ({status}) — internal fault signature")
        if status == "IDIFF_FAST_OPERATED":
            scores["INTERNAL_FAULT"] += 0.3
            evidence.append("I-DIFF Fast threshold exceeded — high-magnitude internal fault")
    else:
        scores["EXTERNAL_FAULT"] += 0.5
        evidence.append("Differential element did not operate — consistent with external/through-fault")

    max_diff = max((s["i_diff"] for s in samples), default=0.0)
    if max_diff > body.params.idiff_fast * 0.7:
        scores["CONDUCTOR_CONTACT"] += 0.2
        evidence.append(f"High differential current peak ({max_diff:.2f}) — possible direct conductor contact")

    total = sum(scores.values()) or 1.0
    labels = {
        "INTERNAL_FAULT": "Internal Fault",
        "EXTERNAL_FAULT": "External / Through-Fault",
        "INRUSH": "Inrush / Overexcitation",
        "INSULATION_BREAKDOWN": "Insulation Breakdown",
        "CONDUCTOR_CONTACT": "Conductor Contact / Short",
    }
    ranking = sorted(
        [{"cause": k, "label_id": k, "label": labels[k], "confidence": round(v / total, 3)} for k, v in scores.items()],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    fault_type = "permanent" if status != "NOT_OPERATED" else "transient"

    return AIFaultResult(
        cause_ranking=ranking,
        fault_type=fault_type,
        overall_confidence=ranking[0]["confidence"] if ranking else 0.0,
        evidence=evidence,
    )
