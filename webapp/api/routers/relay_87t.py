"""Relay 87T (Differential Transformer) — diff/restraint plot + operated status."""

import sys
import asyncio
from pathlib import Path
from functools import partial

import numpy as np
from fastapi import APIRouter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import DiffRestraintRequest, DiffRestraintResponse, DiffRestraintSample
from .relay_87l import _compute_diff_restraint, _characteristic_threshold

router = APIRouter(prefix="/api/analyze/87t", tags=["relay-87t"])

# 87T uses HV and LV current pairs for each phase
HV_CHANNELS = {
    "L1": ["IA_HV", "IHA", "IA1"],
    "L2": ["IB_HV", "IHB", "IB1"],
    "L3": ["IC_HV", "IHC", "IC1"],
}
LV_CHANNELS = {
    "L1": ["IA_LV", "ILA", "IA2"],
    "L2": ["IB_LV", "ILB", "IB2"],
    "L3": ["IC_LV", "ILC", "IC2"],
}


def _find_ch(channels, candidates):
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"])
    return None


def _compute_87t(comtrade_data: dict, params: dict) -> dict:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])

    idiff_pickup = params["idiff_pickup"]
    idiff_fast = params["idiff_fast"]
    slope1 = params["slope1"]
    intersection1 = params["intersection1"]
    slope2 = params["slope2"]
    intersection2 = params["intersection2"]

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    freq = comtrade_data.get("frequency", 50.0)
    win = max(1, int(sr / freq))

    samples = []
    operated_phases = []

    for phase in ["L1", "L2", "L3"]:
        i_hv = _find_ch(channels, HV_CHANNELS[phase])
        i_lv = _find_ch(channels, LV_CHANNELS[phase])

        # Fallback: if no explicit HV/LV split, use generic phase channels
        if i_hv is None:
            from .relay_87l import PHASE_CURRENT_MAP
            i_hv = _find_ch(channels, PHASE_CURRENT_MAP[phase])
        if i_lv is None:
            i_lv = np.zeros(len(time))

        if i_hv is None:
            continue

        phase_samples = []
        for k in range(0, len(time), max(1, win // 4)):
            s = max(0, k - win + 1)
            hv_w = i_hv[s:k+1]
            lv_w = i_lv[s:k+1]
            hv_rms = float(np.sqrt(np.mean(hv_w**2))) if len(hv_w) > 0 else 0.0
            lv_rms = float(np.sqrt(np.mean(lv_w**2))) if len(lv_w) > 0 else 0.0

            i_diff = abs(hv_rms - lv_rms)
            i_rest = (abs(hv_rms) + abs(lv_rms)) / 2.0
            phase_samples.append({
                "t": float(time[k]),
                "i_diff": i_diff,
                "i_rest": i_rest,
                "phase": phase,
            })

        samples.extend(phase_samples)

        phase_operated = any(
            s["i_diff"] >= _characteristic_threshold(
                s["i_rest"], idiff_pickup, slope1, intersection1, slope2, intersection2
            )
            for s in phase_samples
        )
        if phase_operated:
            operated_phases.append(phase)

    if any(s["i_diff"] >= idiff_fast for s in samples):
        status = "IDIFF_FAST_OPERATED"
    elif operated_phases:
        status = "IDIFF_OPERATED"
    else:
        status = "NOT_OPERATED"

    return {"samples": samples, "operated_status": status, "operated_phases": operated_phases}


@router.post("/diff-restraint", response_model=DiffRestraintResponse)
async def diff_restraint_87t(body: DiffRestraintRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_compute_87t, body.comtrade.model_dump(), body.params.model_dump()),
    )
    return DiffRestraintResponse(
        samples=[DiffRestraintSample(**s) for s in result["samples"]],
        params=body.params,
        operated_status=result["operated_status"],
        operated_phases=result["operated_phases"],
    )
