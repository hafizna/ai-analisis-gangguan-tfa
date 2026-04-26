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
from ..ml_predict import run_ml_prediction, extract_ml_features

router = APIRouter(prefix="/api/analyze/21", tags=["relay-21"])

# Phase-to-channel mappings for each loop
LOOP_CHANNELS = {
    "ZA":  {"v": ["VA", "VAN", "UA"], "i": ["IA"], "phase": "A"},
    "ZB":  {"v": ["VB", "VBN", "UB"], "i": ["IB"], "phase": "B"},
    "ZC":  {"v": ["VC", "VCN", "UC"], "i": ["IC"], "phase": "C"},
    # Phase-to-phase loops: current diff is (IB−IA), (IC−IB), (IA−IC) so that
    # the convention Z = V_AB / (IB−IA) matches IEC positive-direction relay setting
    # (forward fault → positive R/X).  Swapping gives -(VA−VB)/(IA−IB) = same sign.
    "ZAB": {"v": ["VAB", "UAB"], "i": ["IB", "IA"], "diff": True, "phases": ("A", "B")},
    "ZBC": {"v": ["VBC", "UBC"], "i": ["IC", "IB"], "diff": True, "phases": ("B", "C")},
    "ZCA": {"v": ["VCA", "UCA"], "i": ["IA", "IC"], "diff": True, "phases": ("C", "A")},
}


def _get_secondary_scale(channels: list) -> float:
    """
    Return scale factor to convert primary Ω → secondary Ω: ct_ratio / vt_ratio.
    Relay zone settings (.rio with IMPPRIM NO) are in secondary Ω, so the locus
    must be in the same unit for meaningful visual comparison.
    """
    vt_ratio = 1.0
    ct_ratio = 1.0
    for ch in channels:
        pri = float(ch.get("ct_primary") or 1)
        sec = float(ch.get("ct_secondary") or 1)
        if sec <= 0:
            sec = 1.0
        if ch.get("measurement") == "voltage" and pri > 1 and vt_ratio == 1.0:
            vt_ratio = pri / sec
        elif ch.get("measurement") == "current" and pri > 1 and ct_ratio == 1.0:
            ct_ratio = pri / sec
    return (ct_ratio / vt_ratio) if vt_ratio > 0 else 1.0


def _find_channel(channels, candidates: list[str]) -> Optional[np.ndarray]:
    """Return samples for the first matching canonical name."""
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"])
    return None


def _find_phase_voltage(channels, phase: str) -> Optional[np.ndarray]:
    for ch in channels:
        if ch.get("measurement") != "voltage":
            continue
        if (ch.get("phase") or "").upper() == phase:
            return np.array(ch["samples"], dtype=float)
    return None


def _find_phase_current(channels, phase: str) -> Optional[np.ndarray]:
    for ch in channels:
        if ch.get("measurement") != "current":
            continue
        if (ch.get("phase") or "").upper() == phase:
            return np.array(ch["samples"], dtype=float)
    return None


def _find_voltage_for_loop(channels, mapping: dict) -> Optional[np.ndarray]:
    direct = _find_channel(channels, mapping["v"])
    if direct is not None:
        return direct.astype(float)

    phase = mapping.get("phase")
    if phase:
        return _find_phase_voltage(channels, phase)

    phases = mapping.get("phases")
    if phases:
        left = _find_phase_voltage(channels, phases[0])
        right = _find_phase_voltage(channels, phases[1])
        if left is not None and right is not None:
            return left - right

    return None


def _compute_locus(comtrade_data: dict, loop: str, k0: float = 0.0, k0_angle_deg: float = 0.0, invert_i: bool = False) -> list[dict]:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    mapping = LOOP_CHANNELS.get(loop, LOOP_CHANNELS["ZA"])
    # Scale to convert primary Ω → secondary Ω so locus aligns with .rio zone polygons
    secondary_scale = _get_secondary_scale(channels)

    v = _find_voltage_for_loop(channels, mapping)
    if v is None:
        raise HTTPException(status_code=422, detail=f"Could not find voltage channel for loop {loop}")

    i_channels = []
    for candidate in mapping["i"]:
        current = _find_channel(channels, [candidate])
        if current is None and candidate.startswith("I") and len(candidate) >= 2:
            current = _find_phase_current(channels, candidate[-1])
        i_channels.append(current)
    i_channels = [c for c in i_channels if c is not None]
    if not i_channels:
        raise HTTPException(status_code=422, detail=f"Could not find current channel(s) for loop {loop}")

    if mapping.get("diff") and len(i_channels) == 2:
        i = i_channels[0].astype(float) - i_channels[1].astype(float)
    else:
        i = i_channels[0].astype(float)

    # Current polarity correction: some DFRs (e.g. Sifang) store I in line-flow
    # direction (positive = away from bus), opposite to the relay convention
    # (positive = toward fault). Negating aligns the locus to the correct quadrant.
    if invert_i:
        i = -i

    freq = comtrade_data.get("frequency", 50.0)
    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    win = max(4, int(sr / freq))  # one cycle window

    # Precompute reference sinusoids for the full window (reused every iteration)
    t_win = np.arange(win) / sr
    cos_win = np.cos(2.0 * np.pi * freq * t_win)
    sin_win = np.sin(2.0 * np.pi * freq * t_win)

    # Prepare k0 residual current for ground loops (complex KZN in frequency domain)
    k0_complex: complex | None = None
    i_res: np.ndarray | None = None
    if loop in ("ZA", "ZB", "ZC") and k0 != 0.0:
        ia = _find_phase_current(channels, "A")
        ib = _find_phase_current(channels, "B")
        ic = _find_phase_current(channels, "C")
        if ia is not None and ib is not None and ic is not None:
            i_res = (ia + ib + ic).astype(float)
            if invert_i:
                i_res = -i_res
            k0_complex = k0 * np.exp(1j * np.radians(k0_angle_deg))

    # Sliding DFT phasor division: Z = V_phasor / I_phasor
    # Compared to the gradient-based LS fit, DFT naturally rejects DC offset
    # (asymmetric fault inception) and converges faster when fault current dominates
    # the window — the large fault I overwhelms any residual pre-fault samples.
    v = v.astype(float)
    scale = 1000.0 * secondary_scale
    r_list, x_list = [], []

    for k in range(len(time)):
        s = max(0, k - win + 1)
        v_w = v[s : k + 1]
        i_w = i[s : k + 1]
        n = len(v_w)

        if n < 4 or np.max(np.abs(i_w)) < 0.01:
            r_list.append(float("nan"))
            x_list.append(float("nan"))
            continue

        # DFT projection: phasor = (2/n) × (Σ x·cos − j·Σ x·sin)
        c_ref = cos_win[:n]
        s_ref = sin_win[:n]
        v_re = (2.0 / n) * float(v_w @ c_ref)
        v_im = -(2.0 / n) * float(v_w @ s_ref)
        i_re = (2.0 / n) * float(i_w @ c_ref)
        i_im = -(2.0 / n) * float(i_w @ s_ref)

        V_ph = complex(v_re, v_im)
        I_ph = complex(i_re, i_im)

        # k0 compensation: I_loop = I_phase + KZN × I_residual (frequency domain)
        if k0_complex is not None and i_res is not None:
            ires_w = i_res[s : k + 1]
            ir_re = (2.0 / n) * float(ires_w @ c_ref)
            ir_im = -(2.0 / n) * float(ires_w @ s_ref)
            I_ph = I_ph + k0_complex * complex(ir_re, ir_im)

        I_mag_sq = I_ph.real ** 2 + I_ph.imag ** 2
        if I_mag_sq < 1e-6:
            r_list.append(float("nan"))
            x_list.append(float("nan"))
            continue

        Z = V_ph / I_ph  # phasor division: Z = R + jX (secondary Ω after scaling)
        r_list.append(Z.real * scale)
        x_list.append(Z.imag * scale)

    r = np.array(r_list)
    x = np.array(x_list)
    current_mag = np.abs(i)
    ok = np.isfinite(r) & np.isfinite(x) & (np.abs(r) <= 500) & (np.abs(x) <= 500)
    ok &= np.isfinite(current_mag) & (current_mag >= 0.01)

    if len(ok) >= 3:
        isolated = ok.copy()
        isolated[1:-1] = ok[1:-1] & ~ok[:-2] & ~ok[2:]
        isolated[0] = ok[0] and not ok[1]
        isolated[-1] = ok[-1] and not ok[-2]
        ok &= ~isolated

    points = []
    for k in range(len(time)):
        if not ok[k]:
            continue
        rv, xv = float(r[k]), float(x[k])
        if np.isnan(rv) or np.isnan(xv):
            continue
        if abs(rv) > 500 or abs(xv) > 500:
            continue
        points.append({"t": float(time[k]), "r": rv, "x": xv})

    return points


def _extract_features_from_payload(payload: dict) -> dict:
    """Auto-extract fault analysis features from a stored COMTRADE payload."""
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    freq = float(payload.get("frequency", 50.0))

    empty = {
        "fault_inception_angle_deg": 0.0,
        "fault_duration_ms": 0.0,
        "prefault_load_a": 0.0,
        "impedance_at_trip_ohm": 0.0,
        "waveform_asymmetry": 0.0,
        "dc_offset": 0.0,
        "ar_result": None,
    }

    if len(time) < 4:
        return empty

    sr = 1.0 / (time[1] - time[0])
    cycle_n = max(4, int(sr / freq))

    # Pick first available phase current and voltage
    i = _find_channel(channels, ["IA", "IL1", "I1", "IB", "IL2", "IC", "IL3"])
    v = _find_channel(channels, ["VA", "VAN", "UA", "VB", "VBN", "VC", "VCN"])
    if i is None:
        return empty

    # Pre-fault RMS (first 2 cycles or first quarter of record)
    pre_end = min(2 * cycle_n, len(i) // 4)
    pre_rms = float(np.sqrt(np.mean(i[:pre_end] ** 2))) if pre_end > 1 else 0.0

    # Fault inception: first sample exceeding 2× pre-fault RMS
    threshold = max(pre_rms * 2.0, np.max(np.abs(i)) * 0.3, 0.05)
    inception_idx = next(
        (k for k in range(pre_end, len(i)) if abs(i[k]) > threshold),
        int(np.argmax(np.abs(i))),
    )

    # Fault extinction: RMS drops back below threshold
    extinction_idx = len(i) - 1
    for k in range(inception_idx + cycle_n, len(i)):
        s = max(0, k - cycle_n // 2)
        if float(np.sqrt(np.mean(i[s : k + 1] ** 2))) < threshold * 0.6:
            extinction_idx = k
            break
    fault_duration_ms = float((time[extinction_idx] - time[inception_idx]) * 1000)

    # FIA: sine of normalised voltage at inception → degrees
    fia_deg = 0.0
    if v is not None and inception_idx < len(v):
        v_peak = float(np.max(np.abs(v[:inception_idx]))) if inception_idx > 0 else float(np.max(np.abs(v)))
        if v_peak > 0:
            ratio = float(np.clip(v[inception_idx] / v_peak, -1.0, 1.0))
            fia_deg = float(np.degrees(np.arcsin(ratio)))

    # DC offset and asymmetry from first fault cycle
    fw = i[inception_idx : inception_idx + cycle_n]
    dc_offset, asymmetry = 0.0, 0.0
    if len(fw) > 4:
        dc_component = float(np.mean(fw))
        ac_amp = float(np.sqrt(2) * np.sqrt(np.mean(fw ** 2)))
        dc_offset = float(np.clip(dc_component / ac_amp, -1.0, 1.0)) if ac_amp > 0 else 0.0
        pos, neg = float(np.max(fw)), float(np.min(fw))
        denom = pos + abs(neg)
        asymmetry = abs(pos - abs(neg)) / denom if denom > 0 else 0.0

    # |Z| at inception — V in kV, I in A → *1000 → Ω
    impedance_ohm = 0.0
    if v is not None and inception_idx < len(v) and abs(i[inception_idx]) > 0.01:
        impedance_ohm = float(abs(v[inception_idx] * 1000.0 / i[inception_idx]))

    # AR result from binary channels
    ar_result = None
    for sch in payload.get("status_channels", []):
        name = sch.get("name", "").upper()
        if any(k in name for k in ("AR", "RECLOSE", "RECLUSE", "RECLOS")):
            samp = sch.get("samples", [])
            if 1 in samp:
                ar_result = "successful"
            break

    return {
        "fault_inception_angle_deg": round(fia_deg, 1),
        "fault_duration_ms": round(max(fault_duration_ms, 0.0), 1),
        "prefault_load_a": round(pre_rms, 2),
        "impedance_at_trip_ohm": round(impedance_ohm, 3),
        "waveform_asymmetry": round(asymmetry, 3),
        "dc_offset": round(dc_offset, 3),
        "ar_result": ar_result,
    }


def _compute_electrical_params(payload: dict) -> dict:
    """Compute extended electrical parameters for the workspace panel."""
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    freq = float(payload.get("frequency", 50.0))

    def find_phase_ch(phases: list[str], meas: str):
        for p in phases:
            for ch in channels:
                if ch.get("measurement") != meas:
                    continue
                ph = (ch.get("phase") or "").upper()
                cn = ch.get("canonical_name", "").upper()
                nm = ch.get("name", "").upper()
                if ph == p or cn.endswith(p) or nm.endswith(p):
                    return np.array(ch["samples"], dtype=float)
        return None

    ia = find_phase_ch(["A", "L1", "1"], "current")
    ib = find_phase_ch(["B", "L2", "2"], "current")
    ic = find_phase_ch(["C", "L3", "3"], "current")
    va = find_phase_ch(["A", "L1", "1"], "voltage")
    vb = find_phase_ch(["B", "L2", "2"], "voltage")
    vc = find_phase_ch(["C", "L3", "3"], "voltage")

    result: dict = {}

    if len(time) < 4:
        return result

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    cycle_n = max(4, int(sr / freq))

    # Use IA to find inception index
    i_ref = ia if ia is not None else (ib if ib is not None else ic)
    pre_end = min(2 * cycle_n, len(i_ref) // 4) if i_ref is not None else 0
    inception_idx = 0
    extinction_idx = len(time) - 1

    if i_ref is not None and pre_end > 1:
        pre_rms = float(np.sqrt(np.mean(i_ref[:pre_end] ** 2)))
        thr = max(pre_rms * 2.0, np.max(np.abs(i_ref)) * 0.3, 0.05)
        inception_idx = next(
            (k for k in range(pre_end, len(i_ref)) if abs(i_ref[k]) > thr),
            int(np.argmax(np.abs(i_ref))),
        )
        for k in range(inception_idx + cycle_n, len(i_ref)):
            s = max(0, k - cycle_n // 2)
            if float(np.sqrt(np.mean(i_ref[s : k + 1] ** 2))) < thr * 0.6:
                extinction_idx = k
                break

    fault_slice = slice(inception_idx, min(extinction_idx + 1, len(time)))

    # I_peak per phase (kA or A)
    for label, arr in [("IA", ia), ("IB", ib), ("IC", ic)]:
        if arr is not None and len(arr) > inception_idx:
            result[f"i_peak_{label.lower()}_a"] = round(float(np.max(np.abs(arr[fault_slice]))), 2)

    # V_sag: prefault RMS vs fault-window RMS (%)
    if va is not None and pre_end > 1:
        v_pre_rms = float(np.sqrt(np.mean(va[:pre_end] ** 2)))
        v_fault_rms = float(np.sqrt(np.mean(va[fault_slice] ** 2))) if len(va) > inception_idx else v_pre_rms
        if v_pre_rms > 0:
            result["v_sag_pct"] = round((1.0 - v_fault_rms / v_pre_rms) * 100, 1)

    # Symmetrical components (Fortescue) at inception using one-cycle window
    a_op = np.exp(1j * 2 * np.pi / 3)
    a2_op = np.exp(-1j * 2 * np.pi / 3)

    def rms_phasor(arr, idx, n):
        """Estimate fundamental phasor at index idx using DFT over n samples."""
        if arr is None or idx + n > len(arr):
            return None
        seg = arr[idx : idx + n]
        t_seg = np.arange(n) / sr
        cos_ref = np.cos(2 * np.pi * freq * t_seg)
        sin_ref = np.sin(2 * np.pi * freq * t_seg)
        re = 2 * np.mean(seg * cos_ref)
        im = -2 * np.mean(seg * sin_ref)
        return complex(re, im) / np.sqrt(2)  # RMS phasor

    pI_a = rms_phasor(ia, inception_idx, cycle_n)
    pI_b = rms_phasor(ib, inception_idx, cycle_n)
    pI_c = rms_phasor(ic, inception_idx, cycle_n)

    if pI_a is not None and pI_b is not None and pI_c is not None:
        I0 = (pI_a + pI_b + pI_c) / 3
        I1 = (pI_a + a_op * pI_b + a2_op * pI_c) / 3
        I2 = (pI_a + a2_op * pI_b + a_op * pI_c) / 3
        result["i_pos_seq_a"] = round(abs(I1), 2)
        result["i_neg_seq_a"] = round(abs(I2), 2)
        result["i_zero_seq_a"] = round(abs(I0), 2)

    # Z at inception and R/X from least-squares (V in kV, I in A → multiply by 1000 → Ω)
    if va is not None and ia is not None:
        win = min(cycle_n, len(va) - inception_idx)
        if win >= 4:
            v_w = va[inception_idx : inception_idx + win] * 1000.0  # kV → V
            i_w = ia[inception_idx : inception_idx + win]
            i_90 = np.gradient(i_w) / (2 * np.pi * freq / sr)
            A = np.column_stack([i_w, i_90])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, v_w, rcond=None)
                r_val = float(coeffs[0])
                x_val = float(coeffs[1])
                result["r_at_fault_ohm"] = round(r_val, 2)
                result["x_at_fault_ohm"] = round(x_val, 2)
                if x_val != 0:
                    result["rx_ratio"] = round(r_val / x_val, 3)
                z_mag = float(np.sqrt(r_val ** 2 + x_val ** 2))
                result["z_at_inception_ohm"] = round(z_mag, 2)
                if z_mag > 0:
                    result["z_angle_deg"] = round(float(np.degrees(np.arctan2(x_val, r_val))), 1)
            except Exception:
                pass

    # AR dead time: time between extinction and next high current (reclose)
    ar_dead_ms = None
    for sch in payload.get("status_channels", []):
        name = sch.get("name", "").upper()
        if any(k in name for k in ("AR", "RECLOSE", "RECLOS")):
            samp = sch.get("samples", [])
            close_idx = next((k for k in range(extinction_idx, len(samp)) if samp[k] == 1), None)
            if close_idx is not None and close_idx < len(time):
                ar_dead_ms = round((time[close_idx] - time[extinction_idx]) * 1000, 1)
            break
    if ar_dead_ms is not None:
        result["ar_dead_time_ms"] = ar_dead_ms

    result["fault_duration_ms"] = round((time[extinction_idx] - time[inception_idx]) * 1000, 1)
    result["inception_time_ms"] = round(float(time[inception_idx]) * 1000, 1)

    return result


def _compute_fault_classification(payload: dict) -> dict:
    """Derive fault type code, phases, zone, trip and timing for the Jenis Gangguan panel."""
    time = np.array(payload.get("time", []))
    empty = {
        "fault_code": "Unknown", "phases": [], "phases_label": "-",
        "to_ground": False, "trip_type": None, "zone": None,
        "prefault_ms": 0.0, "fault_ms": 0.0, "total_ms": 0.0, "ar_status": None,
    }
    if len(time) < 4:
        return empty

    row = extract_ml_features(payload, "21")

    total_ms = round(float((time[-1] - time[0]) * 1000), 1)
    fault_ms = float(row.get("fault_duration_ms", 0))
    prefault_ms = max(0.0, round(total_ms - fault_ms, 1))

    phases_str = row.get("faulted_phases", "") or ""
    phases = [p for p in phases_str.split("+") if p]
    to_ground = bool(row.get("is_ground_fault", False))
    n_phases = len(phases)

    if n_phases >= 3:
        fault_code = "3Ph"
    elif n_phases == 2 and to_ground:
        fault_code = "DLG"
    elif n_phases == 2:
        fault_code = "LL"
    elif n_phases == 1 and to_ground:
        fault_code = "SLG"
    else:
        fault_code = "SL" if n_phases == 1 else "Unknown"

    trip_type = row.get("trip_type") or None
    if trip_type == "unknown":
        trip_type = None
    zone = row.get("zone_operated") or None

    ar_ok = row.get("reclose_successful")
    ar_status = "successful" if ar_ok is True else ("failed" if ar_ok is False else None)

    phases_label = "+".join(phases) + ("-N" if to_ground and n_phases < 3 else "")

    return {
        "fault_code": fault_code,
        "phases": phases,
        "phases_label": phases_label if phases_label else "-",
        "to_ground": to_ground,
        "trip_type": trip_type,
        "zone": zone,
        "prefault_ms": prefault_ms,
        "fault_ms": fault_ms,
        "total_ms": total_ms,
        "ar_status": ar_status,
    }


@router.get("/fault-classification")
async def fault_classification(analysis_id: str):
    """Classify fault type, phases, zone and trip for the Jenis Gangguan panel."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_fault_classification, payload)
    return result


@router.get("/electrical-params")
async def electrical_params(analysis_id: str):
    """Compute extended electrical parameters for the fault analysis panel."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    params = await loop.run_in_executor(None, _compute_electrical_params, payload)
    return params


@router.get("/extract-features")
async def extract_features(analysis_id: str):
    """Auto-extract fault features from stored COMTRADE data."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(None, _extract_features_from_payload, payload)
    return features


@router.post("/locus", response_model=LocusResponse)
async def compute_locus(body: LocusAnalysisRequest):
    """Compute impedance locus (R-X trajectory) for the selected loop."""
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    loop = asyncio.get_event_loop()
    points = await loop.run_in_executor(
        None, partial(_compute_locus, payload, body.loop, body.k0, body.k0_angle_deg, body.invert_i)
    )
    return LocusResponse(
        loop=body.loop,
        points=[LocusPoint(**p) for p in points],
        zones=body.zones,
        fault_inception_idx=None,
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis(features: AIFaultFeatures):
    """Run LightGBM fault cause analysis for relay 21 (distance protection)."""
    if not features.analysis_id:
        raise HTTPException(status_code=422, detail="analysis_id is required for AI analysis.")
    payload = load_analysis(features.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ml_prediction, payload, "21")
    return AIFaultResult(**result)
