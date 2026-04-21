"""Upload router - parses .cfg + .dat pair and returns structured JSON."""

import asyncio
import sys
import tempfile
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.comtrade_parser import ComtradeRecord, parse_comtrade
from ..schemas import AnalysisCreatedResponse, ComtradeOut, RecalcRequest
from ..storage import load_analysis, save_analysis

router = APIRouter(prefix="/api", tags=["upload"])


def _record_to_out(record: ComtradeRecord) -> dict:
    return {
        "station_name": record.station_name,
        "rec_dev_id": record.rec_dev_id,
        "rev_year": record.rev_year,
        "sampling_rates": record.sampling_rates,
        "trigger_time": record.trigger_time,
        "total_samples": record.total_samples,
        "frequency": record.frequency,
        "time": record.time.tolist(),
        "analog_channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "canonical_name": ch.canonical_name,
                "unit": ch.unit,
                "phase": ch.phase,
                "measurement": ch.measurement,
                "ct_primary": ch.ct_primary,
                "ct_secondary": ch.ct_secondary,
                "pors": ch.pors,
                "samples": ch.samples.tolist(),
            }
            for ch in record.analog_channels
        ],
        "status_channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "samples": ch.samples.tolist(),
            }
            for ch in record.status_channels
        ],
        "warnings": record.warnings,
    }


@router.post("/upload", response_model=AnalysisCreatedResponse)
async def upload_comtrade(
    cfg_file: UploadFile = File(...),
    dat_file: UploadFile = File(...),
):
    """Parse a .cfg + .dat COMTRADE pair and create a backend analysis session."""
    cfg_bytes = await cfg_file.read()
    dat_bytes = await dat_file.read()

    with tempfile.TemporaryDirectory(prefix="dfr_upload_") as tmp_dir:
        tmp = Path(tmp_dir)

        cfg_name = Path(cfg_file.filename or "record.cfg").name
        dat_name = Path(dat_file.filename or "record.dat").name

        cfg_path = tmp / cfg_name
        dat_path = tmp / dat_name

        cfg_path.write_bytes(cfg_bytes)
        dat_path.write_bytes(dat_bytes)

        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(
            None,
            partial(parse_comtrade, str(cfg_path), str(dat_path)),
        )

    if record is None:
        raise HTTPException(
            status_code=422,
            detail="Could not parse COMTRADE files. Check that the .cfg and .dat pair is valid.",
        )

    payload = _record_to_out(record)
    analysis_id = save_analysis(payload)
    return AnalysisCreatedResponse(
        analysis_id=analysis_id,
        station_name=payload["station_name"],
        rec_dev_id=payload["rec_dev_id"],
        total_samples=payload["total_samples"],
        analog_channel_count=len(payload["analog_channels"]),
        status_channel_count=len(payload["status_channels"]),
    )


@router.get("/analysis/{analysis_id}", response_model=ComtradeOut)
async def get_analysis(analysis_id: str):
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    return payload


@router.post("/recalculate-ratio")
async def recalculate_ratio(body: RecalcRequest):
    """Apply per-channel CT/VT ratio overrides and return recalculated samples."""
    ratio_map = {r.channel_id: (r.primary, r.secondary) for r in body.ratios}
    updated_channels = []
    for ch in body.comtrade.analog_channels:
        if ch.id in ratio_map:
            pri, sec = ratio_map[ch.id]
            factor = pri / sec if sec != 0 else 1.0
            orig_factor = ch.ct_primary / ch.ct_secondary if ch.ct_secondary != 0 else 1.0
            new_samples = [sample / orig_factor * factor for sample in ch.samples]
            updated_channels.append(
                {
                    **ch.model_dump(),
                    "samples": new_samples,
                    "ct_primary": pri,
                    "ct_secondary": sec,
                }
            )
        else:
            updated_channels.append(ch.model_dump())

    return {**body.comtrade.model_dump(), "analog_channels": updated_channels}
