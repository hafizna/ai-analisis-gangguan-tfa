import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import webapp.app as webapp_app


def _mk_analog(name, canonical, measurement, samples, unit="A", pors="P"):
    return SimpleNamespace(
        id=name,
        name=name,
        canonical_name=canonical,
        measurement=measurement,
        unit=unit,
        ct_primary=1.0,
        ct_secondary=1.0,
        scale_a=1.0,
        scale_b=0.0,
        pors=pors,
        samples=np.asarray(samples, dtype=float),
    )


def _mk_status(name, samples):
    return SimpleNamespace(id=name, name=name, samples=np.asarray(samples, dtype=int))


def test_frequency_channel_not_classified_as_voltage():
    ch = SimpleNamespace(name="Freq:VR PBLGA 1", id="17", canonical_name="VA", measurement="other")
    assert webapp_app._infer_waveform_measurement(ch) == "other"


def test_extract_cfg_ratios_marks_primary_native(monkeypatch):
    record = SimpleNamespace(
        analog_channels=[
            _mk_analog("IA", "IA", "current", [0, 1, 0], pors="P"),
            _mk_analog("IB", "IB", "current", [0, 1, 0], pors="P"),
            _mk_analog("VA", "VA", "voltage", [0, 1, 0], unit="kV", pors="P"),
        ]
    )
    monkeypatch.setattr(webapp_app, "parse_comtrade", lambda _: record)
    ratios = webapp_app._extract_cfg_ratios("dummy.cfg")
    assert ratios["cfg_ct_known"] is False
    assert ratios["cfg_vt_known"] is False
    assert ratios["ct_primary_native"] is True
    assert ratios["vt_primary_native"] is True


def test_resolve_ratio_defaults_keeps_inputs_blank_when_metadata_missing():
    resolved = webapp_app._resolve_ratio_defaults(
        {
            "cfg_ct_known": False,
            "cfg_vt_known": False,
            "ct_primary_native": True,
            "vt_primary_native": False,
        }
    )
    assert resolved["ct_primary"] is None
    assert resolved["ct_secondary"] is None
    assert resolved["vt_primary"] is None
    assert resolved["vt_secondary"] is None
    assert resolved["ct_base_primary"] == 1.0
    assert resolved["ct_base_secondary"] == 1.0
    assert resolved["vt_base_primary"] == 1.0
    assert resolved["vt_base_secondary"] == 1.0
    assert resolved["ct_ratio_source"] == "meta_primary"
    assert resolved["vt_ratio_source"] == "meta"


def test_resolve_ratio_defaults_uses_cfg_ratios_when_available():
    resolved = webapp_app._resolve_ratio_defaults(
        {
            "cfg_ct_known": True,
            "cfg_vt_known": True,
            "cfg_ct_primary": 2000.0,
            "cfg_ct_secondary": 1.0,
            "cfg_vt_primary": 150000.0,
            "cfg_vt_secondary": 100.0,
        }
    )
    assert resolved["ct_ratio_source"] == "cfg"
    assert resolved["vt_ratio_source"] == "cfg"
    assert resolved["ct_primary"] == 2000.0
    assert resolved["ct_secondary"] == 1.0
    assert resolved["vt_primary"] == 150000.0
    assert resolved["vt_secondary"] == 100.0
    assert resolved["ct_base_primary"] == 2000.0
    assert resolved["vt_base_primary"] == 150000.0


def test_recalculate_uses_file_meta_baseline_not_estimated_defaults(monkeypatch):
    analysis = {
        "ct_primary": None,
        "ct_secondary": None,
        "vt_primary": None,
        "vt_secondary": None,
        "ct_primary_default": None,
        "ct_secondary_default": None,
        "vt_primary_default": None,
        "vt_secondary_default": None,
        "ratio_base_ct_primary": 1.0,
        "ratio_base_ct_secondary": 1.0,
        "ratio_base_vt_primary": 1.0,
        "ratio_base_vt_secondary": 1.0,
        "peak_fault_current_a": 18.0,
        "i0_magnitude_a": 2.5,
        "i1_magnitude_a": 5.0,
        "i2_magnitude_a": 1.0,
        "v_prefault_v": 60.0,
        "v_fault_v": 20.0,
        "z_magnitude_ohms": 12.5,
        "fault_duration_ms": 75.0,
        "fault_count": 1,
        "i0_i1_ratio": 0.5,
        "voltage_sag_depth_pu": 0.596,
        "di_dt_max": 1000.0,
        "reclose_successful": "False",
        "trip_type": "three_pole",
        "faulted_phases": "A",
        "fault_type": "SLG",
        "label": "PETIR",
        "confidence": 0.8,
        "tier": 1,
        "rule_name": "dummy",
        "evidence": "",
        "recommendation": "",
        "description": "",
        "cause_pcts": [],
    }
    monkeypatch.setattr(webapp_app, "apply_rules", lambda row: None)
    monkeypatch.setattr(webapp_app, "_load_model", lambda: None)
    updated = webapp_app._recalculate_analysis_with_ratio(analysis, 2000.0, 5.0, 100.0, 10.0)
    assert updated["peak_fault_current_a"] == 7200.0
    assert updated["v_prefault_v"] == 600.0
    assert updated["ratio_override_active"] is True


def test_recalculate_back_to_meta_baseline_restores_blank_inputs(monkeypatch):
    analysis = {
        "ct_primary": 2000.0,
        "ct_secondary": 5.0,
        "vt_primary": 100.0,
        "vt_secondary": 10.0,
        "ct_primary_default": None,
        "ct_secondary_default": None,
        "vt_primary_default": None,
        "vt_secondary_default": None,
        "ratio_base_ct_primary": 1.0,
        "ratio_base_ct_secondary": 1.0,
        "ratio_base_vt_primary": 1.0,
        "ratio_base_vt_secondary": 1.0,
        "ratio_base_values": {
            "peak_fault_current_a": 18.0,
            "i0_magnitude_a": 2.5,
            "i1_magnitude_a": 5.0,
            "i2_magnitude_a": 1.0,
            "v_prefault_v": 60.0,
            "v_fault_v": 20.0,
            "z_magnitude_ohms": 12.5,
            "orig_ct_p": 1.0,
            "orig_ct_s": 1.0,
            "orig_vt_p": 1.0,
            "orig_vt_s": 1.0,
        },
        "fault_duration_ms": 75.0,
        "fault_count": 1,
        "i0_i1_ratio": 0.5,
        "voltage_sag_depth_pu": 0.596,
        "di_dt_max": 1000.0,
        "reclose_successful": "False",
        "trip_type": "three_pole",
        "faulted_phases": "A",
        "fault_type": "SLG",
        "label": "PETIR",
        "confidence": 0.8,
        "tier": 1,
        "rule_name": "dummy",
        "evidence": "",
        "recommendation": "",
        "description": "",
        "cause_pcts": [],
    }
    monkeypatch.setattr(webapp_app, "apply_rules", lambda row: None)
    monkeypatch.setattr(webapp_app, "_load_model", lambda: None)
    updated = webapp_app._recalculate_analysis_with_ratio(analysis, 1.0, 1.0, 1.0, 1.0)
    assert updated["ct_primary"] is None
    assert updated["vt_primary"] is None
    assert updated["ratio_override_active"] is False


def test_build_waveform_payload_filters_to_active_line(monkeypatch):
    time_axis = np.arange(0.0, 0.010, 0.001)
    record = SimpleNamespace(
        time=time_axis,
        analog_channels=[
            _mk_analog("VR PBLGA 1", "VA", "voltage", [80, 81, 79, 82, 60, 55, 58, 80, 81, 80], unit="kV"),
            _mk_analog("VR PBLGA 2", "VA", "voltage", [80, 81, 80, 80, 79, 80, 80, 80, 81, 80], unit="kV"),
            _mk_analog("IR PBLGA 1", "IA", "current", [10, 12, 11, 15, 200, 180, 60, 14, 12, 11]),
            _mk_analog("IS PBLGA 1", "IB", "current", [8, 9, 8, 10, 120, 110, 40, 9, 8, 8]),
            _mk_analog("IT PBLGA 1", "IC", "current", [7, 8, 7, 9, 90, 80, 30, 8, 7, 7]),
            _mk_analog("IR PBLGA 2", "IA", "current", [9, 10, 9, 10, 25, 24, 15, 10, 9, 9]),
            _mk_analog("IS PBLGA 2", "IB", "current", [8, 8, 8, 9, 20, 18, 12, 8, 8, 8]),
            _mk_analog("IT PBLGA 2", "IC", "current", [7, 7, 7, 8, 18, 17, 11, 7, 7, 7]),
            _mk_analog("Freq:VR PBLGA 1", "VA", "other", [50] * 10, unit="Hz"),
            _mk_analog("SPARE", "SPARE", "voltage", [0] * 10, unit="kV"),
        ],
        status_channels=[
            _mk_status("LP OPRT PBLGA 1", [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
            _mk_status("LP OPRT PBLGA 2", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ],
    )
    monkeypatch.setattr(webapp_app, "parse_comtrade", lambda _: record)

    payload = webapp_app._build_waveform_payload("dummy.cfg", inception_ms=4.0, duration_ms=2.0)

    names = [ch["name"] for ch in payload["channels"]]
    assert payload["meta"]["selected_line_tag"] == "1"
    assert "VR PBLGA 1" in names
    assert "IR PBLGA 1" in names
    assert all("PBLGA 2" not in name for name in names)
    assert all("Freq:" not in name for name in names)
    assert "SPARE" not in names
