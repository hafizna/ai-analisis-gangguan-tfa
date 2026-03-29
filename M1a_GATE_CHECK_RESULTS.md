# M1a Gate Check Results

**Status: PASSED** [OK]

Date: 2026-03-22

## Gate Check Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Unit tests pass | [OK] PASS | 18/18 tests passed |
| Real file parses without crashing | [OK] PASS | Tested on 2 different relay vendors |
| Channel names normalize correctly | [OK] PASS | VA, VB, VC, IA, IB, IC, IN all correct |
| Voltage values in kV range (not mV or MV) | [OK] PASS | 150 kV system: ~87 kV RMS, 70 kV system: ~40 kV RMS |
| Current values in A-kA range (not mA) | [OK] PASS | Fault currents 100-5000 A range |
| No unexpected warnings | [OK] PASS | Clean parse, only expected warnings for neutral voltage |

## Files Tested

### 1. Qualitrol Relay - 150 kV System (ULAR Fault)
**File:** `230706,125624990,+7h0,GI 150KV UNGARAN,TAMBAKLOROK 1-2,Qualitrol LLC.cfg`

**Results:**
- Station: GI 150KV UNGARAN
- Relay: TAMBAKLOROK 1-2
- Frequency: 50.0 Hz
- Channels: 18 analog, 64 status
- Sampling rate: 25,600 Hz

**Voltages (kV):**
- VA: 87.80 kV RMS (peak: 124 kV) [OK]
- VB: 85.01 kV RMS (peak: 124 kV) [OK]
- VC: 87.56 kV RMS (peak: 124 kV) [OK]

**Currents (A):**
- Phase A: 78 A RMS (peak: 169 A)
- Phase B: **323 A RMS** (peak: 2565 A) ← **FAULT PHASE**
- Phase C: 88 A RMS (peak: 290 A)
- Neutral: **325 A RMS** (peak: 2645 A) ← **GROUND FAULT**

**Fault Type Identified:** Phase B to Ground (matches snake contact pattern!)

**Channel Normalization:**
- RST phase notation correctly mapped to ABC (VR→VA, VS→VB, VT→VC, IR→IA, IS→IB, IT→IC) [OK]
- All required channels present: VA, VB, VC, IA, IB, IC, IN [OK]

### 2. 70 kV System
**File:** `Bay Pangandaran GI Banjar.cfg`

**Results:**
- Station: GI 70KV BANJAR
- Relay: PGDRN 1-2
- Frequency: 50.0 Hz
- Channels: 18 analog, 64 status

**Voltages (kV):**
- VA: 40.52 kV RMS (peak: 81 kV) [OK]
- VB: 45.32 kV RMS (peak: 102 kV) [OK]
- VC: 36.73 kV RMS (peak: 57 kV) [OK]

**Currents (A):**
- Phase A: 78 A RMS (peak: 142 A)
- Phase B: 77 A RMS (peak: 271 A)
- Phase C: **154 A RMS** (peak: 520 A) ← **FAULT PHASE**
- Neutral: **116 A RMS** (peak: 460 A) ← **GROUND FAULT**

**Fault Type Identified:** Phase C to Ground

**Control Signals:**
- A - CH04: 199 mV (correctly preserved as mV, not converted) [OK]
- A - CH09: 149 mA (correctly preserved as mA, not converted) [OK]

## Key Fixes Applied

### Issue 1: CT/VT Ratio Double-Application
**Problem:** Values were 1000× too high (186 million volts instead of 186 kV)

**Root Cause:** The `comtrade` library already converts samples to primary values using the multiplier `a` from the .cfg file. We were applying the primary/secondary ratio a second time.

**Fix:** Removed duplicate ratio multiplication. Parser now uses values directly from `com.analog[i]` which are already in primary engineering units.

### Issue 2: mV/mA Treated as MV/MA
**Problem:** Control signal channels with mV (millivolts) or mA (milliamps) were being converted to kV or A, causing extreme values.

**Root Cause:** Case-insensitive unit matching: "mV".upper() = "MV" (megavolts)

**Fix:** Added case-sensitive check for mV/mA units before case-insensitive conversion. These control signals are now preserved with original units.

### Issue 3: RST Phase Notation Not Recognized
**Problem:** Qualitrol relays use RST notation (VR, VS, VT for voltages; IR, IS, IT for currents) instead of ABC.

**Root Cause:** Generic pattern matching only checked for ABC notation.

**Fix:** Added RST to ABC mapping in `channel_normalizer.py`:
- VR → VA (phase R = phase A)
- VS → VB (phase S = phase B)
- VT → VC (phase T = phase C)
- Similar for currents: IR→IA, IS→IB, IT→IC

## CT/VT Ratio Validation

Parser now validates CT/VT ratios and warns if unusual:

**Common CT ratios:** X/5 or X/1 where X ≥ 100 (e.g., 300/5, 1600/1, 2000/5)
- If secondary ≠ 1 or 5 → Warning
- If primary < 100 (except 1:1) → Warning

**Common VT ratios:** X/100, X/110, X/125 (e.g., 150000/110 for 150 kV system)
- If secondary ∉ {1, 100, 110, 125} → Warning

These warnings help detect .cfg file data issues (like SIGRA prompting for manual input).

## Unit Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
collected 18 items

tests/test_parser.py::TestChannelNormalization::test_sel_channels PASSED
tests/test_parser.py::TestChannelNormalization::test_abb_channels PASSED
tests/test_parser.py::TestChannelNormalization::test_siemens_channels PASSED
tests/test_parser.py::TestChannelNormalization::test_ge_channels PASSED
tests/test_parser.py::TestChannelNormalization::test_residual_channels PASSED
tests/test_parser.py::TestChannelNormalization::test_unknown_channel PASSED
tests/test_parser.py::TestManufacturerDetection::test_sel_detection PASSED
tests/test_parser.py::TestManufacturerDetection::test_abb_detection PASSED
tests/test_parser.py::TestManufacturerDetection::test_siemens_detection PASSED
tests/test_parser.py::TestManufacturerDetection::test_ge_detection PASSED
tests/test_parser.py::TestManufacturerDetection::test_qualitrol_detection PASSED
tests/test_parser.py::TestManufacturerDetection::test_unknown PASSED
tests/test_parser.py::TestPrimaryConversion::test_ct_conversion PASSED
tests/test_parser.py::TestPrimaryConversion::test_vt_conversion PASSED
tests/test_parser.py::TestParserEdgeCases::test_missing_dat_file PASSED
tests/test_parser.py::TestParserEdgeCases::test_encoding_issues PASSED
tests/test_parser.py::TestParserEdgeCases::test_zero_secondary PASSED
tests/test_parser.py::test_parser_on_real_file PASSED

============================= 18 passed in 1.87s ==============================
```

## Next Steps

**M1a is COMPLETE and VALIDATED.**

Ready to proceed to **M1b: Feature Extraction**:
1. Fault inception detection (identify fault start time)
2. Pre-fault vs fault window segmentation
3. Impedance calculation (R + jX from V and I)
4. Symmetrical components (positive, negative, zero sequence)
5. Transient characteristics (dI/dt, dV/dt, rise time)
6. Harmonic analysis (THD, specific harmonics)

These features will feed into the binary PETIR vs NON-PETIR classifier (M2).

## Architecture Validation

The parser follows all principles from UPDATED_CLAUDE_CODE_PLAYBOOK.md:
- Never crashes (returns None for unreadable files)
- Comprehensive warnings system (not silent failures)
- Vendor-agnostic (handles SEL, ABB, Siemens, GE, Qualitrol)
- Primary value conversion (ready for physics-based features)
- Standardized units (kV for voltage, A for current)
- Clean separation of concerns (parsing separate from feature extraction)

**Foundation is solid. Ready to build M1b on top of this.**
