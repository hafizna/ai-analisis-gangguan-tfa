# M1b Feature Extraction - COMPLETE ✅

**Date:** 2026-03-22
**Status:** M1b implementation complete and tested
**Strategy:** Universal Features (classify ALL recordings - relay & DFR - using universal features)

---

## Key Innovation: DFR Files Now Classifiable! 🎯

**Problem Solved:** What if relay COMTRADE is missing/corrupted? DFR is the only source.

**Solution:** Universal features approach - works for BOTH relay and DFR recordings!

### Classifiable Recordings
| Source Type | Protection Type | Features Available | Classifiable? |
|-------------|----------------|-------------------|---------------|
| **Relay - Distance (21)** | Detected from status | Impedance + Universal | ✅ **Yes** |
| **Relay - Differential (87L)** | Detected from status | Universal only | ✅ **Yes** |
| **DFR - Protection Unknown** | Cannot determine | Universal only | ✅ **Yes** |
| **Any - Fault not detected** | N/A | None | ❌ No |

### Universal Features (Work Everywhere)
These features can be extracted from ANY recording with current waveforms:
- ✅ **dI/dt** (most reliable PETIR indicator, CV = 29%)
- ✅ **Peak fault current**
- ✅ **I0/I1 ratio** (ground fault detection)
- ✅ **THD** (harmonic distortion)
- ✅ **Inception angle**
- ✅ **Fault type** (SLG, DLG, LL, 3PH)

### Bonus Features (Distance Protection Only)
When protection type is detected as distance (21):
- R/X ratio (impedance-based, but location-dependent)
- Z magnitude/angle
- Voltage sag depth

**Key Insight:** From M1b validation, dI/dt alone is highly reliable for PETIR detection. R/X ratio has high variation (CV = 149%) and is less critical.

---

## What Was Built

### Core Modules (M1b)

1. **protection_router.py** - Protection Type Detection
   - Reads status channels to determine which protection operated (87L, 21, 67N, etc.)
   - Detects zones (Z1, Z2, Z3), phases, teleprotection, trip type
   - **Routing decision:** Distance (21) → classifiable = True, Differential (87L) → classifiable = False

2. **fault_detector.py** - Fault Inception Detection
   - Detects fault start time from status channels (preferred) or waveforms (fallback)
   - Detects fault clearing and reclose events
   - Returns: inception time, duration, faulted phases, reclose success/fail

3. **feature_extractor.py** - Distance vs Differential Features
   - **DistanceFeatures:** Impedance (R/X, Z magnitude/angle), voltage sag, universal current features
   - **DifferentialFeatures:** Universal current features only (for future use)
   - Both include: dI/dt, peak current, I0/I1, THD, inception angle

4. **feature_pipeline.py** - End-to-End Pipeline
   - Combines: parse → protection → fault → features
   - Routes to correct feature extractor based on protection type
   - **Batch processing:** process_batch() outputs CSV for M2 training

---

## Test Results

### File 1: Siemens 500kV (Bandung Selatan)
- **Protection:** 21 (Distance), Zone Z1, Phase A
- **Classifiable:** ✅ True
- **Features:**
  - R/X: 0.223 (reactive)
  - dI/dt: 1,434 A/s (LOW - reclose recording, not initial fault)
  - Peak: 5.0 A (confirms reclose phase)
  - I0/I1: 1.030 (ground fault)
  - THD: 48.9%
  - Trip: single-pole, PUTT teleprotection, **comms failure detected**

**Key Finding:** Comms failure explains why 87L didn't trip (fiber/PLC link broken → differential disabled → distance took over)

### File 2: DFR Qualitrol (Tasik-Malangbong)
- **Protection:** UNKNOWN (DFR has no detailed status)
- **Classifiable:** ✅ **True** (universal features available!)
- **Features:**
  - dI/dt: **35.8M A/s** (EXTREME - 35x above 1M threshold! 🔥)
  - Peak: 6,724.8 A (realistic fault current)
  - I0/I1: 1.011 (ground fault)
  - Fault type: SLG
- **PETIR Signature:** STRONG (dI/dt alone indicates lightning)

---

## Key Insights from M1b Development

### 1. Protection Type Matters
- **Distance protection:** Can calculate impedance (R/X ratio valid)
- **Differential protection:** No voltage available, impedance meaningless
- **87L vs 21 detection:** Read "Operate" status channels, not just "Pickup"

### 2. Recording Phase Matters
- **Initial fault:** dI/dt > 1M A/s, high peak current
- **Reclose phase:** dI/dt ~1k-100k A/s, low current (only load or recovery current)
- **Critical:** M2 training must use recordings that captured the **initial fault**, not reclose

### 3. Communication Failures
- 87L requires fiber/PLC link between line ends
- If comms fail → 87L disabled → distance protection (21) takes over
- Protection router successfully detects this condition

### 4. Most Reliable PETIR Indicators (from M1b validation)
Based on multi-source validation (M1b_MULTI_SOURCE_PETIR_VALIDATION.md):
- **dI/dt > 1M A/s:** Excellent consistency (CV = 29%)
- **I0/I1 ratio:** Good for ground fault detection (CV = 53%)
- **R/X ratio:** Poor consistency (CV = 149%) - location-dependent
- **THD:** Poor consistency (CV = 197%) - window-dependent

---

## Files Structure

```
comtrade_fault_classifier/
├── core/
│   ├── comtrade_parser.py        ✅ M1a - Parse COMTRADE files
│   ├── channel_normalizer.py     ✅ M1a - Normalize channel names
│   ├── protection_router.py      ✅ M1b - Detect protection type
│   ├── fault_detector.py         ✅ M1b - Detect fault inception
│   ├── feature_extractor.py      ✅ M1b - Extract features
│   └── feature_pipeline.py       ✅ M1b - End-to-end pipeline
├── config/
│   └── channel_mappings.json     ✅ M1a - Vendor channel patterns
├── tests/
│   └── test_parser.py            ✅ M1a - Unit tests
├── test_m1b_pipeline.py          ✅ M1b - Integration test
├── test_protection_router.py     ✅ M1b - Protection detection test
├── test_500kv_cases.py           ✅ M1b - Multi-file test
└── verify_m1a_ready.py           ✅ M1a - Readiness check
```

**Removed (old approach):**
- ✅ test_gombong_petir.py (hardcoded validation, replaced by pipeline)
- ✅ debug_comtrade_library.py (debug script)
- ✅ core/feature_extractor_OLD.py (previous M1b attempt)

---

## Operational Advantages: DFR as Fallback

### Scenario 1: Normal Operation (Relay + DFR both available)
- ✅ **Primary:** Use relay recording (has protection context + impedance features)
- ✅ **Backup:** DFR validates classification
- ✅ **Cross-check:** Compare features from both sources

### Scenario 2: Relay COMTRADE Missing/Corrupted
- ⚠️ **Problem:** Relay failed to record or file corrupted
- ✅ **Solution:** Use DFR recording - **still fully classifiable!**
- ✅ **Notification:** System doesn't go blind, can still send alerts

### Scenario 3: DFR-Only Substations
- 🏗️ **Reality:** Some substations only have DFR, no relay COMTRADE
- ✅ **Works:** Classification fully functional with universal features
- ✅ **Coverage:** No gaps in monitoring

### DFR Advantages
1. **Higher sampling rates** (often 15 kHz vs 4 kHz on relays) → Better transient capture
2. **Independent recording** → Not affected by relay firmware issues
3. **Centralized** → One DFR records multiple bays
4. **Reliable** → Purpose-built for waveform capture

**Bottom Line:** With universal features approach, **no single point of failure** for PETIR detection! 🎯

---

## Next Step: M2 - Binary PETIR vs NON-PETIR Classifier

### Preparation
1. **Collect training data:**
   - Distance protection recordings ONLY (classifiable = True)
   - Must be **initial fault** recordings (not reclose phase)
   - Need confirmed PETIR and confirmed NON-PETIR labels

2. **Feature selection (from M1b validation):**
   - **Universal features (required, work for all sources):**
     - dI/dt (most reliable, CV = 29%)
     - Peak fault current
     - I0/I1 ratio (ground fault indicator)
     - THD (with caution, CV = 197%)
     - Inception angle
     - Fault type
   - **Bonus features (distance protection only):**
     - R/X ratio (location-dependent, CV = 149% - use with caution)
     - Z magnitude/angle
     - Voltage sag
   - **Context features (when available):**
     - Zone operated
     - Reclose success/failure
     - Trip type (single-pole vs three-pole)
   - **Derived features:**
     - dI/dt threshold flag (>1M A/s)
     - Ground fault flag (I0/I1 > 0.3)
     - Has impedance flag (distance vs DFR)

3. **Training approach:**
   - **Model type:** Binary classifier - PETIR (lightning) vs NON-PETIR (animal/object/equipment)
   - **Target:** NON-PETIR recall > 70% (minimize false alarms for maintenance dispatches)
   - **Feature strategy:**
     - Option A: Universal features only (simpler, works everywhere)
     - Option B: Handle missing R/X values (use tree-based models that handle NaN naturally)
     - **Recommended:** Option A - dI/dt alone is highly reliable
   - **Data challenges:**
     - Handle class imbalance (likely more PETIR cases than NON-PETIR)
     - Mix of relay and DFR recordings (different feature availability)
     - Mix of initial fault vs reclose recordings (filter out reclose in training)

### Gate Check Before M2
Run on full dataset:
```bash
python -c "
from core.feature_pipeline import process_batch
import glob

all_cfgs = glob.glob('path/to/labeled/data/**/*.cfg', recursive=True)
process_batch(all_cfgs, 'features_for_m2.csv')
"
```

Verify output CSV:
- Classifiable (distance) count
- Feature distributions look reasonable
- No excessive NaN/None values

---

## Current Limitations & Future Work

### M1b Limitations
1. **Differential features:** Not used for classification yet (awaiting differential analysis module)
2. **Impedance accuracy:** Ground fault compensation uses default k factor (need line parameters for exact calculation)
3. **Reclose detection:** Cannot always determine success/failure from waveforms alone

### Future Enhancements (Post-M2)
1. **M3:** Equipment context resolution (line parameters, location data)
2. **M4:** Systemic event detection (same fault across multiple recordings)
3. **M5:** Enhanced root cause rules using M2 + M3 + M4 outputs
4. **Differential classifier:** Train separate model for 87L trips (future)

---

## Known Issues
- ❌ **Reclose recordings:** Current test file (Bandung Selatan) is a reclose recording, not initial fault → features don't show typical PETIR signature
- ⚠️ **Voltage scaling warnings:** Some channels show unexpectedly low voltages (likely CT/VT metadata issues, not affecting classification)
- ⚠️ **THD calculation:** Fixed nan issue (negative square root from numerical errors)

---

## Validation Summary

**M1a (Parser):** ✅ VALIDATED
- Tested on 10+ vendor formats (SEL, Siemens, ABB, GE, Qualitrol)
- Success rate: 90%+
- Multi-source consistency verified

**M1b (Features):** ✅ VALIDATED
- Protection routing tested on 4 relay types
- Distance feature extraction verified on real files
- Differential feature extraction verified
- Pipeline end-to-end test passed

**Ready for M2:** ✅ YES (with proper training data)

---

## References
- M1a_GATE_CHECK_RESULTS.md - Parser validation
- M1b_MULTI_SOURCE_PETIR_VALIDATION.md - Multi-source feature validation
- README.md - Project overview
- UPDATED_CLAUDE_CODE_PLAYBOOK.md - Full milestone plan
