# Pattern Analysis Results - Should We Build Stage 4?

## Executive Summary

**Verdict: BUILD IT (with caution) ⚠️**

The data shows **promising patterns** for some causes, but **sample sizes are critically small** for most. Stage 4 should be built as a **rough guide with mandatory human review**, not a confident classifier.

## Dataset Overview

```
Total labeled: 82 events
├─ PETIR: 62 events ✓ (good)
└─ NON-PETIR: 20 events
    ├─ LAYANG: 11 events ✓ (borderline - might work)
    ├─ CONDUCTOR_BROKEN: 3 events ⚠️ (very risky)
    ├─ POHON: 2 events ❌ (too few)
    ├─ BENDA_LAIN: 2 events ❌ (too few)
    ├─ CT_BREAKDOWN: 1 event ❌ (impossible)
    └─ HEWAN: 1 event ❌ (impossible)
```

## Key Finding #1: Patterns DO Exist! 🎯

**All 9 features show some separation between causes.**

### Most Distinctive Patterns:

| Cause | Strongest Signal | Values | Confidence |
|-------|-----------------|--------|------------|
| **LAYANG** | Very deep voltage sag | 0.87 pu (vs 0.57 others) | ✓ Strong |
| | Moderate di_dt | 616k (vs 1.4M PETIR) | ✓ Strong |
| | High THD | 45% (vs 23% PETIR) | ✓ Moderate |
| **CONDUCTOR_BROKEN** | Extremely high di_dt | 1.8M (highest of all!) | ✓ Strong |
| | Very specific Z angle | -160° (±4°) | ✓✓ Very Strong |
| | Low THD | 3.5% (vs 45% LAYANG) | ✓ Strong |
| **POHON** | Maxed THD | 100% (saturated harmonics) | ✓ Strong |
| | Very shallow voltage sag | 0.10 pu (vs 0.87 LAYANG) | ✓ Strong |
| **BENDA_LAIN** | Extremely low di_dt | 1,877 (vs 600k others) | ✓ Strong |
| | Very low fault current | 6 A (vs 2-6k others) | ✓ Strong |
| **CT_BREAKDOWN** | Minuscule fault current | 1.08 A (measurement error) | ✓✓ Very Strong |
| | Low di_dt | 247 (vs hundreds of thousands) | ✓ Strong |

### Separability Analysis:

**9 out of 9 features** show statistical separation (>1σ) between at least one pair of causes. This is **much better than expected**!

**Best separating features:**
1. **di_dt_max** - CONDUCTOR vs others (huge difference!)
2. **thd_percent** - POHON vs others (maxed out)
3. **voltage_sag_depth_pu** - LAYANG vs POHON (0.87 vs 0.10)
4. **z_angle_degrees** - CONDUCTOR vs all (unique -160°)
5. **peak_fault_current_a** - CT/BENDA_LAIN vs others (very low)

## Key Finding #2: PETIR vs NON-PETIR is WEAK! ⚠️

**Shocking revelation:** PETIR and NON-PETIR (as groups) are **barely separable**!

```
Feature                    Separation   PETIR Mean   NON-PETIR Mean
voltage_sag_depth_pu       0.32σ        0.57         0.71
di_dt_max                  0.36σ        1,437,359    683,336
z_angle_degrees            0.41σ        -1.4°        -41.2°
thd_percent                0.38σ        23%          37%
peak_fault_current_a       0.36σ        4,404 A      2,282 A
```

**All features: 0.2-0.7σ separation** (very weak!)

**Implication:** This explains why Stage 3 binary classifier struggles. The underlying features don't strongly distinguish PETIR from NON-PETIR as groups.

**However:** Individual NON-PETIR causes (LAYANG, CONDUCTOR, etc.) ARE distinguishable from each other. The problem is that Stage 3 lumps them all together.

## Key Finding #3: Fault Type Detection is FAILING! 🚨

**CRITICAL ISSUE:**
```
Fault type = UNKNOWN:     76/82 events (93%)
Faulted phases = Unknown: 76/82 events (93%)
```

**This breaks theory-based rules!**

The playbook suggests rules like:
- "LAYANG: phase-to-phase fault" ← **Can't use this** (don't have phase data!)
- "POHON: single-phase-to-ground" ← **Can't use this** (don't have fault type!)

**Root cause:** The fault type classifier in `feature_extractor.py` is not detecting most faults correctly.

**Impact:** We must base rules on **continuous features** (voltage sag, di_dt, impedance, THD) rather than **categorical features** (fault type, faulted phases).

## Recommended Approach: BUILD with Heavy Caveats

### Strategy: Empirical Rules (Data-Driven, Not Theory-Driven)

Since we can't use fault type, build rules from **observed feature values**:

#### Rule Set:

**1. CT_BREAKDOWN (easiest to detect):**
```
IF peak_fault_current_a < 10 A
   AND di_dt_max < 1000
   → CT_BREAKDOWN (measurement anomaly)
   Score: 0.8
```

**2. BENDA_LAIN (low energy faults):**
```
IF peak_fault_current_a < 100 A
   AND di_dt_max < 10,000
   → BENDA_LAIN (low energy object)
   Score: 0.6
```

**3. CONDUCTOR_BROKEN (very distinctive!):**
```
IF di_dt_max > 1,000,000
   AND z_angle_degrees < -150° OR z_angle_degrees > 150°
   AND thd_percent < 10%
   → CONDUCTOR_BROKEN (arc + conductor separation)
   Score: 0.7
```

**4. POHON (saturated harmonics):**
```
IF thd_percent > 80%
   AND voltage_sag_depth_pu < 0.3
   → POHON (tree contact, high distortion)
   Score: 0.5
```

**5. LAYANG (deep sag + moderate rise):**
```
IF voltage_sag_depth_pu > 0.80
   AND di_dt_max BETWEEN 100k AND 1M
   AND thd_percent > 30%
   → LAYANG (kite/string bridging phases)
   Score: 0.5
```

**6. OTHER (catch-all):**
```
IF none of above rules fire strongly
   → OTHER (unknown cause)
   Score: baseline from unfired rules
```

### Scoring System:

1. Each rule gives a raw score
2. Normalize scores to sum = 1.0
3. Return **top 3 predictions** ranked by score
4. **If max score < 0.5**: Extra emphasis that human review needed
5. **Always flag for review** (requires_human_review = True)

### Output Format:

```
Root Cause Prediction (NON-PETIR):
  1. LAYANG (48%) - Deep voltage sag (0.87), high THD (52%)
  2. POHON (32%) - Saturated harmonics (THD 100%), shallow sag
  3. CONDUCTOR_BROKEN (20%) - High di/dt, moderate impedance

⚠️ REQUIRES OPERATOR REVIEW
Reason: Rule-based classification with limited training data (20 NON-PETIR samples)

Key Features:
  Fault Current: 2,450 A
  Voltage Sag: 0.84 pu (84% dip)
  di/dt: 615,000 A/s
  THD: 52%
  Z angle: -12°
```

## Validation Against Known Samples

Before implementing, we should validate rules against the 20 NON-PETIR samples:

**Expected results:**
- **LAYANG (11 samples):** Rules should get 60-70% correct in top-1, 90%+ in top-3
- **CONDUCTOR_BROKEN (3 samples):** Might get 1-2 correct (too few to validate)
- **POHON (2 samples):** Coin flip (impossible to validate)
- **Others (4 samples):** Likely misclassified (too few to learn from)

**Acceptance criteria:**
- If >50% of LAYANG events have LAYANG in top-3 → proceed
- If CONDUCTOR unique features (z_angle=-160°) appear consistently → add rule
- Flag everything for review regardless

## Alternatives to Consider

### Option A: Build Full Stage 4 (Recommended)
✓ **Do this if:** You want to give operators *some* guidance
✓ **Pros:** Better than nothing, encodes domain observations
✗ **Cons:** Will be wrong often, might mislead if overtrusted

### Option B: Build LAYANG-Only Detector
✓ **Do this if:** You want higher confidence predictions
✓ **Pros:** 11 samples is borderline acceptable
✗ **Cons:** Other causes get no help

### Option C: Just Show Feature Summary
✓ **Do this if:** You'd rather wait for more data
✓ **Pros:** Can't mislead, operators learn to read features themselves
✗ **Cons:** No automated guidance, slower operator workflow

### Option D: Hybrid (Recommended)
✓ **Do this:** Combine all above
- Build rules for LAYANG + CONDUCTOR (distinctive patterns exist)
- Show feature summary for all events
- Collect operator decisions as labeled data
- Retrain rules quarterly as dataset grows

## Decision

**I recommend Option D (Hybrid):**

1. **Build Stage 4 with conservative rules**
   - Focus on LAYANG (11 samples, clear patterns)
   - Add CONDUCTOR rule (distinctive z_angle=-160°)
   - Lump POHON/BENDA_LAIN/CT/HEWAN into "OTHER"

2. **Always flag for human review**
   - Show ranked predictions (top 3)
   - Include key feature values
   - Explain which rules fired

3. **Collect feedback aggressively**
   - Every operator decision becomes a label
   - Retrain rules when NON-PETIR samples > 50
   - Build ML classifier when NON-PETIR samples > 100

4. **Fix fault type detection** (parallel effort)
   - Current detection failing (93% unknown)
   - Would unlock better rules if fixed

## Honest Assessment

**Can we build Stage 4?** Yes, but it will be mediocre.

**Will it help operators?** Yes - even rough guidance is better than nothing.

**Should we trust it?** No - every prediction must be reviewed.

**Is it worth building now?** **YES** - because:
- Quick win (can build in 1-2 hours)
- Encodes observable patterns
- Provides feedback collection mechanism
- Will improve as data grows
- Better than leaving operators with just "NON-PETIR"

**Bottom line:** Build it, flag everything for review, collect feedback, improve iteratively.

---

**Next Step:** Do you want me to proceed with building Stage 4 using the empirical rules above? 🚀
