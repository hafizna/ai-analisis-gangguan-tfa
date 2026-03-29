# TFA Classification Pipeline - Usage Guide

## Overview

The TFA (Transmission Fault Analysis) Pipeline integrates:
- **Stage 2**: Context resolution (equipment type + protection function)
- **Stage 3**: PETIR vs NON-PETIR binary classification
- **Feedback System**: Operator validation and retraining data collection

## Quick Start

### 1. Single File Classification

```bash
cd comtrade_fault_classifier
python -m pipeline.classifier path/to/file.cfg --verbose
```

**Output:**
```
================================================================================
Gangguan Saluran Transmisi: GI CIGERELENG
Klasifikasi: PETIR (Sambaran Petir)
Tingkat Keyakinan: MEDIUM (77%)

Parameter Gangguan:
  • Arus Gangguan: 21673 A
  • Tegangan Saat Gangguan: 33.1 kV
  • Durasi Gangguan: 3002 ms

Fitur Utama yang Dipertimbangkan:
  1. Laju Perubahan Arus Maksimum
  2. Arus Gangguan Puncak
  3. Sudut Inception Gangguan

Waktu Event: 2022-01-09 12:40:53 WIB
================================================================================
```

### 2. Batch Classification

```bash
# Process entire directory
python -m pipeline.classifier path/to/folder/ --batch --output results.csv
```

**Output CSV columns:**
- `event_id` - Unique event identifier
- `cfg_path` - Path to COMTRADE file
- `timestamp` - Event timestamp (WIB)
- `station_name` - Station name
- `equipment_type` - transmission_line, transformer, etc.
- `protection_function` - distance, differential, etc.
- `relay_model` - Relay model from .cfg
- `manufacturer` - Detected manufacturer
- `proceeded_to_stage3` - True if Stage 3 ran
- `stage2_skip_reason` - Why Stage 3 was skipped (if applicable)
- `prediction` - PETIR or NON_PETIR
- `confidence` - Confidence score (0-1)
- `confidence_level` - HIGH, MEDIUM, LOW, UNCERTAIN
- `petir_probability` - Probability of PETIR
- `non_petir_probability` - Probability of NON-PETIR
- `fault_current_max_a` - Peak fault current (A)
- `fault_voltage_min_kv` - Voltage during fault (kV)
- `faulted_phases` - Faulted phases (e.g., "A;B")
- `fault_type` - SLG, LL, 3PH, etc.
- `clearing_time_ms` - Fault duration (ms)
- `voltage_level_kv` - System voltage level (kV)
- `operator_confirmed` - Operator validation (empty until reviewed)
- `operator_label` - Operator's classification (empty until reviewed)
- `operator_notes` - Operator notes (empty until reviewed)
- `processing_success` - True if no errors
- `processing_error` - Error message (if any)

### 3. Operator Feedback

Log operator validation/correction:

```bash
# Model was correct
python -m pipeline.feedback_logger log \
  --event-id EVT001 \
  --cfg-path "path/to/file.cfg" \
  --model-prediction PETIR \
  --model-confidence 0.85 \
  --label PETIR \
  --correct true \
  --notes "Confirmed PETIR" \
  --reviewer "Operator Name"

# Model was wrong (correction)
python -m pipeline.feedback_logger log \
  --event-id EVT002 \
  --cfg-path "path/to/file.cfg" \
  --model-prediction PETIR \
  --model-confidence 0.70 \
  --label LAYANG \
  --correct false \
  --notes "Visual inspection shows kite strings" \
  --reviewer "Operator Name"
```

### 4. View Feedback Statistics

```bash
python -m pipeline.feedback_logger stats
```

**Output:**
```
================================================================================
OPERATOR FEEDBACK STATISTICS
================================================================================

Total Reviews: 10
Agreement Rate: 80.0%
  Model Correct: 8
  Model Incorrect: 2

Corrections by Type:
  PETIR->LAYANG: 1
  NON_PETIR->POHON: 1

Corrections by Confidence Level:
  MEDIUM: 2

New Labels Available for Retraining: 10
================================================================================
```

### 5. Export for Retraining

```bash
# Export feedback only
python -m pipeline.feedback_logger export --output retraining_labels.csv

# Merge with original labels
python -m pipeline.feedback_logger export \
  --output merged_labels.csv \
  --original-labels extraction/output/labels_from_folders.csv
```

**Output:**
- `event_id` - Event identifier
- `cfg_path` - Path to file
- `label` - Final label (operator correction or original)
- `source` - original, operator_correction, new_operator_label
- `original_label` - Original label before correction
- `timestamp` - When feedback was logged

## Classification Tiers

### Tier 1: Full Classification (Transmission + Distance)
Events that proceed to Stage 3 PETIR classifier:
- Equipment: `transmission_line`
- Protection: `distance`
- Output: PETIR or NON_PETIR prediction with confidence

**Notification format:**
```
Gangguan Saluran Transmisi: [station]
Klasifikasi: PETIR (Sambaran Petir)
Tingkat Keyakinan: HIGH (92%)

Parameter Gangguan:
  • Arus Gangguan: 15000 A
  • Tegangan Saat Gangguan: 45.2 kV
  • Fase Terganggu: A-B (LL)
  • Durasi Gangguan: 250 ms
```

### Tier 2/3: Event Description Only
Events that skip Stage 3:
- Transformers (`equipment_type: transformer`)
- Feeders (`equipment_type: feeder_20kv`)
- Differential protection (`protection_function: differential`)
- Low confidence equipment identification

**Notification format:**
```
Event Proteksi: [station]
Jenis Peralatan: Transformator
Fungsi Proteksi: Relay Diferensial (87)
Relay: REL670 (ABB)
Catatan: Klasifikasi penyebab tidak tersedia (Transformer protection - use transformer fault classifier)

Parameter Gangguan:
  • Arus Gangguan: 8500 A
  • Durasi Gangguan: 180 ms
```

## Confidence Levels

| Level | Confidence Range | Recommendation |
|-------|-----------------|----------------|
| **HIGH** | > 90% | Accept prediction |
| **MEDIUM** | 70-90% | Flag for validation |
| **LOW** | 50-70% | Present both options to operator |
| **UNCERTAIN** | ≤ 50% | Manual review required |

## File Structure

```
comtrade_fault_classifier/
├── pipeline/
│   ├── classifier.py           # Main classification pipeline
│   └── feedback_logger.py      # Feedback collection system
├── config/
│   ├── relay_lookup.json       # Relay model database (Stage 2)
│   └── channel_mappings.json   # Channel normalization
├── models/
│   ├── stage3_petir_classifier.pkl      # Trained PETIR classifier
│   └── stage3_feature_columns.pkl       # Feature column order
├── feedback/
│   ├── feedback_log.csv        # Operator feedback (append-only)
│   └── feedback_stats.json     # Statistics cache
└── tests/
    └── test_pipeline.py        # Pipeline tests
```

## Integration Example

```python
from pipeline import TFAClassifier, FeedbackLogger

# Initialize classifier
classifier = TFAClassifier()

# Classify single event
result = classifier.classify("path/to/file.cfg")

# Display to operator
print(result.notification_text)

# Operator provides feedback
if result.prediction:
    feedback_logger = FeedbackLogger()
    feedback_logger.log_feedback(
        event_id=result.event_id,
        cfg_path=result.cfg_path,
        model_prediction=result.prediction,
        model_confidence=result.confidence,
        operator_label=user_input_label,  # From operator UI
        is_correct=(user_input_label == result.prediction),
        operator_notes=user_input_notes
    )
```

## API Reference

### TFAClassifier

```python
classifier = TFAClassifier(
    config_dir="config/",      # Relay lookup config directory
    models_dir="models/",      # Trained models directory
    timezone="Asia/Jakarta"    # Timezone for timestamps (WIB)
)

# Single file
result = classifier.classify(cfg_path: str) -> ClassificationResult

# Batch
results = classifier.classify_batch(
    cfg_paths: List[str],
    output_csv: Optional[str] = None
) -> List[ClassificationResult]
```

### FeedbackLogger

```python
logger = FeedbackLogger(log_dir="feedback/")

# Log feedback
logger.log_feedback(
    event_id: str,
    cfg_path: str,
    model_prediction: str,           # "PETIR" or "NON_PETIR"
    model_confidence: float,         # 0-1
    operator_label: str,             # Ground truth from operator
    is_correct: bool,                # Does operator agree?
    operator_notes: str = "",
    model_confidence_level: str = "",
    reviewer_name: str = ""
)

# Get statistics
stats = logger.get_correction_stats() -> Dict

# Export for retraining
count = logger.export_for_retraining(
    output_path: str,
    original_labels_csv: Optional[str] = None
) -> int
```

## Troubleshooting

### "Model not found" Error
```
FileNotFoundError: Model not found: models/stage3_petir_classifier.pkl
```
**Solution:** Train the model first:
```bash
python -m training.train_stage3
```

### "Relay lookup file not found"
```
FileNotFoundError: Relay lookup file not found: config/relay_lookup.json
```
**Solution:** Relay lookup should exist. Check path or reinitialize config.

### No Stage 3 Predictions
If all events skip Stage 3:
- Check `stage2_skip_reason` in output CSV
- Most common: Not transmission line + distance protection
- Expected behavior for transformer/feeder events

### Unicode Errors on Windows
If you see encoding errors with bullet points (•) or checkmarks (✓):
- These are cosmetic only
- CSV output is not affected
- Notifications will display correctly in web interfaces

## Gate Checks (Testing)

Run these checks to validate the pipeline:

```bash
# 1. Single file end-to-end
python -m pipeline.classifier "path/to/known/petir/file.cfg"
# CHECK: Notification text is useful and in Bahasa Indonesia
# CHECK: Confidence level matches the prediction probability

# 2. Batch on all labeled data
python -m pipeline.classifier "path/to/labeled/folder/" --batch --output validation_results.csv
# CHECK: All events produce results (no crashes)
# CHECK: Transmission+distance events have Stage 3 predictions
# CHECK: Other events have stage2_skip_reason populated
# CHECK: notification_text is never empty

# 3. Feedback loop
python -m pipeline.feedback_logger log --event-id TEST001 --cfg-path test.cfg --model-prediction PETIR --model-confidence 0.85 --label PETIR --correct true
python -m pipeline.feedback_logger log --event-id TEST002 --cfg-path test.cfg --model-prediction PETIR --model-confidence 0.70 --label LAYANG --correct false --notes "Model said PETIR"
python -m pipeline.feedback_logger stats
# CHECK: Shows 2 reviews, 50% agreement rate

# 4. Export for retraining
python -m pipeline.feedback_logger export --output test_export.csv
# CHECK: Contains feedback corrections
```

## Next Steps

1. **Deploy Pipeline**: Integrate with monitoring system
2. **Collect Feedback**: Operators validate predictions
3. **Expand Dataset**: Use feedback to retrain with more labeled events
4. **Build Stage 4**: Root cause classification for NON-PETIR (when dataset grows)
5. **Add Stage 1**: Systemic event detection (future enhancement)

## Support

For issues or questions:
- Check PROJECT_SUMMARY_FOR_CONSULTATION.md for architecture details
- Review test suite: `tests/test_pipeline.py`
- Consult training reports: `models/stage3_training_report.txt`
