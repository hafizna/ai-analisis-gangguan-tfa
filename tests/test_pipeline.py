"""
Tests for Production Pipeline
=============================
Tests for TFAClassifier and FeedbackLogger.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import csv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.classifier import TFAClassifier, ClassificationResult
from pipeline.feedback_logger import FeedbackLogger


class TestTFAClassifier:
    """Tests for TFAClassifier."""

    @pytest.fixture
    def classifier(self):
        """Initialize classifier for testing."""
        return TFAClassifier()

    @pytest.fixture
    def sample_cfg_file(self):
        """Get path to a sample COMTRADE file for testing."""
        # Use a known labeled file from the dataset
        base_dir = Path(__file__).parent.parent.parent / "DFR GANGGUAN UPT"

        # Find first .cfg file in labeled data
        cfg_files = list(base_dir.rglob("*.cfg"))
        if cfg_files:
            return str(cfg_files[0])

        pytest.skip("No sample COMTRADE files found")

    def test_classifier_initialization(self, classifier):
        """Test that classifier initializes without errors."""
        assert classifier is not None
        assert classifier.stage2_resolver is not None
        assert classifier.stage3_classifier is not None

    def test_classify_single_file(self, classifier, sample_cfg_file):
        """Test single file classification."""
        result = classifier.classify(sample_cfg_file)

        # Check result structure
        assert isinstance(result, ClassificationResult)
        assert result.event_id is not None
        assert result.cfg_path == sample_cfg_file

        # Stage 2 outputs should always be present
        assert result.relay_model is not None
        assert result.equipment_type is not None
        assert result.protection_function is not None

        # Notification text should always be generated
        assert result.notification_text != ""
        assert len(result.notification_text) > 0

        # Should not have errors
        assert result.processing_success is True
        assert result.processing_error is None

    def test_notification_text_contains_key_info(self, classifier, sample_cfg_file):
        """Test that notification text contains key information."""
        result = classifier.classify(sample_cfg_file)

        notification = result.notification_text

        # Should contain Indonesian text
        assert any(keyword in notification for keyword in [
            "Gangguan", "Klasifikasi", "Event", "Proteksi"
        ])

        # Should contain fault parameters if available
        if result.fault_current_max_a:
            assert "Arus" in notification or "A" in notification

        if result.faulted_phases:
            assert "Fase" in notification

    def test_notification_text_petir_vs_non_petir(self, classifier):
        """Test that PETIR and NON_PETIR have different notification formats."""
        # Find PETIR and NON-PETIR samples
        base_dir = Path(__file__).parent.parent.parent / "DFR GANGGUAN UPT"

        petir_files = list(base_dir.rglob("*PETIR*/*.cfg"))
        non_petir_files = list(base_dir.rglob("*LAYANG*/*.cfg"))

        if petir_files:
            petir_result = classifier.classify(str(petir_files[0]))
            if petir_result.prediction == "PETIR":
                assert "PETIR" in petir_result.notification_text
                assert "Sambaran Petir" in petir_result.notification_text or "PETIR" in petir_result.notification_text

        if non_petir_files:
            non_petir_result = classifier.classify(str(non_petir_files[0]))
            if non_petir_result.prediction == "NON_PETIR":
                assert "NON-PETIR" in non_petir_result.notification_text or "NON_PETIR" in non_petir_result.notification_text

    def test_stage2_skip_generates_notification(self, classifier):
        """Test that events skipped by Stage 2 still get useful notifications."""
        # Find a non-transmission event (transformer, feeder, etc.)
        base_dir = Path(__file__).parent.parent.parent / "DFR GANGGUAN UPT"
        all_files = list(base_dir.rglob("*.cfg"))

        for cfg_file in all_files[:10]:  # Test first 10 files
            result = classifier.classify(str(cfg_file))

            if not result.proceeded_to_stage3:
                # Should have skip reason
                assert result.stage2_skip_reason is not None

                # Should still have notification
                assert result.notification_text != ""

                # Notification should mention equipment type
                assert any(keyword in result.notification_text for keyword in [
                    "Transformator", "Feeder", "Busbar", "Proteksi"
                ])

                break  # Found one, test passes

    def test_confidence_levels(self, classifier, sample_cfg_file):
        """Test that confidence levels are assigned correctly."""
        result = classifier.classify(sample_cfg_file)

        if result.proceeded_to_stage3 and result.confidence is not None:
            # Check confidence level mapping
            if result.confidence > 0.90:
                assert result.confidence_level == "HIGH"
            elif result.confidence > 0.70:
                assert result.confidence_level == "MEDIUM"
            elif result.confidence > 0.50:
                assert result.confidence_level == "LOW"
            else:
                assert result.confidence_level == "UNCERTAIN"

    def test_classify_batch(self, classifier):
        """Test batch classification."""
        base_dir = Path(__file__).parent.parent.parent / "DFR GANGGUAN UPT"
        cfg_files = list(base_dir.rglob("*.cfg"))[:5]  # Test first 5 files

        if len(cfg_files) < 2:
            pytest.skip("Not enough COMTRADE files for batch test")

        cfg_paths = [str(f) for f in cfg_files]

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            results = classifier.classify_batch(cfg_paths, output_csv=tmp_path)

            # Check results
            assert len(results) == len(cfg_paths)
            assert all(isinstance(r, ClassificationResult) for r in results)

            # Check CSV was created
            assert Path(tmp_path).exists()

            # Check CSV content
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == len(cfg_paths)

                # Check columns
                assert 'event_id' in rows[0]
                assert 'prediction' in rows[0]
                assert 'equipment_type' in rows[0]
        finally:
            # Cleanup
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_event_description_extraction(self, classifier, sample_cfg_file):
        """Test that event description is extracted for all events."""
        result = classifier.classify(sample_cfg_file)

        # At least some event parameters should be populated
        # (Not all files will have all parameters, but most should have current)
        event_params_present = any([
            result.fault_current_max_a is not None,
            result.fault_voltage_min_pu is not None,
            result.faulted_phases,
            result.fault_type is not None
        ])

        assert event_params_present, "No event description parameters extracted"


class TestFeedbackLogger:
    """Tests for FeedbackLogger."""

    @pytest.fixture
    def temp_feedback_dir(self):
        """Create temporary feedback directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def feedback_logger(self, temp_feedback_dir):
        """Initialize feedback logger with temp directory."""
        return FeedbackLogger(log_dir=temp_feedback_dir)

    def test_feedback_logger_initialization(self, feedback_logger):
        """Test that feedback logger initializes correctly."""
        assert feedback_logger is not None
        assert feedback_logger.feedback_csv.exists()

        # Check CSV header
        with open(feedback_logger.feedback_csv, 'r') as f:
            header = f.readline().strip()
            assert 'event_id' in header
            assert 'model_prediction' in header
            assert 'operator_label' in header

    def test_log_feedback(self, feedback_logger):
        """Test logging feedback."""
        feedback_logger.log_feedback(
            event_id="TEST001",
            cfg_path="path/to/file.cfg",
            model_prediction="PETIR",
            model_confidence=0.85,
            operator_label="PETIR",
            is_correct=True,
            operator_notes="Looks correct"
        )

        # Check feedback was logged
        entries = feedback_logger.get_feedback_entries()
        assert len(entries) == 1
        assert entries[0]['event_id'] == "TEST001"
        assert entries[0]['model_prediction'] == "PETIR"
        assert entries[0]['operator_label'] == "PETIR"
        assert entries[0]['is_correct'].lower() == 'true'

    def test_multiple_feedbacks(self, feedback_logger):
        """Test logging multiple feedback entries."""
        # Log multiple feedbacks
        feedback_logger.log_feedback(
            event_id="TEST001",
            cfg_path="path/to/file1.cfg",
            model_prediction="PETIR",
            model_confidence=0.90,
            operator_label="PETIR",
            is_correct=True
        )

        feedback_logger.log_feedback(
            event_id="TEST002",
            cfg_path="path/to/file2.cfg",
            model_prediction="PETIR",
            model_confidence=0.70,
            operator_label="LAYANG",
            is_correct=False,
            operator_notes="Model said PETIR but it's kite"
        )

        feedback_logger.log_feedback(
            event_id="TEST003",
            cfg_path="path/to/file3.cfg",
            model_prediction="NON_PETIR",
            model_confidence=0.65,
            operator_label="NON_PETIR",
            is_correct=True
        )

        # Check all entries
        entries = feedback_logger.get_feedback_entries()
        assert len(entries) == 3

    def test_correction_stats(self, feedback_logger):
        """Test feedback statistics calculation."""
        # Log some feedback (2 correct, 1 incorrect)
        feedback_logger.log_feedback(
            event_id="TEST001", cfg_path="file1.cfg",
            model_prediction="PETIR", model_confidence=0.90,
            operator_label="PETIR", is_correct=True
        )
        feedback_logger.log_feedback(
            event_id="TEST002", cfg_path="file2.cfg",
            model_prediction="PETIR", model_confidence=0.70,
            operator_label="LAYANG", is_correct=False
        )
        feedback_logger.log_feedback(
            event_id="TEST003", cfg_path="file3.cfg",
            model_prediction="NON_PETIR", model_confidence=0.85,
            operator_label="NON_PETIR", is_correct=True
        )

        # Get stats
        stats = feedback_logger.get_correction_stats()

        assert stats['total_reviews'] == 3
        assert stats['correct_count'] == 2
        assert stats['incorrect_count'] == 1
        assert abs(stats['agreement_rate'] - 2/3) < 0.01  # 66.7%

        # Check corrections by type
        assert 'PETIR→LAYANG' in stats['corrections_by_type']
        assert stats['corrections_by_type']['PETIR→LAYANG'] == 1

    def test_export_for_retraining(self, feedback_logger, temp_feedback_dir):
        """Test exporting feedback for retraining."""
        # Log some feedback
        feedback_logger.log_feedback(
            event_id="TEST001", cfg_path="file1.cfg",
            model_prediction="PETIR", model_confidence=0.85,
            operator_label="PETIR", is_correct=True
        )
        feedback_logger.log_feedback(
            event_id="TEST002", cfg_path="file2.cfg",
            model_prediction="NON_PETIR", model_confidence=0.75,
            operator_label="LAYANG", is_correct=False
        )

        # Export
        output_path = Path(temp_feedback_dir) / "retraining_labels.csv"
        count = feedback_logger.export_for_retraining(str(output_path))

        assert count == 2
        assert output_path.exists()

        # Check CSV content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['event_id'] == "TEST001"
            assert rows[0]['label'] == "PETIR"
            assert rows[1]['event_id'] == "TEST002"
            assert rows[1]['label'] == "LAYANG"

    def test_export_with_original_labels(self, feedback_logger, temp_feedback_dir):
        """Test merging feedback with original labels."""
        # Create mock original labels CSV
        original_csv = Path(temp_feedback_dir) / "original_labels.csv"
        with open(original_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['event_id', 'filepath', 'binary_label'])
            writer.writeheader()
            writer.writerow({'event_id': 'ORIG001', 'filepath': 'orig1.cfg', 'binary_label': 'PETIR'})
            writer.writerow({'event_id': 'ORIG002', 'filepath': 'orig2.cfg', 'binary_label': 'NON-PETIR'})

        # Log feedback that corrects ORIG002
        feedback_logger.log_feedback(
            event_id="ORIG002", cfg_path="orig2.cfg",
            model_prediction="NON_PETIR", model_confidence=0.70,
            operator_label="LAYANG", is_correct=False
        )

        # Export merged
        output_path = Path(temp_feedback_dir) / "merged_labels.csv"
        count = feedback_logger.export_for_retraining(
            str(output_path),
            original_labels_csv=str(original_csv)
        )

        assert count == 2

        # Check merged content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # ORIG001 should keep original label
            orig001 = next(r for r in rows if r['event_id'] == 'ORIG001')
            assert orig001['label'] == 'PETIR'
            assert orig001['source'] == 'original'

            # ORIG002 should have corrected label
            orig002 = next(r for r in rows if r['event_id'] == 'ORIG002')
            assert orig002['label'] == 'LAYANG'
            assert orig002['source'] == 'operator_correction'
            assert orig002['original_label'] == 'NON-PETIR'

    def test_append_only_behavior(self, feedback_logger):
        """Test that feedback log is append-only."""
        # Log feedback
        feedback_logger.log_feedback(
            event_id="TEST001", cfg_path="file1.cfg",
            model_prediction="PETIR", model_confidence=0.85,
            operator_label="PETIR", is_correct=True
        )

        # Get file size
        size1 = feedback_logger.feedback_csv.stat().st_size

        # Log more feedback
        feedback_logger.log_feedback(
            event_id="TEST002", cfg_path="file2.cfg",
            model_prediction="NON_PETIR", model_confidence=0.75,
            operator_label="NON_PETIR", is_correct=True
        )

        # File should be larger (appended)
        size2 = feedback_logger.feedback_csv.stat().st_size
        assert size2 > size1

        # Should have both entries
        entries = feedback_logger.get_feedback_entries()
        assert len(entries) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
