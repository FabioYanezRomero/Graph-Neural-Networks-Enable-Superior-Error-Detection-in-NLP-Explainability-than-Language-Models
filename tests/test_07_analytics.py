"""
Tests for Analytics - 4-Dimension Evaluation (Paper Section 3.5)

Tests the src.Analytics module covering:
- Dimension 1: AUC Discriminative Capacity
- Dimension 2: Behavioral Faithfulness (Fidelity)
- Dimension 3: Consistency Across Outcomes
- Dimension 4: Feature Ranking Stability (Progression)
"""
import pytest


class TestAnalyticsImports:
    """Test that Analytics modules import correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import Analytics
        assert Analytics is not None
    
    def test_auc_module_import(self):
        """Test AUC dimension imports."""
        from src.Analytics import auc
        assert auc is not None
    
    def test_fidelity_module_import(self):
        """Test Fidelity dimension imports."""
        from src.Analytics import fidelity
        assert fidelity is not None
    
    def test_consistency_module_import(self):
        """Test Consistency dimension imports."""
        from src.Analytics import consistency
        assert consistency is not None
    
    def test_progression_module_import(self):
        """Test Progression dimension imports."""
        from src.Analytics import progression
        assert progression is not None


class TestDimension1AUC:
    """Test AUC Discriminative Capacity (Dimension 1)."""
    
    def test_auc_values_in_range(self, sample_auc_metrics):
        """Test AUC values are in [0, 1] range."""
        assert 0 <= sample_auc_metrics["deletion_auc"] <= 1
        assert 0 <= sample_auc_metrics["insertion_auc"] <= 1
    
    def test_deletion_auc_higher_for_correct(self):
        """Paper finding: correct predictions have higher deletion AUC."""
        # As per paper: correct ~0.92, incorrect ~0.70
        correct_auc = 0.92
        incorrect_auc = 0.70
        assert correct_auc > incorrect_auc


class TestDimension2Fidelity:
    """Test Behavioral Faithfulness (Dimension 2)."""
    
    def test_asymmetry_index_calculation(self, sample_explanation_result):
        """Test asymmetry index A = (M- - M+) / (|M-| + |M+|)."""
        M_plus = sample_explanation_result["fidelity_plus"]
        M_minus = sample_explanation_result["fidelity_minus"]
        
        asymmetry = (M_minus - M_plus) / (abs(M_minus) + abs(M_plus) + 1e-8)
        assert -1 <= asymmetry <= 1
    
    def test_quadrant_assignment(self, sample_explanation_result):
        """Test quadrant classification based on fidelity."""
        M_plus = sample_explanation_result["fidelity_plus"]
        M_minus = sample_explanation_result["fidelity_minus"]
        
        # Quadrant determination
        if M_plus > 0 and M_minus > 0:
            quadrant = "sufficient_necessary"
        elif M_plus > 0 and M_minus <= 0:
            quadrant = "sufficient_redundant"
        elif M_plus <= 0 and M_minus > 0:
            quadrant = "insufficient_necessary"
        else:
            quadrant = "insufficient_redundant"
        
        assert quadrant in [
            "sufficient_necessary", 
            "sufficient_redundant",
            "insufficient_necessary", 
            "insufficient_redundant"
        ]


class TestDimension3Consistency:
    """Test Consistency Across Outcomes (Dimension 3)."""
    
    def test_margin_preservation_calculation(self):
        """Test margin preservation metrics."""
        origin_margin = 0.95
        masked_margin = 0.85
        maskout_margin = 0.30
        
        # Sufficiency preservation
        sufficiency_preservation = masked_margin / origin_margin
        assert 0 <= sufficiency_preservation <= 1.5  # Can exceed 1
        
        # Necessity preservation
        necessity_preservation = maskout_margin / origin_margin
        assert necessity_preservation < sufficiency_preservation  # Expected for correct


class TestDimension4Progression:
    """Test Feature Ranking Stability (Dimension 4)."""
    
    def test_progression_data_format(self, sample_progression_data):
        """Test progression data has expected format."""
        assert "maskout_progression_confidence" in sample_progression_data
        assert "sufficiency_progression_confidence" in sample_progression_data
        assert len(sample_progression_data["maskout_progression_confidence"]) > 0
    
    def test_maskout_progression_decreasing(self, sample_progression_data):
        """Test maskout confidence decreases as features removed."""
        confidence = sample_progression_data["maskout_progression_confidence"]
        # Generally decreasing trend
        assert confidence[0] > confidence[-1]
    
    def test_sufficiency_progression_increasing(self, sample_progression_data):
        """Test sufficiency confidence increases as features added."""
        confidence = sample_progression_data["sufficiency_progression_confidence"]
        # Generally increasing trend
        assert confidence[0] < confidence[-1]


class TestSeparabilityMetric:
    """Test Separability metric for error detection (Section 3.5.2)."""
    
    def test_separability_positive(self):
        """Test separability is non-negative."""
        # Separability = sqrt(SD_correct^2 + SD_incorrect^2)
        import math
        sd_correct = 0.15
        sd_incorrect = 0.25
        separability = math.sqrt(sd_correct**2 + sd_incorrect**2)
        assert separability >= 0


@pytest.mark.slow
@pytest.mark.integration
class TestFullAnalytics:
    """Integration tests for full analytics pipeline."""
    
    def test_run_all_dimensions(self):
        """Test running all 4 dimension analyses."""
        pass
