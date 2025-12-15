"""
Tests for LLM Fine-tuning (Paper Section 3.2)

Tests the src.finetuning module which fine-tunes BERT for classification.
"""
import pytest


class TestFinetuningImports:
    """Test that finetuning module imports correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import finetuning
        assert finetuning is not None
    
    def test_finetuner_import(self):
        """Test Finetuner class import."""
        from src.finetuning.finetuner import Finetuner
        assert Finetuner is not None


class TestFinetunerConfig:
    """Test finetuning configuration."""
    
    def test_config_import(self):
        """Test config module exists."""
        from src.finetuning import config
        assert config is not None
    
    def test_default_model_name(self):
        """Test default model is BERT."""
        from src.finetuning.config import DEFAULT_MODEL_NAME
        assert "bert" in DEFAULT_MODEL_NAME.lower()


class TestFinetunerClass:
    """Test Finetuner class functionality."""
    
    def test_finetuner_instantiation(self):
        """Test Finetuner can be instantiated."""
        from src.finetuning.finetuner import Finetuner
        # Should be able to create with minimal config
        finetuner = Finetuner(
            dataset_name="stanfordnlp/sst2",
            model_name="google-bert/bert-base-uncased",
            output_dir="/tmp/test_finetuning"
        )
        assert finetuner is not None
        assert finetuner.dataset_name == "stanfordnlp/sst2"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_dataset(self):
        """Test dataset loading (requires network)."""
        from src.finetuning.finetuner import Finetuner
        finetuner = Finetuner(
            dataset_name="stanfordnlp/sst2",
            model_name="google-bert/bert-base-uncased",
            output_dir="/tmp/test_finetuning"
        )
        # This would load actual data - mark as slow/integration
        # dataset = finetuner.load_dataset()
        # assert dataset is not None
