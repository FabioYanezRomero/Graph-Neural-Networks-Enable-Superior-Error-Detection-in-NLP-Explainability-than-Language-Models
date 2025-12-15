"""
Configuration settings for the Model_Finetuning module.
"""

# Default device configuration
DEFAULT_DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# Model configuration
DEFAULT_MODEL_NAME = 'google-bert/bert-base-uncased'
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128
DEFAULT_NUM_EPOCHS = 5

# Optimizer configuration
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ADAM_EPSILON = 1e-8
DEFAULT_WARMUP_PROPORTION = 0.1

# Training configuration
DEFAULT_FP16 = True
DEFAULT_SCHEDULER = 'linear'
DEFAULT_SEED = 42

# Output configuration
DEFAULT_OUTPUT_DIR = 'outputs/llm'

# Dataset configuration
DEFAULT_DATASET_NAME = 'snli'
