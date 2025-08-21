"""Base configuration for GBWM RL project"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Environment variables
DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
RANDOM_SEED = 42
USE_WANDB = os.environ.get("USE_WANDB", "false").lower() == "true"