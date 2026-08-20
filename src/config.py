import os
from pathlib import Path

# Base Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "dataset"
WEIGHTS_DIR = WORKSPACE_DIR / "weights"
SNAPSHOTS_DIR = WORKSPACE_DIR / "snapshots"
RUNS_DIR = WORKSPACE_DIR / "runs"

# Ensure essential directories exist
for directory in [DATASET_DIR, WEIGHTS_DIR, SNAPSHOTS_DIR, RUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default Dataset YAML path
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

# Default Model settings
DEFAULT_MODEL_NAME = "yolo11s.pt"
BEST_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
LAST_WEIGHTS_PATH = WEIGHTS_DIR / "last.pt"

# Mapping string labels to standard integer digits
# Supports common naming conventions from Roboflow datasets
NAME_TO_DIGIT = {
    "hand-zero": 0,
    "hand-0": 0,
    "zero": 0,
    "0": 0,
    "hand-one": 1,
    "hand-1": 1,
    "one": 1,
    "1": 1,
    "hand-two": 2,
    "hand-2": 2,
    "two": 2,
    "2": 2,
    "hand-three": 3,
    "hand-3": 3,
    "three": 3,
    "3": 3,
    "hand-four": 4,
    "hand-4": 4,
    "four": 4,
    "4": 4,
    "hand-five": 5,
    "hand-5": 5,
    "five": 5,
    "5": 5,
}

# Color palette for digits (0 to 5) in BGR format for OpenCV
DIGIT_COLORS = {
    0: (128, 128, 128),  # Gray
    1: (255, 99, 71),    # Tomato / Light Red
    2: (50, 205, 50),    # Lime Green
    3: (30, 144, 255),   # Dodger Blue
    4: (255, 165, 0),    # Orange
    5: (186, 85, 211),   # Medium Orchid / Purple
    -1: (200, 200, 200)  # Default fallback
}

# Inference defaults
DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_IOU_THRESHOLD = 0.45
SMOOTHING_WINDOW_SIZE = 4
