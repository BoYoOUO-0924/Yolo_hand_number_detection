import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO
from src.config import DATA_YAML_PATH, BEST_WEIGHTS_PATH, RUNS_DIR

class HandModelEvaluator:
    """
    Evaluates a trained YOLO model on the validation or test split.
    """

    def __init__(
        self,
        weights_path: Path = BEST_WEIGHTS_PATH,
        data_yaml: Path = DATA_YAML_PATH,
        device: str = "0"
    ):
        self.weights_path = Path(weights_path)
        self.data_yaml = Path(data_yaml)
        self.device = device

    def evaluate(self, split: str = "val", imgsz: int = 640, batch_size: int = 16) -> Dict[str, Any]:
        """
        Runs validation and prints performance metrics.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found at: {self.weights_path}")

        print(f"\n[Info] Loading model weights from: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        print(f"[Info] Evaluating on split: '{split}'...")
        metrics = model.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            plots=True,
            verbose=True
        )

        print("\n" + "="*50)
        print(f"       VALIDATION METRICS ({split.upper()} SET)      ")
        print("="*50)
        print(f"mAP@50:       {metrics.box.map50 * 100:.2f}%")
        print(f"mAP@50-95:    {metrics.box.map * 100:.2f}%")
        print(f"Precision:    {metrics.box.mp * 100:.2f}%")
        print(f"Recall:       {metrics.box.mr * 100:.2f}%")
        print("="*50 + "\n")

        return {
            "map50": metrics.box.map50,
            "map": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr
        }
