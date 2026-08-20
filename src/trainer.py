import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO
from src.config import (
    DATA_YAML_PATH, 
    WEIGHTS_DIR, 
    RUNS_DIR, 
    DEFAULT_MODEL_NAME,
    BEST_WEIGHTS_PATH,
    LAST_WEIGHTS_PATH
)
from src.dataset_manager import DatasetManager

class HandModelTrainer:
    """
    Handles YOLO11 training for hand digit recognition.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        data_yaml: Path = DATA_YAML_PATH,
        device: str = "0"  # '0' for GPU 0, or 'cpu'
    ):
        self.model_name = model_name
        self.data_yaml = Path(data_yaml)
        self.device = device
        self.weights_dir = WEIGHTS_DIR
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        lr0: float = 0.01,
        project_name: str = "hand_detection",
        experiment_name: str = "yolo11_hand_digits",
        patience: int = 25,
        workers: int = 4
    ) -> Optional[Path]:
        """
        Starts the YOLO training process.
        """
        # Ensure dataset is ready
        manager = DatasetManager(self.data_yaml.parent)
        health = manager.check_dataset_health()
        if not health.get("healthy"):
            raise RuntimeError(f"Dataset at {self.data_yaml} is not ready or valid.")

        print(f"\n[Info] Initializing YOLO model: {self.model_name}...")
        model = YOLO(self.model_name)

        save_dir = RUNS_DIR / project_name / experiment_name

        print("\n" + "="*50)
        print("              STARTING YOLO TRAINING              ")
        print("="*50)
        print(f"Model:           {self.model_name}")
        print(f"Dataset YAML:    {self.data_yaml}")
        print(f"Epochs:          {epochs}")
        print(f"Batch Size:      {batch_size}")
        print(f"Image Size:      {imgsz}")
        print(f"Device:          {self.device}")
        print(f"Runs Directory:  {save_dir}")
        print("="*50 + "\n")

        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            device=self.device,
            project=str(RUNS_DIR / project_name),
            name=experiment_name,
            patience=patience,
            workers=workers,
            save=True,
            exist_ok=True,
            plots=True,
            verbose=True
        )

        # Locate trained weights
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        if best_pt.exists():
            shutil.copy2(best_pt, BEST_WEIGHTS_PATH)
            print(f"\n[Success] Best weights saved to: {BEST_WEIGHTS_PATH}")
        if last_pt.exists():
            shutil.copy2(last_pt, LAST_WEIGHTS_PATH)

        print("\n[Success] Training complete! Results saved in:", save_dir)
        return BEST_WEIGHTS_PATH if BEST_WEIGHTS_PATH.exists() else None
