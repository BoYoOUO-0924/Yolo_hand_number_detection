import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_MODEL_NAME, DATASET_DIR, DATA_YAML_PATH
from src.dataset_manager import DatasetManager
from src.trainer import HandModelTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 Hand Digit Recognition Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Base YOLO model (e.g. yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index ('0') or 'cpu'")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience epochs (default: 25)")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to Roboflow dataset zip file to extract")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Check or extract dataset
    manager = DatasetManager(DATASET_DIR)
    if args.zip_path:
        print(f"[Dataset] Extracting dataset from {args.zip_path}...")
        success = manager.extract_zip(args.zip_path)
        if not success:
            sys.exit(1)
    elif not DATA_YAML_PATH.exists():
        print(f"[Notice] data.yaml not found at {DATA_YAML_PATH}.")
        print("Checking if any zip files exist in Downloads or workspace...")
        downloads_zip = Path.home() / "Downloads" / "Hand Detection.v1i.yolov11.zip"
        if downloads_zip.exists():
            print(f"Found dataset zip in Downloads: {downloads_zip}")
            manager.extract_zip(str(downloads_zip))
        else:
            print("Please specify --zip-path or place extracted dataset in 'dataset/' directory.")
            sys.exit(1)

    # 2. Check health
    health = manager.check_dataset_health()
    if not health.get("healthy"):
        print("[Error] Dataset health check failed.")
        sys.exit(1)

    # 3. Start training
    trainer = HandModelTrainer(
        model_name=args.model,
        data_yaml=DATA_YAML_PATH,
        device=args.device
    )

    best_weights = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        lr0=args.lr0,
        patience=args.patience
    )

    if best_weights:
        print(f"\n[Done] Training succeeded! Best model ready at: {best_weights}")
        print("You can now run real-time inference using: python main_infer.py")

if __name__ == "__main__":
    main()
