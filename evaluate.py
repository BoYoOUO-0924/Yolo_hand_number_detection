import argparse
import sys
from pathlib import Path

from src.config import BEST_WEIGHTS_PATH, DATA_YAML_PATH
from src.evaluator import HandModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained YOLO Model on Dataset")
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS_PATH), help="Path to weights file (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0') or 'cpu'")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_file = Path(args.weights)

    if not weights_file.exists():
        print(f"[Error] Weights file not found: {weights_file}")
        print("Please train a model first using 'python train.py' or specify --weights <path>")
        sys.exit(1)

    evaluator = HandModelEvaluator(
        weights_path=weights_file,
        data_yaml=DATA_YAML_PATH,
        device=args.device
    )

    evaluator.evaluate(
        split=args.split,
        imgsz=args.imgsz,
        batch_size=args.batch
    )

if __name__ == "__main__":
    main()
