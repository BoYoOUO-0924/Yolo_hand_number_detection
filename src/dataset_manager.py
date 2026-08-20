import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import yaml
from src.config import DATASET_DIR, DATA_YAML_PATH, WORKSPACE_DIR

class DatasetManager:
    """
    Handles unzipping, path sanitization, and verification of YOLO datasets.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.yaml_path = self.dataset_dir / "data.yaml"

    def extract_zip(self, zip_path: str, clean_existing: bool = False) -> bool:
        """
        Extracts a YOLO dataset zip file into the dataset directory.
        """
        zip_file = Path(zip_path).resolve()
        if not zip_file.exists():
            print(f"[Error] Zip file not found: {zip_file}")
            return False

        if clean_existing and self.dataset_dir.exists():
            print(f"[Info] Cleaning existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir, ignore_errors=True)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Info] Extracting '{zip_file.name}' to '{self.dataset_dir}'...")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
            print(f"[Success] Extracted {len(zip_ref.namelist())} files.")
            self.fix_yaml_paths()
            return True
        except Exception as e:
            print(f"[Error] Failed to extract zip: {e}")
            return False

    def fix_yaml_paths(self) -> Optional[Path]:
        """
        Ensures data.yaml exists and paths for train/val/test are correctly resolved.
        """
        # Find data.yaml in dataset_dir or nested subdirectory
        candidates = list(self.dataset_dir.glob("**/data.yaml"))
        if not candidates:
            print(f"[Warning] data.yaml not found in {self.dataset_dir}")
            return None

        found_yaml = candidates[0]
        root_dir = found_yaml.parent

        with open(found_yaml, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Set path to absolute dataset directory to prevent relative path ambiguity in YOLO
        data['path'] = str(root_dir.resolve()).replace('\\', '/')

        # Ensure train/val/test keys exist properly
        for split, possible_folders in [
            ('train', ['train/images', 'train', 'images/train']),
            ('val', ['valid/images', 'val/images', 'valid', 'val', 'images/val']),
            ('test', ['test/images', 'test', 'images/test'])
        ]:
            if split in data:
                val_path = data[split]
                # If relative, check if folder exists
                if not (root_dir / val_path).exists():
                    for folder in possible_folders:
                        if (root_dir / folder).exists():
                            data[split] = folder
                            break
            else:
                for folder in possible_folders:
                    if (root_dir / folder).exists():
                        data[split] = folder
                        break

        # Save corrected data.yaml in root dataset_dir
        target_yaml = self.dataset_dir / "data.yaml"
        with open(target_yaml, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        print(f"[Success] data.yaml configured at: {target_yaml}")
        self.yaml_path = target_yaml
        return target_yaml

    def check_dataset_health(self) -> Dict[str, Any]:
        """
        Inspects the dataset and prints summary statistics.
        """
        if not self.yaml_path.exists():
            self.fix_yaml_paths()

        if not self.yaml_path.exists():
            print(f"[Error] No data.yaml found in {self.dataset_dir}")
            return {"healthy": False}

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        classes = data.get('names', [])
        num_classes = data.get('nc', len(classes))

        print("\n" + "="*50)
        print("          DATASET HEALTH & SUMMARY REPORT         ")
        print("="*50)
        print(f"Dataset Root: {self.dataset_dir}")
        print(f"Classes ({num_classes}): {classes}")

        stats = {
            "healthy": True,
            "classes": classes,
            "nc": num_classes,
            "splits": {}
        }

        # Check image and label counts
        for split_key, split_name in [('train', 'Train'), ('val', 'Valid'), ('test', 'Test')]:
            split_rel = data.get(split_key)
            if not split_rel:
                continue

            img_dir = self.dataset_dir / split_rel
            if not img_dir.exists():
                # Try resolving relative to yaml
                img_dir = self.yaml_path.parent / split_rel

            if img_dir.exists():
                image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
                stats["splits"][split_name] = len(image_files)
                print(f"- {split_name} Split: {len(image_files)} images found at '{img_dir.relative_to(WORKSPACE_DIR) if img_dir.is_relative_to(WORKSPACE_DIR) else img_dir}'")
            else:
                stats["splits"][split_name] = 0
                print(f"- {split_name} Split: [Not Found] Directory: {img_dir}")

        print("="*50 + "\n")
        return stats


if __name__ == "__main__":
    manager = DatasetManager()
    if len(sys.argv) > 1:
        zip_arg = sys.argv[1]
        manager.extract_zip(zip_arg)
    manager.check_dataset_health()
