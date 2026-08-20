import argparse
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

from src.config import (
    BEST_WEIGHTS_PATH,
    SNAPSHOTS_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    WORKSPACE_DIR
)
from src.hand_detector import HandDetector
from src.visualizer import HandVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Hand Digit Recognition & HUD")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0, 1, 2) or path to video/image file")
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS_PATH), help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold (default: 0.50)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU NMS threshold (default: 0.45)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0') or 'cpu'")
    parser.add_argument("--no-smooth", action="store_true", help="Disable temporal box smoothing")
    return parser.parse_args()

def run_image_inference(detector: HandDetector, visualizer: HandVisualizer, image_path: Path):
    """Runs inference on a single image and displays the result."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[Error] Could not read image at: {image_path}")
        return

    result = detector.detect(img)
    annotated = visualizer.draw_hud(img, result)

    out_path = SNAPSHOTS_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[Info] Detection result saved to: {out_path}")

    cv2.imshow("Hand Digit Recognition - Image", annotated)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_camera(index: int):
    """Attempts to open a camera using DirectShow, MSMF, or default backends."""
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("DEFAULT", cv2.CAP_ANY)
    ]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"[Camera] Successfully opened camera index {index} using backend {name}")
            return cap
    return None

def run_stream_inference(detector: HandDetector, visualizer: HandVisualizer, source: str):
    """Runs real-time inference on webcam stream or video file."""
    # Determine camera index vs video file
    is_webcam = source.isdigit()
    
    if is_webcam:
        cam_index = int(source)
        cap = open_camera(cam_index)
        if cap is None:
            # Probe other indices
            for alt_idx in [1, 2, 0]:
                if alt_idx != cam_index:
                    cap = open_camera(alt_idx)
                    if cap is not None:
                        cam_index = alt_idx
                        break
        if cap is None:
            print(f"\n[Error] No available webcam found on indices (0, 1, 2).")
            print("[Tip] If you don't have a physical webcam connected, you can test images using:")
            print("      python main_infer.py --source dataset/test/images/<any_image>.jpg\n")
            return
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Error] Failed to open video source: {source}")
            return

    # Attempt to set HD resolution for webcam
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "YOLO11 Hand Digit Recognition HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "="*55)
    print("           REAL-TIME RECOGNITION STARTED          ")
    print("="*55)
    print(" Hotkeys:")
    print("   [q] / [ESC]  : Quit")
    print("   [s]          : Save Snapshot")
    print("   [c]          : Switch Camera")
    print("   [Space]      : Pause / Resume")
    print("   [+] / [-]    : Adjust Confidence Threshold (+/- 0.05)")
    print("="*55 + "\n")

    is_paused = False
    last_frame = None
    last_result = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    # Loop video if reached the end
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Warning] Failed to grab frame from camera.")
                    break

            last_frame = frame.copy()
            last_result = detector.detect(frame)

        if last_frame is not None and last_result is not None:
            display_frame = last_frame.copy()
            display_frame = visualizer.draw_hud(display_frame, last_result, is_paused=is_paused)
            cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1 if not is_paused else 30) & 0xFF

        # [q] or [ESC]: Quit
        if key == ord('q') or key == 27:
            print("[Info] Exiting...")
            break

        # [Space]: Pause / Resume
        elif key == ord(' '):
            is_paused = not is_paused

        # [s]: Save Snapshot
        elif key == ord('s') and last_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            shot_file = SNAPSHOTS_DIR / f"snapshot_{timestamp}.jpg"
            annotated_frame = visualizer.draw_hud(last_frame.copy(), last_result)
            cv2.imwrite(str(shot_file), annotated_frame)
            msg = f"Saved: {shot_file.name}"
            visualizer.set_notification(msg)
            print(f"[Snapshot] {shot_file}")

        # [c]: Switch camera
        elif key == ord('c') and is_webcam:
            cam_index = (cam_index + 1) % 4
            print(f"[Info] Switching to Camera Index: {cam_index}...")
            cap.release()
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                print(f"[Warning] Camera {cam_index} unavailable. Reverting to 0.")
                cam_index = 0
                cap = cv2.VideoCapture(0)
            visualizer.set_notification(f"Switched to Camera {cam_index}")

        # [+] / [-]: Adjust Confidence
        elif key == ord('+') or key == ord('='):
            detector.conf_threshold = min(0.95, detector.conf_threshold + 0.05)
            visualizer.set_notification(f"Confidence: {detector.conf_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            detector.conf_threshold = max(0.10, detector.conf_threshold - 0.05)
            visualizer.set_notification(f"Confidence: {detector.conf_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"[Warning] Model weights not found at: {weights_path}")
        print("Using base 'yolo11s.pt' as fallback. Please train the model with 'python train.py' for custom hand digits.")
        weights_to_load = "yolo11s.pt"
    else:
        weights_to_load = str(weights_path)

    detector = HandDetector(
        weights_path=weights_to_load,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_smoothing=not args.no_smooth,
        device=args.device
    )

    visualizer = HandVisualizer()

    # Check if source is image file
    source_path = Path(args.source)
    if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"] and source_path.exists():
        run_image_inference(detector, visualizer, source_path)
    else:
        run_stream_inference(detector, visualizer, args.source)

if __name__ == "__main__":
    main()
