import os
import sys
import time
from typing import Optional, List, Tuple
from pathlib import Path
import cv2
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.hand_detector import DetectionResult, HandDetection

class HandVisualizer:
    """
    Renders high-quality HUD banners, stylized bounding boxes, and notifications on OpenCV frames.
    """

    def __init__(self):
        self.notification_text: Optional[str] = None
        self.notification_expire_time: float = 0.0

    def set_notification(self, text: str, duration_sec: float = 2.0):
        """Displays a temporary notification banner."""
        self.notification_text = text
        self.notification_expire_time = time.time() + duration_sec

    def draw_rounded_rectangle(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_radius: int = 10
    ):
        """Draws a rectangle with rounded corners or stylized corner accents."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw standard rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner accents for futuristic HUD look
        accent_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        accent_thickness = thickness + 2

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + accent_len, y1), color, accent_thickness)
        cv2.line(img, (x1, y1), (x1, y1 + accent_len), color, accent_thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - accent_len, y1), color, accent_thickness)
        cv2.line(img, (x2, y1), (x2, y1 + accent_len), color, accent_thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + accent_len, y2), color, accent_thickness)
        cv2.line(img, (x1, y2), (x1, y2 - accent_len), color, accent_thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - accent_len, y2), color, accent_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - accent_len), color, accent_thickness)

    def draw_hud(self, frame: np.ndarray, result: DetectionResult, is_paused: bool = False) -> np.ndarray:
        """
        Renders HUD overlays, bounding boxes, and status bars onto the frame.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Top HUD Banner (semi-transparent dark bar)
        hud_height = 70
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Bottom subtle gradient / border line
        cv2.line(frame, (0, hud_height), (w, hud_height), (70, 70, 80), 2)

        # 2. Draw Left Status (FPS & Latency)
        fps_text = f"FPS: {result.fps:.1f} | {result.inference_time_ms:.1f}ms"
        cv2.putText(frame, "HAND DIGIT RECOGNITION", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Draw Center Summary (Digits & Formula)
        if result.hand_count == 0:
            summary_text = "No Hands Detected"
            cv2.putText(frame, summary_text, (w // 2 - 110, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2, cv2.LINE_AA)
        else:
            if len(result.digits) == 1:
                digit_str = f"Digit: {result.digits[0]}"
                formula_str = f"[{result.digits[0]}]"
            else:
                formula_str = " + ".join([f"[{d}]" for d in result.digits]) + f" = {result.digit_sum}"
                digit_str = f"Total: {result.digit_sum}"

            # Calculate text size for centering
            (text_w, _), _ = cv2.getTextSize(formula_str, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cv2.putText(frame, formula_str, (max(180, (w - text_w) // 2), 44), cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 255, 120), 2, cv2.LINE_AA)

        # 4. Draw Right Controls Guide
        guide_lines = ["'s': Snapshot | 'c': Cam", "'space': Pause | 'q': Quit"]
        cv2.putText(frame, guide_lines[0], (w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, guide_lines[1], (w - 240, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # 5. Draw Detections (Bounding boxes & labels)
        for i, det in enumerate(result.detections):
            x1, y1, x2, y2 = det.bbox
            color = det.color

            # Draw bounding box
            self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            # Label text
            if det.digit >= 0:
                label_text = f"Digit {det.digit} ({det.confidence*100:.0f}%)"
            else:
                label_text = f"{det.raw_label} ({det.confidence*100:.0f}%)"

            # Label background pill
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lbl_y1 = max(hud_height + 5, y1 - th - 12)
            lbl_y2 = lbl_y1 + th + 10
            lbl_x2 = min(w, x1 + tw + 16)

            # Draw label box
            cv2.rectangle(frame, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
            cv2.putText(frame, label_text, (x1 + 8, lbl_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 6. Notifications / Pause overlay
        now = time.time()
        if is_paused:
            pause_text = "PAUSED (Press Space to Resume)"
            (ptw, _), _ = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            cv2.rectangle(frame, (w // 2 - ptw // 2 - 20, h // 2 - 25), (w // 2 + ptw // 2 + 20, h // 2 + 25), (0, 0, 0), -1)
            cv2.putText(frame, pause_text, (w // 2 - ptw // 2, h // 2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        elif self.notification_text and now < self.notification_expire_time:
            (ntw, _), _ = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (w // 2 - ntw // 2 - 15, h - 60), (w // 2 + ntw // 2 + 15, h - 20), (30, 140, 30), -1)
            cv2.putText(frame, self.notification_text, (w // 2 - ntw // 2, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
