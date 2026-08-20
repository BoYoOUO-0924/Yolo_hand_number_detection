import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import numpy as np
import cv2

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO
from src.config import (
    BEST_WEIGHTS_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    NAME_TO_DIGIT,
    DIGIT_COLORS,
    SMOOTHING_WINDOW_SIZE
)

@dataclass
class HandDetection:
    """Represents a single detected hand."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    raw_label: str
    digit: int  # Mapped digit integer (0~5), or -1 if unknown
    color: Tuple[int, int, int]  # BGR

@dataclass
class DetectionResult:
    """Result of hand detection for a single frame."""
    detections: List[HandDetection] = field(default_factory=list)
    hand_count: int = 0
    digits: List[int] = field(default_factory=list)
    digit_sum: int = 0
    fps: float = 0.0
    inference_time_ms: float = 0.0


class TemporalSmoother:
    """
    Stabilizes detections across consecutive frames to reduce flickering and jitter.
    """
    def __init__(self, window_size: int = SMOOTHING_WINDOW_SIZE):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def update(self, detections: List[HandDetection]) -> List[HandDetection]:
        if not detections:
            self.history.append([])
            return []

        # Sort detections from left to right (by center x)
        sorted_dets = sorted(detections, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
        self.history.append(sorted_dets)

        # If only 1 frame in history, return current
        if len(self.history) < 2:
            return sorted_dets

        # Temporal smoothing on bounding box coordinates for matching hands
        smoothed_detections = []
        for i, det in enumerate(sorted_dets):
            x1_list, y1_list, x2_list, y2_list = [det.bbox[0]], [det.bbox[1]], [det.bbox[2]], [det.bbox[3]]
            
            # Look at past frames for matching index or nearest bounding box
            for past_dets in list(self.history)[:-1]:
                if i < len(past_dets):
                    past = past_dets[i]
                    x1_list.append(past.bbox[0])
                    y1_list.append(past.bbox[1])
                    x2_list.append(past.bbox[2])
                    y2_list.append(past.bbox[3])

            smooth_x1 = int(np.mean(x1_list))
            smooth_y1 = int(np.mean(y1_list))
            smooth_x2 = int(np.mean(x2_list))
            smooth_y2 = int(np.mean(y2_list))

            smoothed_detections.append(
                HandDetection(
                    bbox=(smooth_x1, smooth_y1, smooth_x2, smooth_y2),
                    confidence=det.confidence,
                    class_id=det.class_id,
                    raw_label=det.raw_label,
                    digit=det.digit,
                    color=det.color
                )
            )

        return smoothed_detections


class HandDetector:
    """
    Encapsulates YOLO inference, class mapping, smoothing, and multi-hand arithmetic.
    """

    def __init__(
        self,
        weights_path: str = str(BEST_WEIGHTS_PATH),
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        enable_smoothing: bool = True,
        device: str = "0"
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_smoothing = enable_smoothing
        self.device = device

        print(f"[Info] Loading HandDetector weights from: {weights_path}")
        self.model = YOLO(weights_path)
        self.names = self.model.names if hasattr(self.model, 'names') else {}
        self.smoother = TemporalSmoother()

        # FPS calculation
        self._prev_time = time.time()

    def parse_digit(self, label: str) -> int:
        """
        Parses digit from raw class name (e.g. 'Hand-five' -> 5, 'hand_3' -> 3, '4' -> 4).
        """
        cleaned = label.strip().lower().replace("_", "-").replace(" ", "-")
        if cleaned in NAME_TO_DIGIT:
            return NAME_TO_DIGIT[cleaned]
        # Check if contains digit substring
        for char in cleaned:
            if char.isdigit():
                return int(char)
        return -1

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Performs detection on a single frame (BGR numpy image).
        """
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections: List[HandDetection] = []

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_label = self.names.get(cls_id, str(cls_id))
                digit = self.parse_digit(raw_label)
                color = DIGIT_COLORS.get(digit, DIGIT_COLORS[-1])

                detections.append(
                    HandDetection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        raw_label=raw_label,
                        digit=digit,
                        color=color
                    )
                )

        # Apply temporal smoothing if enabled
        if self.enable_smoothing:
            detections = self.smoother.update(detections)
        else:
            # Sort left to right
            detections = sorted(detections, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)

        # Calculate FPS
        current_time = time.time()
        inference_time_ms = (current_time - start_time) * 1000.0
        fps = 1.0 / (current_time - self._prev_time) if (current_time - self._prev_time) > 0 else 0.0
        self._prev_time = current_time

        # Calculate digits and sum
        valid_digits = [d.digit for d in detections if d.digit >= 0]
        digit_sum = sum(valid_digits)

        return DetectionResult(
            detections=detections,
            hand_count=len(detections),
            digits=valid_digits,
            digit_sum=digit_sum,
            fps=fps,
            inference_time_ms=inference_time_ms
        )
