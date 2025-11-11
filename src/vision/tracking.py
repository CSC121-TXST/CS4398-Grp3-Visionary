"""
Object Tracking Module

This module handles tracking of detected objects across frames
to enable smooth camera movement and object following.

Integrates YOLOv8 for object detection and DeepSORT for multi-object tracking
with persistent IDs across frames.

Pulling Code from the following GITHUB Repo: 
https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking/blob/main/README.md
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterable
from ultralytics import YOLO

# Add DeepSORT module to path for imports
_deep_sort_path = os.path.join(os.path.dirname(__file__), "deep_sort_pytorch", "deep_sort_pytorch")
if _deep_sort_path not in sys.path:
    sys.path.insert(0, _deep_sort_path)

from deep_sort.deep_sort import DeepSort


class ObjectTracker:
    """
    ObjectTracker integrates YOLOv8 detection with DeepSORT tracking.
    
    This class provides a simple API for Visionary:
    - Uses YOLOv8 to detect objects in each frame
    - Uses DeepSORT to maintain persistent IDs across frames
    - Returns processed frames with boxes + IDs drawn
    - Returns tracking metadata for each detected object
    """

    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        conf: float = 0.35, 
        device: str = "cpu",
        deepsort_ckpt: Optional[str] = None,
        target_classes: Optional[Iterable[str]] = None,
    ):
        """
        Initialize the ObjectTracker with YOLOv8 and DeepSORT.
        
        Args:
            model_path: path or name of YOLO model (yolov8n.pt is small & fast)
            conf: confidence threshold for detections
            device: "cpu" or "cuda"
            deepsort_ckpt: path to DeepSORT checkpoint file (ckpt.t7)
                          If None, uses default path relative to deep_sort_pytorch
        """
        self.enabled = False  # Disabled by default
        self.conf = conf
        self.device = device
        self._set_target_classes(target_classes)

        # Load YOLO model for detection
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            # If device transfer fails, just stay on default
            pass

        # Initialize DeepSORT tracker
        # Default checkpoint path relative to deep_sort_pytorch module
        if deepsort_ckpt is None:
            deepsort_ckpt = os.path.join(
                os.path.dirname(__file__),
                "deep_sort_pytorch", "deep_sort_pytorch",
                "deep_sort", "deep", "checkpoint", "ckpt.t7"
            )
        
        # Check if checkpoint exists
        if not os.path.exists(deepsort_ckpt):
            raise FileNotFoundError(
                f"DeepSORT checkpoint not found at {deepsort_ckpt}\n"
                "Please ensure ckpt.t7 exists in the checkpoint directory."
            )
        
        # Initialize DeepSORT with default parameters
        # These can be tuned later if needed
        use_cuda = (device == "cuda")
        self.deepsort = DeepSort(
            model_path=deepsort_ckpt,
            max_dist=0.2,
            min_confidence=0.3,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
            use_cuda=use_cuda
        )

    def set_enabled(self, enabled: bool):
        """
        Allow Visionary to toggle tracking on/off from the UI.
        
        Args:
            enabled: True to enable tracking, False to disable
        """
        self.enabled = enabled
        print(f"ObjectTracker enabled set to: {self.enabled}")

    def _set_target_classes(self, target_classes: Optional[Iterable[str]]):
        """Store the target classes (case-insensitive) or None for all classes."""
        if target_classes:
            self.target_classes = {cls_name.lower() for cls_name in target_classes}
        else:
            self.target_classes = None

    def set_target_classes(self, target_classes: Optional[Iterable[str]]):
        """Public helper so the UI can adjust which classes we care about."""
        self._set_target_classes(target_classes)

    def _xyxy_to_xywh(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from (x1, y1, x2, y2) to (center_x, center_y, width, height).
        
        DeepSORT expects bboxes in xywh format (center coordinates).
        """
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w, h

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main entry point Visionary will use.
        
        Processes a frame through YOLOv8 detection and DeepSORT tracking.
        
        Args:
            frame_bgr: OpenCV BGR frame from the camera
            
        Returns:
            processed_frame: BGR image with boxes + IDs drawn
            tracked_objects: list of dicts with tracking info:
                [
                    {
                        "id": int,           # Persistent track ID
                        "cls": str,          # Class name (e.g., "person", "car")
                        "conf": float,       # Detection confidence
                        "bbox": (x1, y1, x2, y2)  # Bounding box coordinates
                    },
                    ...
                ]
        """
        if not self.enabled:
            return frame_bgr, []

        # Run YOLO inference (convert to RGB explicitly)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, conf=self.conf, verbose=False)
        result = results[0]

        # Extract detections from YOLO results
        names = result.names  # Maps class id -> class name
        
        # Prepare data for DeepSORT
        bbox_xywh = []  # List of (cx, cy, w, h) boxes
        confidences = []  # List of confidence scores
        oids = []  # List of object class IDs (for DeepSORT's oid parameter)
        filtered_boxes = []  # Keep references to YOLO boxes that passed filtering
        
        print(f"YOLO returned {len(result.boxes)} raw boxes")
        for box in result.boxes:
            # Get box coordinates in xyxy format
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get confidence and class ID
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))

            # Filter detections to requested classes (e.g., person only)
            if self.target_classes and cls_name.lower() not in self.target_classes:
                continue

            # Convert to xywh (center format) for DeepSORT
            cx, cy, w, h = self._xyxy_to_xywh(x1, y1, x2, y2)
            bbox_xywh.append([cx, cy, w, h])
            confidences.append(conf)
            oids.append(cls_id)
            filtered_boxes.append(box)

        print(f"Filtered {len(bbox_xywh)} boxes for classes: {self.target_classes if self.target_classes else 'ALL'}")
        if len(bbox_xywh) > 0:
            bbox_xywh_np = np.array(bbox_xywh)
            confidences_np = np.array(confidences)
            oids_np = np.array(oids)
            
            # Update DeepSORT tracker
            # Returns: array of [x1, y1, x2, y2, track_id, oid] for each tracked object
            # or empty list if no tracks
            tracked_outputs = self.deepsort.update(bbox_xywh_np, confidences_np, oids_np, frame_bgr)
        else:
            # No detections, so no tracks
            tracked_outputs = []
        print(f"DeepSORT outputs: {0 if tracked_outputs is None else len(tracked_outputs)} tracks")

        # Draw results on output frame
        output_frame = frame_bgr.copy()
        tracked_objects = []

        # Draw tracked objects with persistent IDs
        # tracked_outputs can be a list or numpy array
        if tracked_outputs is not None and len(tracked_outputs) > 0:
            for output in tracked_outputs:
                x1, y1, x2, y2, track_id, oid = output
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                oid = int(oid)
                
                # Get class name
                cls_name = names.get(oid, str(oid))
                
                # Find corresponding confidence (approximate - use first match)
                # In a more robust implementation, we'd track this better
                conf = 0.5  # Default confidence
                for i, (box, conf_val) in enumerate(zip(filtered_boxes, confidences)):
                    box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].tolist()
                    if abs(box_x1 - x1) < 5 and abs(box_y1 - y1) < 5:
                        conf = conf_val
                        break
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with track ID
                label = f"ID:{track_id} {cls_name} {conf:.2f}"
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                
                # Add to tracked objects list
                tracked_objects.append({
                    "id": track_id,
                    "cls": cls_name,
                    "conf": float(conf),
                    "bbox": (x1, y1, x2, y2),
                })

        # If DeepSORT did not return any tracks (e.g., first few frames), fall back to drawing YOLO detections
        if len(tracked_outputs) == 0 and filtered_boxes:
            for box, conf in zip(filtered_boxes, confidences):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = f"{cls_name} {float(conf):.2f}"
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return output_frame, tracked_objects
