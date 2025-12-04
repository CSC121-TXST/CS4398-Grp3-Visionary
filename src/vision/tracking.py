"""
Object Tracking Module

This module handles tracking of detected objects across frames to enable smooth
camera movement and object following. It integrates YOLOv8 for object detection
and DeepSORT for multi-object tracking with persistent IDs across frames.

Architecture:
    Camera Frame → YOLOv8 Detection → Class Filtering → DeepSORT Tracking → Annotated Frame

The tracking pipeline processes frames through:
1. YOLOv8: Detects objects in each frame
2. Class Filtering: Filters detections to user-selected classes (e.g., person, cell phone)
3. DeepSORT: Maintains persistent track IDs across frames using Kalman filtering and feature matching
4. Visualization: Draws bounding boxes and track IDs on the output frame

Reference Implementation:
    Based on: https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterable
from ultralytics import YOLO

# === DeepSORT Module Path Setup ===
# Add DeepSORT module to Python path for imports
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
        debug: bool = False,
        process_interval: int = 2,
        imgsz: int = 640,
    ):
        """
        Initialize the ObjectTracker with YOLOv8 and DeepSORT.
        
        Args:
            model_path: Path or name of YOLO model (yolov8n.pt is small & fast)
            conf: Confidence threshold for detections (0.0-1.0)
            device: "cpu" or "cuda" for model inference
            deepsort_ckpt: Path to DeepSORT checkpoint file (ckpt.t7).
                          If None, uses default path relative to deep_sort_pytorch
            target_classes: List of class names to track (e.g., ["person", "cell phone"]).
                           If None, tracks all detected classes
            debug: Enable debug print statements
            process_interval: Process every Nth frame (1 = every frame, 2 = every other frame)
            imgsz: Image size for YOLO inference (larger = more accurate but slower)
        
        Raises:
            FileNotFoundError: If DeepSORT checkpoint file is not found
            RuntimeError: If YOLO model fails to load
        """
        # === State Initialization ===
        self.enabled = False  # Disabled by default until UI enables it
        self.conf = conf
        self.device = device
        self._set_target_classes(target_classes)
        self.debug = debug
        self.process_interval = max(1, int(process_interval))  # Ensure at least 1
        self.imgsz = int(imgsz)
        
        # Frame processing state
        self._frame_counter = 0
        self._last_output_frame = None  # Cached frame for skipped frames
        self._last_tracked_count = 0

        # === YOLO Model Initialization ===
        try:
            self.model = YOLO(model_path)
            # Move model to specified device (CPU or CUDA)
            try:
                self.model.to(device)
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not move YOLO model to {device}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

        # === DeepSORT Tracker Initialization ===
        # Resolve checkpoint path (default or user-provided)
        if deepsort_ckpt is None:
            deepsort_ckpt = os.path.join(
                os.path.dirname(__file__),
                "deep_sort_pytorch", "deep_sort_pytorch",
                "deep_sort", "deep", "checkpoint", "ckpt.t7"
            )
        
        # Validate checkpoint exists
        if not os.path.exists(deepsort_ckpt):
            raise FileNotFoundError(
                f"DeepSORT checkpoint not found at {deepsort_ckpt}\n"
                "Please ensure ckpt.t7 exists in the checkpoint directory."
            )
        
        # Initialize DeepSORT with tuned parameters
        # These parameters control tracking behavior:
        # - max_dist: Maximum cosine distance for feature matching (lower = stricter)
        # - max_iou_distance: Maximum IoU distance for track association
        # - max_age: Frames to keep a track after it's lost
        # - n_init: Frames required to confirm a new track
        use_cuda = (device == "cuda")
        try:
            self.deepsort = DeepSort(
                model_path=deepsort_ckpt,
                max_dist=0.2,          # Feature matching threshold
                min_confidence=0.3,    # Minimum detection confidence for DeepSORT
                nms_max_overlap=1.0,   # Non-maximum suppression overlap threshold
                max_iou_distance=0.7,   # Maximum IoU distance for track association
                max_age=70,            # Frames to keep lost tracks
                n_init=3,              # Frames needed to confirm a track
                nn_budget=100,         # Feature budget for appearance matching
                use_cuda=use_cuda
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DeepSORT tracker: {e}")

    def set_enabled(self, enabled: bool):
        """
        Allow Visionary to toggle tracking on/off from the UI.
        
        Args:
            enabled: True to enable tracking, False to disable
        """
        self.enabled = enabled
        if self.debug:
            print(f"ObjectTracker enabled set to: {self.enabled}")

    def set_debug(self, enabled: bool):
        """Enable/disable debug prints."""
        self.debug = bool(enabled)

    def set_process_interval(self, interval: int):
        self.process_interval = max(1, int(interval))

    def set_imgsz(self, imgsz: int):
        self.imgsz = int(imgsz)

    def set_conf(self, conf: float):
        try:
            self.conf = float(conf)
        except Exception:
            pass

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
        Main entry point for frame processing.
        
        Processes a frame through YOLOv8 detection and DeepSORT tracking pipeline.
        Handles frame skipping for performance and error recovery.
        
        Args:
            frame_bgr: OpenCV BGR frame from the camera (numpy array)
            
        Returns:
            Tuple of (processed_frame, tracked_objects):
                - processed_frame: BGR image with bounding boxes and track IDs drawn
                - tracked_objects: List of dicts with tracking info:
                    [
                        {
                            "id": int,           # Persistent track ID
                            "cls": str,          # Class name (e.g., "person", "car")
                            "conf": float,       # Detection confidence (0.0-1.0)
                            "bbox": (x1, y1, x2, y2)  # Bounding box coordinates
                        },
                        ...
                    ]
        
        Note:
            Returns empty list and original frame if tracking is disabled or frame is skipped.
        """
        # === Early Exit: Tracking Disabled ===
        if not self.enabled:
            return frame_bgr, []

        # === Frame Skipping for Performance ===
        # Process only every Nth frame to improve FPS
        self._frame_counter += 1
        should_process = (self._frame_counter % self.process_interval) == 0
        if not should_process and self._last_output_frame is not None:
            # Return cached frame for skipped frames
            return self._last_output_frame, []

        # === Error Handling: Validate Input Frame ===
        if frame_bgr is None or frame_bgr.size == 0:
            if self.debug:
                print("Warning: Received empty frame, returning cached frame")
            return self._last_output_frame if self._last_output_frame is not None else frame_bgr, []

        try:
            # === Step 1: YOLO Object Detection ===
            # Convert BGR to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb, conf=self.conf, verbose=False, imgsz=self.imgsz)
            result = results[0]

            # Extract class name mapping
            names = result.names  # Maps class id -> class name
            
            if self.debug:
                # Print available classes on first run
                if not hasattr(self, "_printed_classes"):
                    print(f"DEBUG: Available YOLO classes: {list(names.values())}")
                    self._printed_classes = True
            
            # === Step 2: Filter and Prepare Detections ===
            filtered_data = self._filter_and_prepare_detections(result.boxes, names)
            bbox_xywh, confidences, oids, filtered_boxes = filtered_data

            if self.debug:
                print(f"Filtered {len(bbox_xywh)} boxes for classes: {self.target_classes if self.target_classes else 'ALL'}")

            # === Step 3: DeepSORT Tracking Update ===
            tracked_outputs = self._update_tracker(bbox_xywh, confidences, oids, frame_bgr)

            if self.debug:
                track_count = 0 if tracked_outputs is None else len(tracked_outputs)
                print(f"DeepSORT outputs: {track_count} tracks")

            # === Step 4: Draw Results on Frame ===
            output_frame, tracked_objects = self._draw_tracking_results(
                frame_bgr, tracked_outputs, filtered_boxes, confidences, names
            )

            # === Cache Results for Skipped Frames ===
            self._last_output_frame = output_frame
            self._last_tracked_count = len(tracked_objects)
            
            return output_frame, tracked_objects

        except Exception as e:
            # Error recovery: return cached frame or original frame
            if self.debug:
                print(f"Error processing frame: {e}")
            return self._last_output_frame if self._last_output_frame is not None else frame_bgr, []
    
    def _filter_and_prepare_detections(self, boxes, names) -> Tuple[List, List, List, List]:
        """
        Filter YOLO detections by class and confidence, prepare for DeepSORT.
        
        Args:
            boxes: YOLO detection boxes
            names: Class ID to name mapping
        
        Returns:
            Tuple of (bbox_xywh, confidences, oids, filtered_boxes):
                - bbox_xywh: List of [cx, cy, w, h] bounding boxes (center format)
                - confidences: List of confidence scores
                - oids: List of object class IDs
                - filtered_boxes: List of YOLO box objects that passed filtering
        """
        bbox_xywh = []      # List of (cx, cy, w, h) boxes for DeepSORT
        confidences = []   # List of confidence scores
        oids = []          # List of object class IDs
        filtered_boxes = []  # Keep references to YOLO boxes for later use
        
        if self.debug:
            print(f"YOLO returned {len(boxes)} raw boxes")
        
        for box in boxes:
            # Extract box coordinates in xyxy format (top-left, bottom-right)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Extract confidence and class ID
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))

            # === Class Filtering ===
            # Normalize class name for case-insensitive matching
            cls_name_normalized = cls_name.lower().strip()
            
            # Filter by target classes if specified
            if self.target_classes and cls_name_normalized not in self.target_classes:
                if self.debug:
                    print(f"DEBUG: Filtered out {cls_name} (not in target classes)")
                continue
            
            # === Confidence Filtering ===
            # Double-check confidence threshold (YOLO already filters, but be safe)
            if conf < self.conf:
                if self.debug:
                    print(f"DEBUG: Filtered out {cls_name} due to low confidence: {conf:.2f} < {self.conf}")
                continue
            
            if self.debug:
                print(f"DEBUG: Keeping detection: {cls_name} (conf: {conf:.2f})")

            # Convert to xywh (center format) for DeepSORT
            cx, cy, w, h = self._xyxy_to_xywh(x1, y1, x2, y2)
            bbox_xywh.append([cx, cy, w, h])
            confidences.append(conf)
            oids.append(cls_id)
            filtered_boxes.append(box)

        return bbox_xywh, confidences, oids, filtered_boxes
    
    def _update_tracker(self, bbox_xywh: List, confidences: List, oids: List, frame_bgr: np.ndarray):
        """
        Update DeepSORT tracker with new detections.
        
        Args:
            bbox_xywh: List of [cx, cy, w, h] bounding boxes
            confidences: List of confidence scores
            oids: List of object class IDs
            frame_bgr: Original BGR frame for feature extraction
        
        Returns:
            Array of [x1, y1, x2, y2, track_id, oid] for each tracked object,
            or empty list if no tracks
        """
        if len(bbox_xywh) == 0:
            # No detections, so no tracks
            return []
        
        # Convert to numpy arrays for DeepSORT
        bbox_xywh_np = np.array(bbox_xywh)
        confidences_np = np.array(confidences)
        oids_np = np.array(oids)
        
        # Update DeepSORT tracker
        # Returns: array of [x1, y1, x2, y2, track_id, oid] for each tracked object
        tracked_outputs = self.deepsort.update(bbox_xywh_np, confidences_np, oids_np, frame_bgr)
        
        # Handle None return (shouldn't happen, but be safe)
        if tracked_outputs is None:
            return []
        
        return tracked_outputs
    
    def _draw_tracking_results(
        self, 
        frame_bgr: np.ndarray, 
        tracked_outputs: List, 
        filtered_boxes: List, 
        confidences: List,
        names: Dict
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Draw tracking results on the frame and build tracked objects list.
        
        Args:
            frame_bgr: Original BGR frame
            tracked_outputs: DeepSORT tracking outputs [x1, y1, x2, y2, track_id, oid]
            filtered_boxes: YOLO boxes that passed filtering
            confidences: Confidence scores for filtered boxes
            names: Class ID to name mapping
        
        Returns:
            Tuple of (output_frame, tracked_objects):
                - output_frame: Frame with bounding boxes and labels drawn
                - tracked_objects: List of tracking metadata dicts
        """
        output_frame = frame_bgr.copy()
        tracked_objects = []

        # === Color Map for Class Visualization (BGR format) ===
        class_colors = {
            "person": (0, 255, 0),        # Green
            "cell phone": (255, 0, 0),    # Blue
            "dog": (0, 165, 255),         # Orange
            "cat": (255, 20, 147),        # Deep pink
            "book": (255, 255, 0),        # Cyan
            "backpack": (128, 0, 128),     # Purple
        }

        # === Draw Tracked Objects with Persistent IDs ===
        if tracked_outputs is not None and len(tracked_outputs) > 0:
            for output in tracked_outputs:
                x1, y1, x2, y2, track_id, oid = output
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                oid = int(oid)
                
                # Get class name from ID
                cls_name = names.get(oid, str(oid))
                
                # Find corresponding confidence by matching box coordinates
                # (Approximate matching - in production, track this more robustly)
                conf = 0.5  # Default confidence fallback
                for box, conf_val in zip(filtered_boxes, confidences):
                    box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].tolist()
                    # Match if boxes are close (within 5 pixels)
                    if abs(box_x1 - x1) < 5 and abs(box_y1 - y1) < 5:
                        conf = conf_val
                        break
                
                # Draw bounding box and label
                self._draw_detection_box(output_frame, x1, y1, x2, y2, track_id, cls_name, conf, class_colors)
                
                # Add to tracked objects list
                tracked_objects.append({
                    "id": track_id,
                    "cls": cls_name,
                    "conf": float(conf),
                    "bbox": (x1, y1, x2, y2),
                })

        # === Fallback: Draw YOLO Detections if No Tracks ===
        # This handles cases where DeepSORT hasn't confirmed tracks yet (first few frames)
        if len(tracked_outputs) == 0 and filtered_boxes:
            for box, conf in zip(filtered_boxes, confidences):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw without track ID (detection only)
                label = f"{cls_name} {float(conf):.2f}"
                color = class_colors.get(cls_name.lower(), (0, 255, 255))  # Default to cyan
                self._draw_detection_box(output_frame, x1, y1, x2, y2, None, cls_name, conf, class_colors, label_override=label)

        return output_frame, tracked_objects
    
    def _draw_detection_box(
        self, 
        frame: np.ndarray, 
        x1: int, y1: int, x2: int, y2: int, 
        track_id: Optional[int], 
        cls_name: str, 
        conf: float,
        class_colors: Dict,
        label_override: Optional[str] = None
    ):
        """
        Draw a bounding box and label on the frame.
        
        Args:
            frame: BGR frame to draw on (modified in-place)
            x1, y1, x2, y2: Bounding box coordinates
            track_id: Optional track ID to include in label
            cls_name: Class name
            conf: Confidence score
            class_colors: Color mapping for classes
            label_override: Optional custom label (if provided, track_id is ignored)
        """
        # Get color for this class (default to cyan for unknown classes)
        color = class_colors.get(cls_name.lower(), (0, 255, 255))
        thickness = 2
        
        # Draw bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Build label text
        if label_override:
            label = label_override
        elif track_id is not None:
            label = f"ID:{track_id} {cls_name} {conf:.2f}"
        else:
            label = f"{cls_name} {conf:.2f}"
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for better text visibility
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text in white for contrast
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA,
        )
    
    def get_tracking_stats(self) -> Dict:
        """
        Get current tracking statistics.
        
        Returns:
            Dictionary with tracking statistics:
            {
                "tracked_count": int,      # Number of currently tracked objects
                "enabled": bool,            # Whether tracking is enabled
                "process_interval": int,    # Frame processing interval
                "imgsz": int,               # Image size for inference
                "conf": float,              # Confidence threshold
                "target_classes": list or None  # Target classes or None for all
            }
        """
        return {
            "tracked_count": self._last_tracked_count,
            "enabled": self.enabled,
            "process_interval": self.process_interval,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "target_classes": list(self.target_classes) if self.target_classes else None
        }
