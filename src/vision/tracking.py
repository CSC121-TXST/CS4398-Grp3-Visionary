"""
Object Tracking Module

This module handles tracking of detected objects across frames
to enable smooth camera movement and object following.

Pulling Code from the following GITHUB Repo: 
https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking/blob/main/README.md
"""

import cv2
from ultralytics import YOLO


class ObjectTracker:
    """
    Simple wrapper around YOLOv8 for object detection (first step).
    Later we can plug DeepSORT in here for ID-based tracking.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.5, device: str = "cpu"):
        """
        model_path: path or name of YOLO model (yolov8n.pt is small & fast)
        conf: confidence threshold for detections
        device: "cpu" or "cuda"
        """
        self.enabled = True
        self.conf = conf

        # Load YOLO model
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            # If device transfer fails, just stay on default
            pass

    def set_enabled(self, enabled: bool):
        """Allow Visionary to toggle detection on/off from the UI."""
        self.enabled = enabled

    def process_frame(self, frame_bgr):
        """
        Main entry point Visionary will use.

        Input:
            frame_bgr: OpenCV BGR frame from the camera

        Output:
            processed_frame_bgr: frame with boxes/labels drawn
            detections: list of dicts with basic info about each detection
                        [{ 'cls': str, 'conf': float, 'bbox': (x1,y1,x2,y2) }, ...]
        """
        if not self.enabled:
            return frame_bgr, []

        # Run YOLO inference
        # results is a list; we only care about the first (and only) frame here
        results = self.model(frame_bgr, conf=self.conf, verbose=False)
        result = results[0]

        output_frame = frame_bgr.copy()
        detections = []

        # result.names maps class id -> class name
        names = result.names

        for box in result.boxes:
            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))

            # Draw rectangle + label on the output_frame
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(output_frame, p1, p2, (0, 255, 0), 2)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                output_frame,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            detections.append({
                "cls": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
            })

        return output_frame, detections
