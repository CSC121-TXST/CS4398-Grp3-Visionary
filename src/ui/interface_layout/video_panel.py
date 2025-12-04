# ui/interface_layout/video_panel.py
import tkinter as tk
from tkinter import ttk
from typing import Optional
from ui.style import add_video_grid
from vision.camera_control import SimpleCamera

class VideoPanel(ttk.LabelFrame):
    """
    Left video area, owns canvas and camera.
    
    This panel can optionally use an ObjectTracker to process frames
    for object detection and tracking before displaying them.
    """
    def __init__(self, parent, on_fps, tracker=None, on_tracking_update=None, on_narration_detection=None, on_servo_tracking=None):
        """
        Initialize VideoPanel.
        
        Args:
            parent: Parent Tkinter widget
            on_fps: Callback(float) for FPS updates
            tracker: Optional ObjectTracker instance for object detection/tracking
            on_tracking_update: Optional callback(int) for tracking count updates
            on_narration_detection: Optional callback(List[Dict]) for narration period detections
            on_servo_tracking: Optional callback(List[Dict], frame_shape) for servo tracking
                               Called with tracked objects and frame dimensions
        """
        super().__init__(parent, text="Video Feed")
        self.tracker = tracker
        self.on_tracking_update = on_tracking_update
        self.on_narration_detection = on_narration_detection
        self.on_servo_tracking = on_servo_tracking
        
        # Canvas with grid/crosshair
        self.canvas = add_video_grid(self)
        
        # Camera instance that draws into the canvas
        # Always pass _process_frame as frame_processor (it handles None tracker gracefully)
        self.camera = SimpleCamera(
            canvas=self.canvas,
            index=0,
            mirror=True,
            on_fps=on_fps,
            frame_processor=self._process_frame
        )
    
    def _process_frame(self, frame_bgr):
        """
        Process a frame through the tracker if available.
        
        This method is called by SimpleCamera for each frame before rendering.
        It applies object detection and tracking, then returns the processed frame
        with bounding boxes and IDs drawn.
        
        Args:
            frame_bgr: BGR frame from the camera
            
        Returns:
            processed_frame_bgr: Frame with tracking annotations (or original if no tracker)
        """
        if self.tracker is not None:
            # Process frame through tracker
            # Returns: (processed_frame, tracked_objects)
            processed_frame, tracked_objects = self.tracker.process_frame(frame_bgr)
            # Update tracking statistics if callback is provided
            if self.on_tracking_update is not None:
                self.on_tracking_update(len(tracked_objects))
            # Pass detections to narration system if callback provided
            if self.on_narration_detection is not None:
                self.on_narration_detection(tracked_objects)
            # Pass detections to servo tracking system if callback provided
            if self.on_servo_tracking is not None and len(tracked_objects) > 0:
                h, w = frame_bgr.shape[:2]
                self.on_servo_tracking(tracked_objects, (w, h))
            return processed_frame
        elif self.on_tracking_update is not None:
            # No tracker, so no objects being tracked
            self.on_tracking_update(0)
        return frame_bgr