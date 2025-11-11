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
    def __init__(self, parent, on_fps, tracker=None):
        """
        Initialize VideoPanel.
        
        Args:
            parent: Parent Tkinter widget
            on_fps: Callback(float) for FPS updates
            tracker: Optional ObjectTracker instance for object detection/tracking
        """
        super().__init__(parent, text="Video Feed")
        self.tracker = tracker
        
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
            # For now, we only use the processed frame (with boxes drawn)
            # The tracked_objects metadata could be used for future features
            return processed_frame
        return frame_bgr