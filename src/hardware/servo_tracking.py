"""
Servo Tracking Module

Converts object detection coordinates from the camera view to servo pan/tilt angles
for automatic target tracking.

Servo Configuration:
- Bottom servo (Pan, Pin 10): Left-right movement, positioned 2 inches back from camera
- Top servo (Tilt, Pin 9): Up-down movement, positioned 3 inches above bottom servo

Safety: Aims laser lower than center mass to avoid face striking.
"""

import numpy as np
from typing import Tuple, Optional


class ServoTracker:
    """
    Converts camera coordinates to servo angles for target tracking.
    
    Takes into account:
    - Camera field of view (FOV)
    - Servo positioning relative to camera
    - Safety offsets to aim below center mass
    """
    
    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        fov_horizontal: float = 62.0,  # Typical webcam horizontal FOV in degrees
        fov_vertical: float = 48.0,    # Typical webcam vertical FOV in degrees
        pan_center: int = 90,          # Center pan angle
        tilt_center: int = 90,         # Center tilt angle
        pan_range: Tuple[int, int] = (30, 150),    # Pan angle limits
        tilt_range: Tuple[int, int] = (40, 140),   # Tilt angle limits
        offset_vertical: float = 0.15,  # Aim 15% below center mass (fraction of bbox height)
    ):
        """
        Initialize the ServoTracker.
        
        Args:
            image_width: Width of camera image in pixels
            image_height: Height of camera image in pixels
            fov_horizontal: Camera horizontal field of view in degrees
            fov_vertical: Camera vertical field of view in degrees
            pan_center: Center pan angle (typically 90)
            tilt_center: Center tilt angle (typically 90)
            pan_range: (min, max) pan angle limits
            tilt_range: (min, max) tilt angle limits
            offset_vertical: Vertical offset as fraction of bounding box height
                            (aims this much below center mass, e.g., 0.15 = 15% below)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.pan_center = pan_center
        self.tilt_center = tilt_center
        self.pan_min, self.pan_max = pan_range
        self.tilt_min, self.tilt_max = tilt_range
        self.offset_vertical = offset_vertical
        
        # Calculate degrees per pixel
        self.degrees_per_pixel_h = self.fov_horizontal / self.image_width
        self.degrees_per_pixel_v = self.fov_vertical / self.image_height
    
    def update_image_size(self, width: int, height: int):
        """Update image dimensions if camera resolution changes."""
        self.image_width = width
        self.image_height = height
        self.degrees_per_pixel_h = self.fov_horizontal / self.image_width
        self.degrees_per_pixel_v = self.fov_vertical / self.image_height
    
    def bbox_to_angles(
        self,
        bbox: Tuple[float, float, float, float],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Convert bounding box coordinates to servo pan/tilt angles.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates in image space
            image_width: Optional override for image width
            image_height: Optional override for image height
            
        Returns:
            (pan_angle, tilt_angle): Servo angles in degrees (0-180)
        """
        x1, y1, x2, y2 = bbox
        
        # Use provided dimensions or defaults
        img_w = image_width or self.image_width
        img_h = image_height or self.image_height
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Calculate bounding box dimensions
        bbox_height = y2 - y1
        
        # Apply vertical offset: aim lower than center mass
        # Move aim point down by offset_vertical fraction of bbox height
        offset_pixels = bbox_height * self.offset_vertical
        target_y = center_y + offset_pixels
        
        # Convert pixel coordinates to normalized coordinates (-0.5 to 0.5)
        # Center of image is (0, 0)
        norm_x = (center_x - img_w / 2.0) / img_w
        norm_y = (target_y - img_h / 2.0) / img_h
        
        # Convert normalized coordinates to angle offsets
        # Positive norm_x means object is to the right, so pan right (increase angle)
        # Positive norm_y means object is below center, so tilt down (increase angle)
        pan_offset = norm_x * self.fov_horizontal
        tilt_offset = norm_y * self.fov_vertical
        
        # Calculate target angles
        pan_angle = self.pan_center + pan_offset
        tilt_angle = self.tilt_center + tilt_offset
        
        # Clamp to safe ranges
        pan_angle = int(np.clip(pan_angle, self.pan_min, self.pan_max))
        tilt_angle = int(np.clip(tilt_angle, self.tilt_min, self.tilt_max))
        
        return pan_angle, tilt_angle
    
    def center_to_angles(
        self,
        center_x: float,
        center_y: float,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        vertical_offset_pixels: float = 0.0
    ) -> Tuple[int, int]:
        """
        Convert center coordinates to servo angles.
        
        Args:
            center_x: X coordinate of target center in image space
            center_y: Y coordinate of target center in image space
            image_width: Optional override for image width
            image_height: Optional override for image height
            vertical_offset_pixels: Additional vertical offset in pixels (positive = down)
            
        Returns:
            (pan_angle, tilt_angle): Servo angles in degrees (0-180)
        """
        img_w = image_width or self.image_width
        img_h = image_height or self.image_height
        
        # Apply vertical offset
        target_y = center_y + vertical_offset_pixels
        
        # Convert to normalized coordinates
        norm_x = (center_x - img_w / 2.0) / img_w
        norm_y = (target_y - img_h / 2.0) / img_h
        
        # Convert to angle offsets
        pan_offset = norm_x * self.fov_horizontal
        tilt_offset = norm_y * self.fov_vertical
        
        # Calculate target angles
        pan_angle = self.pan_center + pan_offset
        tilt_angle = self.tilt_center + tilt_offset
        
        # Clamp to safe ranges
        pan_angle = int(np.clip(pan_angle, self.pan_min, self.pan_max))
        tilt_angle = int(np.clip(tilt_angle, self.tilt_min, self.tilt_max))
        
        return pan_angle, tilt_angle

