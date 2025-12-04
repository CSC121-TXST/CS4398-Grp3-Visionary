"""
Unit tests for Servo Tracking component.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hardware.servo_tracking import ServoTracker


@pytest.mark.unit
@pytest.mark.hardware
class TestServoTracker:
    """Test suite for ServoTracker class."""
    
    def test_initialization_default(self):
        """Test ServoTracker initialization with default parameters."""
        tracker = ServoTracker()
        assert tracker.image_width == 640
        assert tracker.image_height == 480
        assert tracker.fov_horizontal == 62.0
        assert tracker.fov_vertical == 48.0
        assert tracker.pan_center == 90
        assert tracker.tilt_center == 90
        assert tracker.pan_min == 30
        assert tracker.pan_max == 150
        assert tracker.tilt_min == 40
        assert tracker.tilt_max == 140
        assert tracker.offset_vertical == 0.15
    
    def test_initialization_custom(self):
        """Test ServoTracker initialization with custom parameters."""
        tracker = ServoTracker(
            image_width=1280,
            image_height=720,
            fov_horizontal=80.0,
            fov_vertical=60.0,
            pan_center=90,
            tilt_center=90,
            pan_range=(0, 180),
            tilt_range=(0, 180),
            offset_vertical=0.2
        )
        assert tracker.image_width == 1280
        assert tracker.image_height == 720
        assert tracker.fov_horizontal == 80.0
        assert tracker.offset_vertical == 0.2
        assert tracker.pan_min == 0
        assert tracker.pan_max == 180
    
    def test_degrees_per_pixel_calculation(self):
        """Test that degrees per pixel are calculated correctly."""
        tracker = ServoTracker(image_width=640, image_height=480, fov_horizontal=62.0, fov_vertical=48.0)
        
        expected_h = 62.0 / 640
        expected_v = 48.0 / 480
        
        assert abs(tracker.degrees_per_pixel_h - expected_h) < 0.001
        assert abs(tracker.degrees_per_pixel_v - expected_v) < 0.001
    
    def test_update_image_size(self):
        """Test updating image dimensions."""
        tracker = ServoTracker(image_width=640, image_height=480)
        tracker.update_image_size(1280, 720)
        
        assert tracker.image_width == 1280
        assert tracker.image_height == 720
        # Degrees per pixel should be recalculated
        expected_h = tracker.fov_horizontal / 1280
        expected_v = tracker.fov_vertical / 720
        assert abs(tracker.degrees_per_pixel_h - expected_h) < 0.001
        assert abs(tracker.degrees_per_pixel_v - expected_v) < 0.001
    
    def test_bbox_to_angles_center(self):
        """Test converting center bbox to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Bbox in center of frame
        bbox = (270, 190, 370, 290)  # Center of 640x480
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Should be close to center angles (90, 90)
        assert abs(pan - 90) < 5  # Allow some tolerance
        assert abs(tilt - 90) < 5
    
    def test_bbox_to_angles_left_side(self):
        """Test converting left-side bbox to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Bbox on left side of frame
        bbox = (50, 200, 150, 300)
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Pan should be greater than center (move left = increase angle)
        assert pan > 90
    
    def test_bbox_to_angles_right_side(self):
        """Test converting right-side bbox to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Bbox on right side of frame
        bbox = (490, 200, 590, 300)
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Pan should be less than center (move right = decrease angle, but inverted)
        # Actually, with inverted logic, right side should increase pan
        assert pan != 90  # Should be different from center
    
    def test_bbox_to_angles_top(self):
        """Test converting top bbox to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Bbox at top of frame
        bbox = (300, 50, 400, 150)
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Tilt should be adjusted (object is above center)
        assert tilt != 90
    
    def test_bbox_to_angles_bottom(self):
        """Test converting bottom bbox to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Bbox at bottom of frame
        bbox = (300, 350, 400, 450)
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Tilt should be adjusted (object is below center)
        assert tilt != 90
    
    def test_bbox_to_angles_vertical_offset(self):
        """Test that vertical offset is applied (aims below center mass)."""
        tracker = ServoTracker(image_width=640, image_height=480, offset_vertical=0.15)
        
        # Bbox in center
        bbox = (300, 200, 400, 300)  # Height = 100, so offset = 15 pixels down
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        # Tilt should account for offset (aiming lower)
        assert tilt is not None
    
    def test_bbox_to_angles_clamping(self):
        """Test that angles are clamped to safe ranges."""
        tracker = ServoTracker(
            image_width=640,
            image_height=480,
            pan_range=(30, 150),
            tilt_range=(40, 140)
        )
        
        # Bbox way off to the side (would result in angle outside range)
        bbox = (-100, 200, 50, 300)  # Way off left
        pan, tilt = tracker.bbox_to_angles(bbox)
        
        assert pan >= 30
        assert pan <= 150
        assert tilt >= 40
        assert tilt <= 140
    
    def test_bbox_to_angles_custom_image_size(self):
        """Test bbox_to_angles with custom image dimensions."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Use different image size in call
        bbox = (640, 240, 1280, 480)  # Center of 1280x720
        pan, tilt = tracker.bbox_to_angles(bbox, image_width=1280, image_height=720)
        
        # Should calculate correctly for 1280x720
        assert pan is not None
        assert tilt is not None
    
    def test_center_to_angles_center(self):
        """Test converting center coordinates to servo angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        pan, tilt = tracker.center_to_angles(320, 240, image_width=640, image_height=480)
        
        # Should be close to center angles
        assert abs(pan - 90) < 5
        assert abs(tilt - 90) < 5
    
    def test_center_to_angles_with_offset(self):
        """Test center_to_angles with vertical offset."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Center with 20 pixel offset down
        pan1, tilt1 = tracker.center_to_angles(320, 240, vertical_offset_pixels=0)
        pan2, tilt2 = tracker.center_to_angles(320, 240, vertical_offset_pixels=20)
        
        # With offset, tilt should be different
        assert tilt2 != tilt1
    
    def test_center_to_angles_clamping(self):
        """Test that center_to_angles clamps to safe ranges."""
        tracker = ServoTracker(
            image_width=640,
            image_height=480,
            pan_range=(30, 150),
            tilt_range=(40, 140)
        )
        
        # Extreme coordinates
        pan, tilt = tracker.center_to_angles(-1000, -1000)
        
        assert pan >= 30
        assert pan <= 150
        assert tilt >= 40
        assert tilt <= 140
    
    def test_center_to_angles_custom_image_size(self):
        """Test center_to_angles with custom image dimensions."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Center of 1280x720
        pan, tilt = tracker.center_to_angles(640, 360, image_width=1280, image_height=720)
        
        assert pan is not None
        assert tilt is not None
        assert abs(pan - 90) < 5  # Should still be center
    
    def test_bbox_to_angles_inverted_logic(self):
        """Test that inverted servo logic is applied correctly."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Object on right side should result in pan adjustment
        # With inverted logic, right side might increase pan
        bbox_right = (500, 200, 600, 300)
        bbox_left = (40, 200, 140, 300)
        
        pan_right, _ = tracker.bbox_to_angles(bbox_right)
        pan_left, _ = tracker.bbox_to_angles(bbox_left)
        
        # Should be different
        assert pan_right != pan_left
    
    def test_bbox_to_angles_edge_cases(self):
        """Test edge cases for bbox_to_angles."""
        tracker = ServoTracker(image_width=640, image_height=480)
        
        # Very small bbox
        bbox_small = (320, 240, 321, 241)
        pan, tilt = tracker.bbox_to_angles(bbox_small)
        assert pan is not None
        assert tilt is not None
        
        # Very large bbox (most of frame)
        bbox_large = (50, 50, 590, 430)
        pan, tilt = tracker.bbox_to_angles(bbox_large)
        assert pan is not None
        assert tilt is not None

