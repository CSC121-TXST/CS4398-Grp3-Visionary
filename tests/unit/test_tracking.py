"""
Unit tests for ObjectTracker component.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.tracking import ObjectTracker


@pytest.mark.unit
@pytest.mark.vision
class TestObjectTracker:
    """Test suite for ObjectTracker class."""
    
    def test_initialization(self):
        """Test ObjectTracker initialization."""
        # This will fail if YOLO model can't be loaded, but that's expected in CI
        # In real environment, yolov8n.pt should be available
        try:
            tracker = ObjectTracker(
                model_path="yolov8n.pt",
                conf=0.25,
                target_classes=["person", "cell phone"]
            )
            assert tracker is not None
            assert tracker.conf == 0.25
            assert tracker.target_classes == {"person", "cell phone"}
            assert tracker.enabled == False  # Disabled by default
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_enabled(self):
        """Test enabling/disabling tracker."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_enabled(True)
            assert tracker.enabled == True
            tracker.set_enabled(False)
            assert tracker.enabled == False
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_target_classes(self):
        """Test setting target classes."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_target_classes(["person", "book"])
            assert tracker.target_classes == {"person", "book"}
            
            tracker.set_target_classes(None)
            assert tracker.target_classes is None
            
            tracker.set_target_classes([])
            assert tracker.target_classes is None
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_target_classes_case_insensitive(self):
        """Test that target classes are case-insensitive."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_target_classes(["Person", "CELL PHONE", "Book"])
            assert "person" in tracker.target_classes
            assert "cell phone" in tracker.target_classes
            assert "book" in tracker.target_classes
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_conf(self):
        """Test setting confidence threshold."""
        try:
            tracker = ObjectTracker(conf=0.35)
            tracker.set_conf(0.25)
            assert tracker.conf == 0.25
            tracker.set_conf(0.50)
            assert tracker.conf == 0.50
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_imgsz(self):
        """Test setting image size."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_imgsz(480)
            assert tracker.imgsz == 480
            tracker.set_imgsz(832)
            assert tracker.imgsz == 832
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_process_interval(self):
        """Test setting process interval."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_process_interval(4)
            assert tracker.process_interval == 4
            tracker.set_process_interval(1)
            assert tracker.process_interval == 1
            # Should enforce minimum of 1
            tracker.set_process_interval(0)
            assert tracker.process_interval == 1
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_set_debug(self):
        """Test setting debug mode."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_debug(True)
            assert tracker.debug == True
            tracker.set_debug(False)
            assert tracker.debug == False
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_process_frame_disabled(self, sample_frame):
        """Test that process_frame returns original frame when disabled."""
        try:
            tracker = ObjectTracker(conf=0.25)
            tracker.set_enabled(False)
            result_frame, tracked_objects = tracker.process_frame(sample_frame)
            assert np.array_equal(result_frame, sample_frame)
            assert tracked_objects == []
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_xyxy_to_xywh(self):
        """Test bounding box coordinate conversion."""
        try:
            tracker = ObjectTracker(conf=0.25)
            cx, cy, w, h = tracker._xyxy_to_xywh(10, 20, 50, 80)
            assert cx == 30.0  # (10 + 50) / 2
            assert cy == 50.0  # (20 + 80) / 2
            assert w == 40.0    # 50 - 10
            assert h == 60.0   # 80 - 20
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")
    
    def test_get_tracking_stats(self):
        """Test getting tracking statistics."""
        try:
            tracker = ObjectTracker(conf=0.25)
            stats = tracker.get_tracking_stats()
            assert "tracked_count" in stats
            assert "enabled" in stats
            assert "process_interval" in stats
            assert "imgsz" in stats
            assert "conf" in stats
            assert "target_classes" in stats
            assert stats["enabled"] == False
            assert stats["tracked_count"] == 0
        except FileNotFoundError:
            pytest.skip("YOLO model file not found - skipping test")

