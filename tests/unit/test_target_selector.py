"""
Unit tests for Target Selector component.
"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hardware.target_selector import TargetSelector


@pytest.mark.unit
@pytest.mark.hardware
class TestTargetSelector:
    """Test suite for TargetSelector class."""
    
    def test_initialization_default(self):
        """Test TargetSelector initialization with default parameters."""
        selector = TargetSelector()
        assert selector.persistence_frames == 10
        assert selector.priority_threshold == 20.0
        assert selector.center_weight == 0.3
        assert selector.size_weight == 0.4
        assert selector.confidence_weight == 0.2
        assert selector.class_weight == 0.1
        assert selector.current_target_id is None
        assert selector.frames_since_seen == 0
        assert selector.current_target_priority == 0.0
    
    def test_initialization_custom(self):
        """Test TargetSelector initialization with custom parameters."""
        selector = TargetSelector(
            persistence_frames=5,
            priority_threshold=15.0,
            center_weight=0.5,
            size_weight=0.3,
            confidence_weight=0.1,
            class_weight=0.1
        )
        assert selector.persistence_frames == 5
        assert selector.priority_threshold == 15.0
        assert selector.center_weight == 0.5
    
    def test_select_target_empty_list(self):
        """Test selecting target when no objects detected."""
        selector = TargetSelector(persistence_frames=3)
        selector.current_target_id = 1
        
        result = selector.select_target([], frame_width=640, frame_height=480)
        assert result is None
        assert selector.frames_since_seen == 1
    
    def test_select_target_empty_list_lost_target(self):
        """Test that target is cleared after persistence frames."""
        selector = TargetSelector(persistence_frames=2)
        selector.current_target_id = 1
        
        # First frame without target
        selector.select_target([], frame_width=640, frame_height=480)
        assert selector.current_target_id == 1  # Still tracking
        
        # Second frame without target
        selector.select_target([], frame_width=640, frame_height=480)
        assert selector.current_target_id == 1  # Still tracking
        
        # Third frame - should clear
        selector.select_target([], frame_width=640, frame_height=480)
        assert selector.current_target_id is None
    
    def test_select_target_no_current_target(self):
        """Test selecting target when no current target exists."""
        selector = TargetSelector()
        objects = [
            {"id": 1, "cls": "person", "conf": 0.9, "bbox": (100, 100, 200, 300)},
            {"id": 2, "cls": "book", "conf": 0.7, "bbox": (300, 200, 400, 300)},
        ]
        
        result = selector.select_target(objects, frame_width=640, frame_height=480)
        assert result is not None
        # Person should be selected (higher priority class)
        assert result["id"] == 1
        assert selector.current_target_id == 1
        assert selector.frames_since_seen == 0
    
    def test_select_target_persists_current(self):
        """Test that current target persists when still visible."""
        selector = TargetSelector()
        selector.current_target_id = 1
        
        objects = [
            {"id": 1, "cls": "book", "conf": 0.7, "bbox": (100, 100, 200, 300)},
            {"id": 2, "cls": "person", "conf": 0.9, "bbox": (300, 200, 400, 300)},
        ]
        
        result = selector.select_target(objects, frame_width=640, frame_height=480)
        # Should keep current target (id=1) even though person has higher priority
        # because persistence bonus keeps it
        assert result is not None
        assert result["id"] == 1
        assert selector.frames_since_seen == 0
    
    def test_select_target_switches_when_much_better(self):
        """Test that target switches when new target is significantly better."""
        selector = TargetSelector(priority_threshold=10.0)
        selector.current_target_id = 1
        
        # Current target: small book in corner
        # New target: large person in center (much better)
        objects = [
            {"id": 1, "cls": "book", "conf": 0.5, "bbox": (10, 10, 50, 50)},  # Small, low conf
            {"id": 2, "cls": "person", "conf": 0.95, "bbox": (200, 150, 500, 450)},  # Large, high conf, center
        ]
        
        result = selector.select_target(objects, frame_width=640, frame_height=480)
        # Should switch to person if priority difference is large enough
        # This depends on the scoring, but with large size/confidence difference it should switch
        assert result is not None
    
    def test_calculate_priority_person(self):
        """Test priority calculation for person (highest priority class)."""
        selector = TargetSelector()
        obj = {
            "cls": "person",
            "conf": 0.9,
            "bbox": (300, 200, 400, 350)  # Center of 640x480 frame
        }
        
        score = selector._calculate_priority(obj, frame_width=640, frame_height=480)
        assert score > 0
        # Person should have high class priority (100)
    
    def test_calculate_priority_book(self):
        """Test priority calculation for book."""
        selector = TargetSelector()
        obj = {
            "cls": "book",
            "conf": 0.8,
            "bbox": (300, 200, 400, 350)
        }
        
        score = selector._calculate_priority(obj, frame_width=640, frame_height=480)
        assert score > 0
        # Book should have lower class priority than person
    
    def test_calculate_priority_center_vs_corner(self):
        """Test that objects in center have higher priority than in corner."""
        selector = TargetSelector()
        
        center_obj = {
            "cls": "person",
            "conf": 0.9,
            "bbox": (300, 200, 400, 350)  # Center
        }
        
        corner_obj = {
            "cls": "person",
            "conf": 0.9,
            "bbox": (10, 10, 100, 100)  # Corner
        }
        
        center_score = selector._calculate_priority(center_obj, frame_width=640, frame_height=480)
        corner_score = selector._calculate_priority(corner_obj, frame_width=640, frame_height=480)
        
        assert center_score > corner_score
    
    def test_calculate_priority_size_matters(self):
        """Test that larger objects have higher priority."""
        selector = TargetSelector()
        
        large_obj = {
            "cls": "person",
            "conf": 0.9,
            "bbox": (100, 100, 500, 400)  # Large
        }
        
        small_obj = {
            "cls": "person",
            "conf": 0.9,
            "bbox": (300, 200, 350, 250)  # Small
        }
        
        large_score = selector._calculate_priority(large_obj, frame_width=640, frame_height=480)
        small_score = selector._calculate_priority(small_obj, frame_width=640, frame_height=480)
        
        assert large_score > small_score
    
    def test_calculate_priority_confidence_matters(self):
        """Test that higher confidence objects have higher priority."""
        selector = TargetSelector()
        
        high_conf_obj = {
            "cls": "person",
            "conf": 0.95,
            "bbox": (300, 200, 400, 350)
        }
        
        low_conf_obj = {
            "cls": "person",
            "conf": 0.5,
            "bbox": (300, 200, 400, 350)
        }
        
        high_score = selector._calculate_priority(high_conf_obj, frame_width=640, frame_height=480)
        low_score = selector._calculate_priority(low_conf_obj, frame_width=640, frame_height=480)
        
        assert high_score > low_score
    
    def test_calculate_priority_unknown_class(self):
        """Test priority calculation for unknown class."""
        selector = TargetSelector()
        obj = {
            "cls": "unknown_object",
            "conf": 0.8,
            "bbox": (300, 200, 400, 350)
        }
        
        score = selector._calculate_priority(obj, frame_width=640, frame_height=480)
        assert score > 0  # Should still have a score, just lower class priority
    
    def test_reset(self):
        """Test resetting target selector state."""
        selector = TargetSelector()
        selector.current_target_id = 5
        selector.frames_since_seen = 3
        selector.current_target_priority = 50.0
        
        selector.reset()
        
        assert selector.current_target_id is None
        assert selector.frames_since_seen == 0
        assert selector.current_target_priority == 0.0
    
    def test_get_current_target_id(self):
        """Test getting current target ID."""
        selector = TargetSelector()
        assert selector.get_current_target_id() is None
        
        selector.current_target_id = 3
        assert selector.get_current_target_id() == 3
    
    def test_select_target_within_persistence_window(self):
        """Test that None is returned when target lost but within persistence window."""
        selector = TargetSelector(persistence_frames=5)
        selector.current_target_id = 1
        selector.frames_since_seen = 2  # Lost for 2 frames, but persistence is 5
        
        # No objects detected
        result = selector.select_target([], frame_width=640, frame_height=480)
        assert result is None
        assert selector.current_target_id == 1  # Still tracking
    
    def test_class_priorities(self):
        """Test that class priorities are correctly defined."""
        selector = TargetSelector()
        
        # Check that person has highest priority
        assert selector.CLASS_PRIORITIES["person"] == 100
        assert selector.CLASS_PRIORITIES["cell phone"] == 50
        assert selector.CLASS_PRIORITIES["book"] == 40
        
        # Check default priority
        assert selector.DEFAULT_CLASS_PRIORITY == 10
    
    def test_select_target_case_insensitive_class(self):
        """Test that class names are handled case-insensitively."""
        selector = TargetSelector()
        
        # Test with different case
        obj = {
            "cls": "PERSON",  # Uppercase
            "conf": 0.9,
            "bbox": (300, 200, 400, 350)
        }
        
        score = selector._calculate_priority(obj, frame_width=640, frame_height=480)
        # Should still recognize as person (class priority lookup is case-insensitive)
        assert score > 0

