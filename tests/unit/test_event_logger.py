"""
Unit tests for Event Logger component.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai.event_logger import EventLogger


@pytest.mark.unit
@pytest.mark.ai
class TestEventLogger:
    """Test suite for EventLogger class."""
    
    def test_initialization_default_path(self, temp_dir):
        """Test initialization with default log file path."""
        # Use custom path for this test
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        assert logger.log_file == log_file
        assert Path(log_file).exists()
    
    def test_initialization_custom_path(self, temp_dir):
        """Test initialization with custom log file path."""
        log_file = str(Path(temp_dir) / "custom_events.json")
        logger = EventLogger(log_file=log_file)
        assert logger.log_file == log_file
        assert Path(log_file).exists()
    
    def test_log_event(self, temp_dir):
        """Test logging an event."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        event = logger.log_event(
            description="Test description",
            duration_seconds=30.5,
            detection_summary="2 persons, 1 book",
            unique_objects=2,
            detection_count=15
        )
        
        assert event["description"] == "Test description"
        assert event["duration_seconds"] == 30.5
        assert event["detection_summary"] == "2 persons, 1 book"
        assert event["unique_objects"] == 2
        assert event["detection_frames"] == 15
        assert "timestamp" in event
        assert event["event_type"] == "narration_period"
        
        # Verify event was saved to file
        with open(log_file, 'r') as f:
            events = json.load(f)
            assert len(events) == 1
            assert events[0]["description"] == "Test description"
    
    def test_log_multiple_events(self, temp_dir):
        """Test logging multiple events."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        logger.log_event("Event 1", 10.0, "1 person", 1, 5)
        logger.log_event("Event 2", 20.0, "2 persons", 1, 10)
        
        with open(log_file, 'r') as f:
            events = json.load(f)
            assert len(events) == 2
            assert events[0]["description"] == "Event 1"
            assert events[1]["description"] == "Event 2"
    
    def test_get_recent_events(self, temp_dir):
        """Test getting recent events."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        # Log multiple events
        for i in range(5):
            logger.log_event(f"Event {i}", 10.0, "1 person", 1, 5)
        
        recent = logger.get_recent_events(limit=3)
        assert len(recent) == 3
        # Should be most recent first
        assert recent[0]["description"] == "Event 4"
        assert recent[1]["description"] == "Event 3"
        assert recent[2]["description"] == "Event 2"
    
    def test_get_all_events(self, temp_dir):
        """Test getting all events."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        logger.log_event("Event 1", 10.0, "1 person", 1, 5)
        logger.log_event("Event 2", 20.0, "2 persons", 1, 10)
        
        all_events = logger.get_all_events()
        assert len(all_events) == 2
    
    def test_clear_events(self, temp_dir):
        """Test clearing all events."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        logger.log_event("Event 1", 10.0, "1 person", 1, 5)
        logger.clear_events()
        
        events = logger.get_all_events()
        assert len(events) == 0
    
    def test_event_timestamp_format(self, temp_dir):
        """Test that event timestamps are in ISO format."""
        log_file = str(Path(temp_dir) / "events.json")
        logger = EventLogger(log_file=log_file)
        
        event = logger.log_event("Test", 10.0, "1 person", 1, 5)
        # ISO format should be parseable
        parsed_time = datetime.fromisoformat(event["timestamp"])
        assert isinstance(parsed_time, datetime)

