"""
Unit tests for Vision Narrator component.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai.vision_narrator import VisionNarrator
from ai.llm_integration import LLMClient


@pytest.mark.unit
@pytest.mark.ai
class TestVisionNarrator:
    """Test suite for VisionNarrator class."""
    
    @patch('ai.vision_narrator.LLMClient')
    def test_initialization_with_client(self, mock_llm_client_class):
        """Test initialization with provided LLM client."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        assert narrator.llm_client == mock_client
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('ai.vision_narrator.LLMClient')
    def test_initialization_without_client(self, mock_llm_client_class):
        """Test initialization without provided client (creates default)."""
        mock_client = MagicMock()
        mock_llm_client_class.return_value = mock_client
        narrator = VisionNarrator()
        assert narrator.llm_client == mock_client
        mock_llm_client_class.assert_called_once()
    
    def test_format_detections_empty(self):
        """Test formatting empty detection list."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        result = narrator.format_detections([])
        assert result == "No objects detected"
    
    def test_format_detections_single_object(self, mock_tracked_objects):
        """Test formatting single object detection."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        single_obj = [mock_tracked_objects[0]]  # Just one person
        result = narrator.format_detections(single_obj)
        assert "1 person" in result
    
    def test_format_detections_multiple_objects(self, mock_tracked_objects):
        """Test formatting multiple object detections."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        result = narrator.format_detections(mock_tracked_objects)
        assert "person" in result.lower()
        assert "cell phone" in result.lower()
        assert "book" in result.lower()
    
    def test_format_detections_pluralization(self):
        """Test pluralization in detection formatting."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        multiple_persons = [
            {"id": 1, "cls": "person", "conf": 0.85, "bbox": (0, 0, 10, 10)},
            {"id": 2, "cls": "person", "conf": 0.90, "bbox": (20, 20, 30, 30)},
        ]
        result = narrator.format_detections(multiple_persons)
        assert "2" in result
        assert "person" in result.lower()
    
    def test_start_recording_period(self):
        """Test starting a recording period."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.start_recording_period()
        assert narrator.is_recording_period_active() == True
        assert narrator._period_start_time is not None
    
    def test_stop_recording_period_no_detections(self):
        """Test stopping recording period with no detections."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.start_recording_period()
        result = narrator.stop_recording_period()
        assert result is None
        assert narrator.is_recording_period_active() == False
    
    def test_add_detection_to_period(self, mock_tracked_objects):
        """Test adding detections during recording period."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.start_recording_period()
        narrator.add_detection_to_period(mock_tracked_objects)
        assert len(narrator._period_detections) == 1
        assert narrator._period_detections[0] == mock_tracked_objects
    
    def test_add_detection_when_not_recording(self, mock_tracked_objects):
        """Test that adding detections when not recording does nothing."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.add_detection_to_period(mock_tracked_objects)
        assert len(narrator._period_detections) == 0
    
    @patch('ai.vision_narrator.LLMClient')
    def test_stop_recording_period_with_detections(self, mock_llm_client_class, mock_tracked_objects):
        """Test stopping recording period and generating summary."""
        mock_client = MagicMock()
        mock_client.generate_description.return_value = "I saw two people and a cell phone."
        mock_llm_client_class.return_value = mock_client
        
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.start_recording_period()
        narrator.add_detection_to_period(mock_tracked_objects)
        time.sleep(0.1)  # Small delay to ensure timestamp difference
        result = narrator.stop_recording_period()
        
        assert result is not None
        assert "I saw" in result
        mock_client.generate_description.assert_called_once()
    
    def test_get_period_stats_not_active(self):
        """Test getting period stats when not recording."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        stats = narrator.get_period_stats()
        assert stats["active"] == False
        assert stats["duration_seconds"] == 0
        assert stats["detection_count"] == 0
    
    def test_get_period_stats_active(self, mock_tracked_objects):
        """Test getting period stats when recording."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator.start_recording_period()
        narrator.add_detection_to_period(mock_tracked_objects)
        time.sleep(0.1)
        stats = narrator.get_period_stats()
        assert stats["active"] == True
        assert stats["detection_count"] == 1
        assert stats["duration_seconds"] > 0
    
    def test_set_description_callback(self):
        """Test setting description callback."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        callback = Mock()
        narrator.set_description_callback(callback)
        assert narrator._on_description_callback == callback
    
    def test_get_last_description(self):
        """Test getting last description."""
        mock_client = MagicMock()
        narrator = VisionNarrator(llm_client=mock_client)
        narrator._last_description = "Test description"
        assert narrator.get_last_description() == "Test description"
    
    def test_is_available(self):
        """Test availability check."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        narrator = VisionNarrator(llm_client=mock_client)
        assert narrator.is_available() == True

