"""
Integration tests for tracking and narration components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.ai
class TestTrackingNarrationIntegration:
    """Test integration between ObjectTracker and VisionNarrator."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('ai.vision_narrator.LLMClient')
    def test_tracking_to_narration_flow(self, mock_llm_client_class, mock_tracked_objects):
        """Test complete flow from tracking to narration."""
        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.generate_description.return_value = "I saw two people and a cell phone."
        mock_llm_client_class.return_value = mock_llm_client
        
        from ai.vision_narrator import VisionNarrator
        
        narrator = VisionNarrator(llm_client=mock_llm_client)
        narrator.start_recording_period()
        
        # Simulate adding detections from tracker
        narrator.add_detection_to_period(mock_tracked_objects)
        
        # Generate narration
        import time
        time.sleep(0.1)
        description = narrator.stop_recording_period()
        
        assert description is not None
        assert "I saw" in description
        mock_llm_client.generate_description.assert_called_once()
    
    def test_narration_with_empty_tracking(self):
        """Test narration when no objects are tracked."""
        mock_llm_client = MagicMock()
        from ai.vision_narrator import VisionNarrator
        
        narrator = VisionNarrator(llm_client=mock_llm_client)
        narrator.start_recording_period()
        # No detections added
        result = narrator.stop_recording_period()
        
        assert result is None
        mock_llm_client.generate_description.assert_not_called()
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('ai.vision_narrator.LLMClient')
    def test_event_logging_integration(self, mock_llm_client_class, mock_tracked_objects, temp_dir):
        """Test that events are logged when narration period ends."""
        import tempfile
        from ai.event_logger import EventLogger
        
        mock_llm_client = MagicMock()
        mock_llm_client.generate_description.return_value = "Test description"
        mock_llm_client_class.return_value = mock_llm_client
        
        log_file = str(Path(temp_dir) / "test_events.json")
        
        from ai.vision_narrator import VisionNarrator
        narrator = VisionNarrator(llm_client=mock_llm_client)
        narrator.event_logger = EventLogger(log_file=log_file)
        
        narrator.start_recording_period()
        narrator.add_detection_to_period(mock_tracked_objects)
        import time
        time.sleep(0.1)
        narrator.stop_recording_period()
        
        # Check that event was logged
        events = narrator.event_logger.get_all_events()
        assert len(events) == 1
        assert events[0]["description"] == "Test description"

