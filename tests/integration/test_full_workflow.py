"""
Integration tests for full workflow: Detection -> Tracking -> Narration -> Event Logging.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflow:
    """Test complete workflow from detection to narration."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('ai.vision_narrator.LLMClient')
    def test_detection_to_narration_workflow(self, mock_llm_client_class, mock_tracked_objects, temp_dir):
        """Test complete workflow: detection -> tracking -> narration -> event logging."""
        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.generate_description.return_value = "I observed two people and a cell phone during the recording period."
        mock_llm_client_class.return_value = mock_llm_client
        
        from ai.vision_narrator import VisionNarrator
        from ai.event_logger import EventLogger
        
        # Initialize components
        log_file = str(Path(temp_dir) / "workflow_events.json")
        narrator = VisionNarrator(llm_client=mock_llm_client)
        narrator.event_logger = EventLogger(log_file=log_file)
        
        # Simulate workflow
        # 1. Start recording period
        narrator.start_recording_period()
        assert narrator.is_recording_period_active() == True
        
        # 2. Add detections (simulating what tracker would provide)
        narrator.add_detection_to_period(mock_tracked_objects)
        narrator.add_detection_to_period(mock_tracked_objects)  # Add again to simulate multiple frames
        
        # 3. Stop recording and generate narration
        import time
        time.sleep(0.1)
        description = narrator.stop_recording_period()
        
        # Verify results
        assert description is not None
        assert "I observed" in description
        assert narrator.is_recording_period_active() == False
        
        # 4. Verify event was logged
        events = narrator.event_logger.get_all_events()
        assert len(events) == 1
        assert events[0]["description"] == description
        assert events[0]["event_type"] == "narration_period"
        assert events[0]["detection_frames"] == 2
        
        # Verify LLM was called
        mock_llm_client.generate_description.assert_called_once()

