"""
Unit tests for TTS Engine component.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.unit
@pytest.mark.ai
class TestTTSEngine:
    """Test suite for TTSEngine class."""
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_initialization_success(self, mock_init):
        """Test successful TTS engine initialization."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        
        assert tts.available == True
        assert tts.engine == mock_engine
        mock_init.assert_called_once()
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_initialization_failure(self, mock_init):
        """Test TTS initialization failure handling."""
        mock_init.side_effect = Exception("TTS not available")
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        
        assert tts.available == False
        assert tts.engine is None
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_speak_async(self, mock_init):
        """Test speaking text in async mode."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        tts.speak("Hello world", async_mode=True)
        
        # Give it a moment to start
        import time
        time.sleep(0.1)
        
        # Should have called say and runAndWait
        mock_engine.say.assert_called_once_with("Hello world")
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_speak_sync(self, mock_init):
        """Test speaking text in sync mode."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        tts.speak("Hello world", async_mode=False)
        
        mock_engine.say.assert_called_once_with("Hello world")
        mock_engine.runAndWait.assert_called_once()
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_speak_not_available(self, mock_init):
        """Test speaking when TTS is not available."""
        mock_init.side_effect = Exception("TTS not available")
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        # Should not raise exception, just skip
        tts.speak("Hello world")
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_stop(self, mock_init):
        """Test stopping speech."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        tts.stop()
        
        mock_engine.stop.assert_called_once()
    
    @patch('ai.tts_engine.pyttsx3.init')
    def test_is_available(self, mock_init):
        """Test availability check."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        from ai.tts_engine import TTSEngine
        tts = TTSEngine()
        assert tts.is_available() == True

