"""
Text-to-Speech Engine for Accessibility

Provides audio output for narration descriptions.
"""

import pyttsx3
from typing import Optional
import threading


class TTSEngine:
    """
    Text-to-speech engine for reading narration descriptions aloud.
    """
    
    def __init__(self):
        """Initialize TTS engine."""
        try:
            self.engine = pyttsx3.init()
            self._configure_voice()
            self.available = True
        except Exception as e:
            print(f"Warning: TTS not available: {e}")
            self.engine = None
            self.available = False
    
    def _configure_voice(self):
        """Configure voice settings for better accessibility."""
        if self.engine is None:
            return
        
        try:
            # Set speech rate (words per minute)
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate - 20)  # Slightly slower for clarity
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', 0.9)
            
            # Try to set a more natural voice (if available)
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voices (often clearer), but use any available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"Warning: Could not configure TTS voice: {e}")
    
    def speak(self, text: str, async_mode: bool = True):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            async_mode: If True, speak in background thread (non-blocking)
        """
        if not self.available or self.engine is None:
            print("TTS not available, skipping audio output")
            return
        
        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error during TTS: {e}")
        
        if async_mode:
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
        else:
            _speak()
    
    def stop(self):
        """Stop any ongoing speech."""
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self.available

