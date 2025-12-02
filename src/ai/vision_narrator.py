"""
Vision Narrator Module

This module provides the Vision Narrator functionality that:
- Takes object detection data from ObjectTracker
- Formats it into structured text
- Sends to LLM for natural language generation
- Returns human-friendly descriptions
"""

from typing import List, Dict, Optional, Callable
from collections import Counter, defaultdict
import time
from datetime import datetime

from .llm_integration import LLMClient
from .event_logger import EventLogger
from .tts_engine import TTSEngine


class VisionNarrator:
    """
    Vision Narrator converts object detection data into natural language descriptions.
    
    This class serves as the main interface for generating accessibility-friendly
    descriptions of what the camera system sees.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        auto_narrate_interval: int = 60,
        enable_auto_narrate: bool = True,
    ):
        """
        Initialize Vision Narrator.
        
        Args:
            llm_client: LLMClient instance (if None, creates default OpenAI client)
            auto_narrate_interval: Seconds between auto-narrations
            enable_auto_narrate: Whether to auto-narrate periodically
        
        Raises:
            ValueError: If OpenAI API key is not configured in .env file
            ImportError: If openai package is not installed
        """
        if llm_client:
            self.llm_client = llm_client
        else:
            # Initialize real OpenAI client - will raise exception if API key not found
            print("DEBUG: Initializing OpenAI client...")
            self.llm_client = LLMClient()
            print("DEBUG: Vision Narrator initialized with OpenAI API")
        
        self.auto_narrate_interval = auto_narrate_interval
        self.enable_auto_narrate = enable_auto_narrate
        self._last_narration_time = 0
        self._last_description = ""
        self._on_description_callback: Optional[Callable[[str], None]] = None
        
        # Recording period mode (for manual Start/End narration)
        self._recording_period_active = False
        self._period_start_time: Optional[float] = None
        self._period_detections: List[Dict] = []  # Store all detections during period
        self._period_detection_timestamps: List[float] = []  # Track when each detection occurred
        
        # Event logging
        self.event_logger = EventLogger()
        
        # Text-to-speech for accessibility
        self.tts = TTSEngine()
    
    def format_detections(self, tracked_objects: List[Dict]) -> str:
        """
        Format tracked objects into a concise text summary.
        
        Args:
            tracked_objects: List of detection dicts from ObjectTracker
                Each dict has: {"id", "cls", "conf", "bbox"}
        
        Returns:
            Formatted string like "2 persons, 1 laptop, 1 cell phone"
        """
        if not tracked_objects:
            return "No objects detected"
        
        # Count objects by class
        class_counts = Counter()
        for obj in tracked_objects:
            cls_name = obj.get("cls", "unknown")
            class_counts[cls_name] += 1
        
        # Format into readable text
        parts = []
        for cls_name, count in class_counts.items():
            if count == 1:
                parts.append(f"1 {cls_name}")
            else:
                # Handle pluralization (simple version)
                plural = cls_name + "s" if not cls_name.endswith("s") else cls_name
                parts.append(f"{count} {plural}")
        
        return ", ".join(parts)
    
    def generate_description(
        self,
        tracked_objects: List[Dict],
        force: bool = False
    ) -> Optional[str]:
        """
        Generate a natural language description from tracked objects.
        
        Args:
            tracked_objects: List of detection dicts from ObjectTracker
            force: Force generation even if recently generated
        
        Returns:
            Description string or None if generation fails
        """
        # Format detection data
        detection_summary = self.format_detections(tracked_objects)
        
        # Check if we should generate (rate limiting for auto-narrate)
        current_time = time.time()
        if not force and self.enable_auto_narrate:
            time_since_last = current_time - self._last_narration_time
            if time_since_last < self.auto_narrate_interval:
                return None  # Too soon, skip
        
        # Generate description via LLM
        try:
            description = self.llm_client.generate_description(detection_summary)
            self._last_description = description
            self._last_narration_time = current_time
            
            # Call callback if set
            if self._on_description_callback:
                self._on_description_callback(description)
            
            return description
            
        except Exception as e:
            print(f"Error generating description: {e}")
            # Fallback to simple description
            return f"I see {detection_summary.lower()} in the frame."
    
    def set_description_callback(self, callback: Callable[[str], None]):
        """
        Set callback function to be called when new description is generated.
        
        Args:
            callback: Function that takes description string as argument
        """
        self._on_description_callback = callback
    
    def get_last_description(self) -> str:
        """Get the last generated description."""
        return self._last_description
    
    def is_available(self) -> bool:
        """Check if LLM is available and configured."""
        return self.llm_client.is_available()
    
    def start_recording_period(self):
        """
        Start a narration recording period.
        
        During this period, all detections will be collected and stored.
        Call end_recording_period() to generate a summary of everything seen.
        """
        self._recording_period_active = True
        self._period_start_time = time.time()
        self._period_detections = []
        self._period_detection_timestamps = []
    
    def stop_recording_period(self) -> Optional[str]:
        """
        Stop the recording period and generate a summary description.
        
        Returns:
            Natural language description of all detections during the period,
            or None if period was not active or no detections were recorded.
        """
        if not self._recording_period_active:
            return None
        
        self._recording_period_active = False
        
        if not self._period_detections:
            return None
        
        # Generate summary from all collected detections
        description = self._generate_period_summary()
        
        # Clear period data
        self._period_start_time = None
        self._period_detections = []
        self._period_detection_timestamps = []
        
        return description
    
    def add_detection_to_period(self, tracked_objects: List[Dict]):
        """
        Add current detections to the recording period.
        
        This should be called periodically (e.g., each frame or every few frames)
        while the recording period is active.
        
        Args:
            tracked_objects: Current detections from ObjectTracker
        """
        if not self._recording_period_active:
            return
        
        if tracked_objects:
            self._period_detections.append(tracked_objects)
            self._period_detection_timestamps.append(time.time())
    
    def is_recording_period_active(self) -> bool:
        """Check if a recording period is currently active."""
        return self._recording_period_active
    
    def get_period_stats(self) -> Dict:
        """
        Get statistics about the current recording period.
        
        Returns:
            Dictionary with period statistics:
            {
                "active": bool,
                "duration_seconds": float,
                "detection_count": int,
                "unique_objects": int
            }
        """
        if not self._recording_period_active or self._period_start_time is None:
            return {
                "active": False,
                "duration_seconds": 0,
                "detection_count": 0,
                "unique_objects": 0
            }
        
        duration = time.time() - self._period_start_time
        
        # Count unique objects seen during period
        all_objects = []
        for detections in self._period_detections:
            all_objects.extend([obj.get("cls", "unknown") for obj in detections])
        unique_objects = len(set(all_objects))
        
        return {
            "active": True,
            "duration_seconds": duration,
            "detection_count": len(self._period_detections),
            "unique_objects": unique_objects
        }
    
    def _generate_period_summary(self) -> str:
        """
        Generate a summary description from all detections collected during the period.
        
        This aggregates all detections and creates a comprehensive summary.
        Uses unique track IDs to avoid counting the same object multiple times.
        """
        # Aggregate all detections from the period
        all_detections = []
        for detections in self._period_detections:
            all_detections.extend(detections)
        
        if not all_detections:
            return "No objects were detected during this period."
        
        # Count unique objects by class using track IDs
        # Track IDs are persistent across frames, so we can deduplicate
        seen_track_ids = set()
        class_counts = Counter()
        
        for obj in all_detections:
            track_id = obj.get("id")
            cls_name = obj.get("cls", "unknown")
            
            # Use track ID if available, otherwise fall back to counting all detections
            if track_id is not None:
                # Only count each unique track ID once per class
                unique_key = (track_id, cls_name)
                if unique_key not in seen_track_ids:
                    seen_track_ids.add(unique_key)
                    class_counts[cls_name] += 1
            else:
                # If no track ID, count all (fallback for detections without tracking)
                class_counts[cls_name] += 1
        
        # Format summary with more detail for period
        parts = []
        for cls_name, count in class_counts.items():
            if count == 1:
                parts.append(f"1 {cls_name}")
            else:
                plural = cls_name + "s" if not cls_name.endswith("s") else cls_name
                parts.append(f"{count} {plural}")
        
        detection_summary = ", ".join(parts)
        
        # Calculate period duration
        if self._period_start_time and self._period_detection_timestamps:
            duration = self._period_detection_timestamps[-1] - self._period_start_time
            duration_str = f"{duration:.1f} seconds"
        else:
            duration_str = "the recording period"
        
        # Create enhanced prompt for period summary (event-style with suggestions)
        period_prompt = (
            f"During {duration_str}, the following objects were detected: {detection_summary}. "
            f"Analyze this detection data and provide: "
            f"1. A clear description of what was observed "
            f"2. Suggestions about what activity or event might have been occurring "
            f"3. Context about the scene (e.g., 'people moving', 'objects being used', 'activity patterns')"
        )
        
        try:
            # Use an event-focused system prompt that provides interpretive suggestions
            system_prompt = (
                "You are an accessibility assistant analyzing security camera-style event logs. "
                "For each detection period, provide: "
                "1. A clear, natural description of what was detected (use first person: 'I observed...') "
                "2. Interpretive suggestions about what activity or event was likely occurring "
                "3. Context about movement patterns, object usage, or scene dynamics "
                "Format as a brief event log entry that helps someone understand what happened. "
                "Be descriptive but concise. Focus on actionable insights about the activity."
            )
            description = self.llm_client.generate_description(period_prompt, system_prompt)
            self._last_description = description
            
            # Log as event
            if self._period_start_time and self._period_detection_timestamps:
                duration = self._period_detection_timestamps[-1] - self._period_start_time
            else:
                duration = time.time() - self._period_start_time if self._period_start_time else 0
            
            unique_objects = len(class_counts)
            self.event_logger.log_event(
                description=description,
                duration_seconds=duration,
                detection_summary=detection_summary,
                unique_objects=unique_objects,
                detection_count=len(self._period_detections)
            )
            
            # Speak the description for accessibility
            if self.tts.is_available():
                self.tts.speak(description, async_mode=True)
            
            return description
        except Exception as e:
            print(f"Error generating period summary: {e}")
            # Fallback description
            fallback = f"During the recording period, I saw {detection_summary.lower()}."
            
            # Still log the event even with fallback
            try:
                duration = self._period_detection_timestamps[-1] - self._period_start_time if self._period_start_time and self._period_detection_timestamps else 0
                self.event_logger.log_event(
                    description=fallback,
                    duration_seconds=duration,
                    detection_summary=detection_summary,
                    unique_objects=len(class_counts),
                    detection_count=len(self._period_detections)
                )
            except Exception:
                pass  # Don't fail if logging fails
            
            return fallback

