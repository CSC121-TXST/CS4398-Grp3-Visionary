"""
Vision Narrator Module

This module provides the Vision Narrator functionality that:
1. Takes YOLO detection results (or placeholders)
2. Summarizes them into text
3. Sends to an LLM to generate human-friendly descriptions
4. Returns accessible descriptions for display

The narrator is designed as an accessibility tool that explains what
the vision system sees in natural language.
"""

import os
import time
from typing import List, Optional, Callable
from collections import Counter

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class DetectionResult:
    """Represents a single detection result."""
    def __init__(self, class_name: str, confidence: float = 1.0, 
                 position: Optional[str] = None):
        self.class_name = class_name
        self.confidence = confidence
        self.position = position  # e.g., "center", "left", "right", "top", "bottom"


class VisionNarrator:
    """
    Vision Narrator that converts detection results into human-friendly descriptions.
    
    This class handles:
    - Summarizing detection results into text
    - Calling LLM to generate accessible descriptions
    - Managing narration state and timing
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the Vision Narrator.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
            model: Model to use for narration (default: gpt-3.5-turbo)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._last_narration_time = 0
        self._last_detections = []
        self._narration_callback: Optional[Callable[[str], None]] = None
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.api_key and OpenAI:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
    
    def set_narration_callback(self, callback: Callable[[str], None]):
        """Set a callback function to be called when narration is generated."""
        self._narration_callback = callback
    
    def summarize_detections(self, detections: List[DetectionResult]) -> str:
        """
        Summarize detection results into a short text description.
        
        Args:
            detections: List of DetectionResult objects
            
        Returns:
            A short text summary of the detections
        """
        if not detections:
            return "No objects detected."
        
        # Count objects by class
        object_counts = Counter(det.class_name for det in detections)
        
        # Build description parts
        parts = []
        for obj_name, count in object_counts.items():
            if count == 1:
                parts.append(f"1 {obj_name}")
            else:
                parts.append(f"{count} {obj_name}s")
        
        # Add position information if available
        position_info = []
        for det in detections:
            if det.position:
                position_info.append(f"{det.class_name} at {det.position}")
        
        # Combine into summary
        if len(parts) == 1:
            summary = parts[0]
        elif len(parts) == 2:
            summary = f"{parts[0]} and {parts[1]}"
        else:
            summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        
        if position_info:
            summary += f". Positions: {', '.join(position_info)}"
        else:
            summary += " detected."
        
        return summary
    
    def generate_narration(self, detection_summary: str) -> str:
        """
        Generate a human-friendly narration using LLM.
        
        Args:
            detection_summary: The summarized detection text
            
        Returns:
            A human-friendly description suitable for accessibility
        """
        prompt = (
            "Turn this into a short spoken-style description for an accessibility assistant. "
            "Keep it concise (1-2 sentences), natural, and easy to understand. "
            "Do not add information that wasn't in the original description.\n\n"
            f"Description: {detection_summary}"
        )
        
        # Try to use OpenAI API
        if self._client and self.api_key:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an accessibility assistant that describes what a camera sees in clear, natural language."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                # Fall through to fallback
        
        # Fallback: Simple transformation if LLM is not available
        return self._fallback_narration(detection_summary)
    
    def _fallback_narration(self, summary: str) -> str:
        """
        Fallback narration when LLM is not available.
        Provides a simple transformation to make it more natural.
        """
        # Simple transformations for fallback
        if "No objects detected" in summary:
            return "I don't see any objects in the camera view right now."
        
        # Replace "detected" with more natural phrasing
        summary = summary.replace(" detected.", ".")
        summary = summary.replace(" detected", "")
        
        # Make it first-person
        if not summary.startswith("I see"):
            summary = "I see " + summary.lower()
        
        return summary
    
    def narrate(self, detections: List[DetectionResult]) -> str:
        """
        Main method to generate narration from detections.
        
        Args:
            detections: List of DetectionResult objects
            
        Returns:
            Human-friendly narration string
        """
        # Store current detections
        self._last_detections = detections
        
        # Summarize detections
        summary = self.summarize_detections(detections)
        
        # Generate narration using LLM
        narration = self.generate_narration(summary)
        
        # Update timestamp
        self._last_narration_time = time.time()
        
        # Call callback if set
        if self._narration_callback:
            try:
                self._narration_callback(narration)
            except Exception as e:
                print(f"Error in narration callback: {e}")
        
        return narration
    
    def should_narrate(self, interval_seconds: float = 60.0) -> bool:
        """
        Check if enough time has passed since last narration.
        
        Args:
            interval_seconds: Minimum seconds between narrations
            
        Returns:
            True if narration should be triggered
        """
        elapsed = time.time() - self._last_narration_time
        return elapsed >= interval_seconds


def get_placeholder_detections() -> List[DetectionResult]:
    """
    Get placeholder detection results for testing.
    
    This function should be replaced with actual YOLO detection results
    when the Detection + Tracking branch is merged.
    
    Returns:
        List of DetectionResult objects (placeholder data)
    """
    # Placeholder detections for demo
    # In real implementation, this would call YOLO detection
    return [
        DetectionResult("person", confidence=0.95, position="center"),
        DetectionResult("person", confidence=0.92, position="center"),
        DetectionResult("laptop", confidence=0.88, position="center"),
        DetectionResult("chair", confidence=0.85, position="right"),
    ]

