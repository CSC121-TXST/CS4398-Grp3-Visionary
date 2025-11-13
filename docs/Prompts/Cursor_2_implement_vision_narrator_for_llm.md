# Implement vision narrator for LLM integration
_Exported on 11/11/2025 at 09:48:28 CST from Cursor (2.0.69)_

---

**User**

Implement the following for our LLM Integration Branch:

I would like you to design the beginning of a “Vision Narrator” which serves as a LLM that explains what the system sees

What it should do:

Every minute, or when asked, Visionary:

- Takes the current YOLO detections (e.g. 2 persons, 1 chair, 1 laptop).

- Summarizes them into a short text like:

“There are two people standing near the center, and a laptop on the table.”

- Sends that text to an LLM with a prompt like:

- “Turn this into a short spoken-style description for an accessibility assistant.”

- LLM responds with something nicer:

- “I see two people in front of the camera and a laptop on the table.”

- You display that text in a read-only “Vision Narration” box in the UI (and optionally TTS).



For now it only needs a text in text out step.

It should not affect servos or tracking logic, so no risk to core functionality.

Great demo line to make the goal clearer:

The detector finds people and objects, and our LLM generates human-friendly descriptions imagine this as the base for an accessibility tool.



Currently you are in a Branch that does not have the YOLO + DeepSort implementation, so be sure to implement this instruction with placeholders were applicable before we merge back to the Detection + Tracking branch

---

**Cursor**


Exploring the codebase to understand structure and existing components.



Reviewing key files to understand the current structure:



Checking style and branch info:



Creating the Vision Narrator. Implementing step by step:



Creating the Vision Narrator module:


```python
"""
AI Module for Visionary

This module contains AI-related functionality including LLM integration
for the Vision Narrator feature.
"""

```

```python
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
from typing import List, Dict, Optional, Callable
from collections import Counter

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                self._client = openai
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
                response = self._client.ChatCompletion.create(
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

```


Updating the narrator to use the OpenAI client correctly and adding a placeholder detection function:


```python
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
from typing import List, Dict, Optional, Callable
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
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                self._client = openai
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
                response = self._client.ChatCompletion.create(
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


```

```python
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
from typing import List, Dict, Optional, Callable
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
                response = self._client.ChatCompletion.create(
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


```

```python
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
from typing import List, Dict, Optional, Callable
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


```


Adding a placeholder detection function and creating the UI component:


```python
"""
Object Detection Module

This module handles real-time object detection using computer vision
algorithms to identify objects in camera feed.

Note: This is a placeholder implementation for the LLM Integration branch.
When merged with the Detection + Tracking branch, this will be replaced
with actual YOLO + DeepSort implementation.
"""

from typing import List
from ai.vision_narrator import DetectionResult, get_placeholder_detections


def get_current_detections() -> List[DetectionResult]:
    """
    Get current detection results from YOLO.
    
    This is a placeholder function for the LLM Integration branch.
    In the Detection + Tracking branch, this will call the actual YOLO detector.
    
    Returns:
        List of DetectionResult objects representing current detections
    """
    # Placeholder: Return mock detections
    # TODO: Replace with actual YOLO detection when merging branches
    return get_placeholder_detections()

```

```python
"""
Vision Narration Panel

This module provides a UI component for displaying the Vision Narrator's
human-friendly descriptions of what the camera sees.
"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2, MUTED, SEC_SURF


class VisionNarrationPanel(ttk.LabelFrame):
    """
    Panel that displays the Vision Narrator's descriptions.
    
    Shows a read-only text box with the current narration, and optionally
    a button to trigger narration on demand.
    """
    
    def __init__(self, parent, narrator=None, on_narrate_request=None):
        """
        Initialize the Vision Narration Panel.
        
        Args:
            parent: Parent widget
            narrator: VisionNarrator instance (optional)
            on_narrate_request: Callback function when user requests narration
        """
        super().__init__(parent, text="Vision Narration", padding=10)
        self.narrator = narrator
        self.on_narrate_request = on_narrate_request
        
        self._build()
        
        # Set up narration callback if narrator is provided
        if self.narrator:
            self.narrator.set_narration_callback(self.update_narration)
    
    def _build(self):
        """Build the UI components."""
        # Title/description
        desc_label = ttk.Label(
            self,
            text="Accessibility description of what the camera sees:",
            foreground=MUTED,
            font=("Segoe UI", 9)
        )
        desc_label.pack(anchor="w", pady=(0, 8))
        
        # Read-only text box for narration
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Text widget with scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.text_widget = tk.Text(
            text_frame,
            height=6,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=SEC_SURF,
            fg="#e6edf3",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#232a33",
            highlightcolor=ACCENT,
            yscrollcommand=scrollbar.set
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        # Default message
        self.update_narration("Waiting for narration...")
        
        # Button to trigger narration on demand
        if self.on_narrate_request:
            button_frame = ttk.Frame(self)
            button_frame.pack(fill="x")
            
            self.narrate_button = ttk.Button(
                button_frame,
                text="Narrate Now",
                command=self._on_narrate_clicked,
                style="Accent.TButton"
            )
            self.narrate_button.pack(side="left", padx=(0, 5))
            
            # Status label
            self.status_label = ttk.Label(
                button_frame,
                text="",
                foreground=MUTED,
                font=("Segoe UI", 8)
            )
            self.status_label.pack(side="left")
    
    def update_narration(self, narration: str):
        """
        Update the narration text display.
        
        Args:
            narration: The narration text to display
        """
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, narration)
        self.text_widget.config(state=tk.DISABLED)
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Updated")
            # Clear status after 2 seconds
            self.after(2000, lambda: self.status_label.config(text="") if hasattr(self, 'status_label') else None)
    
    def _on_narrate_clicked(self):
        """Handle narration button click."""
        if self.on_narrate_request:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Generating...", foreground=ACCENT_2)
            self.on_narrate_request()
    
    def set_narrator(self, narrator):
        """Set the narrator instance and configure callback."""
        self.narrator = narrator
        if self.narrator:
            self.narrator.set_narration_callback(self.update_narration)

```


Integrating the narration panel into the control panel and connecting it in the main interface:


```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls
from ui.interface_layout.vision_narration_panel import VisionNarrationPanel


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(self, master=None, camera=None, arduino=None, on_status=None, on_laser=None):
        """
        camera:  object with start(), stop(), is_running()
        arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
        on_status: callback(str) to update status bar
        on_laser:  callback(bool) to update laser label
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 15))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Placeholder for future stuff
        ttk.Label(self, text="[Additional controls coming soon...]", foreground="#666").pack(anchor="w", pady=(10, 0))

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls
from ui.interface_layout.vision_narration_panel import VisionNarrationPanel


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(self, master=None, camera=None, arduino=None, on_status=None, on_laser=None, narrator=None, on_narrate_request=None):
        """
        camera:  object with start(), stop(), is_running()
        arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
        on_status: callback(str) to update status bar
        on_laser:  callback(bool) to update laser label
        narrator: VisionNarrator instance (optional)
        on_narrate_request: callback() to trigger narration (optional)
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.narrator = narrator
        self.on_narrate_request = on_narrate_request

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 15))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Placeholder for future stuff
        ttk.Label(self, text="[Additional controls coming soon...]", foreground="#666").pack(anchor="w", pady=(10, 0))

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls
from ui.interface_layout.vision_narration_panel import VisionNarrationPanel


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(self, master=None, camera=None, arduino=None, on_status=None, on_laser=None, narrator=None, on_narrate_request=None):
        """
        camera:  object with start(), stop(), is_running()
        arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
        on_status: callback(str) to update status bar
        on_laser:  callback(bool) to update laser label
        narrator: VisionNarrator instance (optional)
        on_narrate_request: callback() to trigger narration (optional)
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.narrator = narrator
        self.on_narrate_request = on_narrate_request

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 15))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Vision Narration Panel
        if self.narrator or self.on_narrate_request:
            self.narration_panel = VisionNarrationPanel(
                self,
                narrator=self.narrator,
                on_narrate_request=self.on_narrate_request
            )
            self.narration_panel.pack(fill="both", expand=True, pady=(0, 0))

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```


Integrating the narrator into the main interface with timer functionality:


```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}")
        )
        self._control_widget.pack(expand=True, fill="both")

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}")
        )
        self._control_widget.pack(expand=True, fill="both")

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            narrator=self.narrator,
            on_narrate_request=self._trigger_narration
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Set up narrator callback to update UI
        if hasattr(self._control_widget, 'narration_panel'):
            self.narrator.set_narration_callback(
                self._control_widget.narration_panel.update_narration
            )

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            narrator=self.narrator,
            on_narrate_request=self._trigger_narration
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Set up narrator callback to update UI
        if hasattr(self._control_widget, 'narration_panel'):
            self.narrator.set_narration_callback(
                self._control_widget.narration_panel.update_narration
            )

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")
    
    def _start_narration_timer(self, interval_ms=60000):
        """
        Start the periodic narration timer.
        
        Args:
            interval_ms: Interval in milliseconds (default: 60000 = 1 minute)
        """
        self._trigger_narration()
        # Schedule next narration
        self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _schedule_next_narration(self, interval_ms):
        """Schedule the next periodic narration."""
        if self.winfo_exists():  # Check if window still exists
            self._trigger_narration()
            self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _trigger_narration(self):
        """Trigger narration generation from current detections."""
        try:
            # Get current detections (placeholder for now)
            detections = get_current_detections()
            
            # Generate narration
            narration = self.narrator.narrate(detections)
            
            # Update UI if narration panel exists
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(narration)
        except Exception as e:
            print(f"Error generating narration: {e}")
            # Update UI with error message
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(
                    f"Error generating narration: {str(e)}"
                )

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            narrator=self.narrator,
            on_narrate_request=self._trigger_narration
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Set up narrator callback to update UI
        if hasattr(self._control_widget, 'narration_panel'):
            self.narrator.set_narration_callback(
                self._control_widget.narration_panel.update_narration
            )

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")
    
    def _start_narration_timer(self, interval_ms=60000):
        """
        Start the periodic narration timer.
        
        Args:
            interval_ms: Interval in milliseconds (default: 60000 = 1 minute)
        """
        self._trigger_narration()
        # Schedule next narration
        self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _schedule_next_narration(self, interval_ms):
        """Schedule the next periodic narration."""
        if self.winfo_exists():  # Check if window still exists
            self._trigger_narration()
            self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _trigger_narration(self):
        """Trigger narration generation from current detections."""
        try:
            # Get current detections (placeholder for now)
            detections = get_current_detections()
            
            # Generate narration
            narration = self.narrator.narrate(detections)
            
            # Update UI if narration panel exists
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(narration)
        except Exception as e:
            print(f"Error generating narration: {e}")
            # Update UI with error message
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(
                    f"Error generating narration: {str(e)}"
                )

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        try:
            # Cancel any pending narration timers
            if hasattr(self, "_narration_timer_id"):
                self.after_cancel(self._narration_timer_id)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```


Updating requirements.txt and checking for errors:


```pip-requirements
# Visionary Project Dependencies
# CS4398 Group 3

# Core dependencies
opencv-python>=4.9.0
numpy>=1.25.0
Pillow>=10.0.0

# Hardware communication (future)
pyserial>=3.5

# Machine Learning / LLM (planned)
tensorflow>=2.13.0
torch>=2.0.0
transformers>=4.30.0

# LLM Integration for Vision Narrator
openai>=1.0.0

# GUI Framework (Tkinter included with Python)
# (No install required for most systems right now)

# Testing & Dev Tools
pytest>=7.4.0
pytest-cov>=4.1.0

```


Fixing a small issue and verifying the implementation:


```python
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


```


Verifying the control panel layout and fixing a potential issue:



Fixing callback setup order and ensuring the narration panel is created before triggering narration:


```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            narrator=self.narrator,
            on_narrate_request=self._trigger_narration
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Set up narrator callback to update UI (the narration panel should have already set this,
        # but we ensure it's set correctly here)
        if hasattr(self._control_widget, 'narration_panel'):
            # The narration panel should have already set up the callback, but we can verify
            pass

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")
    
    def _start_narration_timer(self, interval_ms=60000):
        """
        Start the periodic narration timer.
        
        Args:
            interval_ms: Interval in milliseconds (default: 60000 = 1 minute)
        """
        self._trigger_narration()
        # Schedule next narration
        self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _schedule_next_narration(self, interval_ms):
        """Schedule the next periodic narration."""
        if self.winfo_exists():  # Check if window still exists
            self._trigger_narration()
            self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _trigger_narration(self):
        """Trigger narration generation from current detections."""
        try:
            # Get current detections (placeholder for now)
            detections = get_current_detections()
            
            # Generate narration
            narration = self.narrator.narrate(detections)
            
            # Update UI if narration panel exists
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(narration)
        except Exception as e:
            print(f"Error generating narration: {e}")
            # Update UI with error message
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(
                    f"Error generating narration: {str(e)}"
                )

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        try:
            # Cancel any pending narration timers
            if hasattr(self, "_narration_timer_id"):
                self.after_cancel(self._narration_timer_id)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()

```


Creating a summary document and verifying everything works. Checking for a README in the ai directory:



Verifying the implementation by checking imports:



Fixing the PowerShell command:



Imports succeed. Creating a summary of the implementation:



## Summary

Vision Narrator implemented.

### Components

1. `src/ai/vision_narrator.py` — Core narrator module:
   - `DetectionResult` class for detection data
   - `VisionNarrator` class that:
     - Summarizes detections into text
     - Calls OpenAI API for human-friendly descriptions
     - Falls back to simple transformations if API unavailable
   - `get_placeholder_detections()` for testing (replace with YOLO later)

2. `src/vision/object_detection.py` — Placeholder detection function:
   - `get_current_detections()` returns placeholder data
   - Ready to replace with YOLO when merging branches

3. `src/ui/interface_layout/vision_narration_panel.py` — UI component:
   - Read-only text display for narration
   - "Narrate Now" button for on-demand narration
   - Status indicator

4. Integration:
   - Added to control panel
   - Integrated into main interface with automatic timer (every 60 seconds)
   - Manual trigger via button

5. Dependencies:
   - Added `openai>=1.0.0` to `requirements.txt`

### Features

- Automatic narration every 60 seconds
- Manual trigger via "Narrate Now"
- LLM integration with OpenAI API (fallback if unavailable)
- Placeholder detections ready for YOLO integration
- Isolated from servos/tracking
- Error handling with fallback descriptions

### Usage

1. Set OpenAI API key (optional):
   - Set `OPENAI_API_KEY` environment variable, or
   - The narrator falls back to simple transformations

2. Run the application:
   - Narration starts automatically on launch
   - Updates every 60 seconds
   - Use "Narrate Now" for immediate narration

### Next steps (when merging with Detection + Tracking branch)

Replace the placeholder in `src/vision/object_detection.py`:
- Replace `get_current_detections()` to call the YOLO detector
- Return `DetectionResult` objects from actual detections

No changes needed elsewhere; the narrator uses the same interface.

Implementation is complete and ready to test. Works with placeholder data and will integrate with real YOLO detections after merging branches.

