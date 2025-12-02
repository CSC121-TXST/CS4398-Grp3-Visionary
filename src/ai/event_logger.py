"""
Event Logger Module

Manages event logs for narration periods, similar to security camera event logs.
Each narration period becomes a logged event with timestamp, description, and metadata.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class EventLogger:
    """
    Logs narration periods as events with timestamps and descriptions.
    Similar to security camera event logs.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize event logger.
        
        Args:
            log_file: Path to JSON log file (default: events.json in project root)
        """
        if log_file is None:
            # Default to events.json in project root
            project_root = Path(__file__).parent.parent.parent
            log_file = str(project_root / "events.json")
        
        self.log_file = log_file
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file exists, create if it doesn't."""
        log_path = Path(self.log_file)
        if not log_path.exists():
            # Create empty log file with empty array
            with open(log_path, 'w') as f:
                json.dump([], f, indent=2)
    
    def log_event(
        self,
        description: str,
        duration_seconds: float,
        detection_summary: str,
        unique_objects: int,
        detection_count: int
    ) -> Dict:
        """
        Log a narration period as an event.
        
        Args:
            description: LLM-generated description of the event
            duration_seconds: How long the period lasted
            detection_summary: Formatted detection summary (e.g., "2 persons, 1 book")
            unique_objects: Number of unique object types detected
            detection_count: Number of detection frames collected
        
        Returns:
            Dictionary representing the logged event
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "narration_period",
            "duration_seconds": round(duration_seconds, 2),
            "description": description,
            "detection_summary": detection_summary,
            "unique_objects": unique_objects,
            "detection_frames": detection_count
        }
        
        # Load existing events
        events = self._load_events()
        
        # Add new event
        events.append(event)
        
        # Save back to file
        self._save_events(events)
        
        print(f"DEBUG: Event logged: {event['timestamp']}")
        return event
    
    def _load_events(self) -> List[Dict]:
        """Load events from log file."""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_events(self, events: List[Dict]):
        """Save events to log file."""
        with open(self.log_file, 'w') as f:
            json.dump(events, f, indent=2)
    
    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent events.
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            List of event dictionaries, most recent first
        """
        events = self._load_events()
        return events[-limit:][::-1]  # Return last N, reversed (newest first)
    
    def get_all_events(self) -> List[Dict]:
        """Get all logged events."""
        return self._load_events()
    
    def clear_events(self):
        """Clear all events from the log."""
        self._save_events([])
        print("DEBUG: Event log cleared")

