"""
Target Selector Module

Implements intelligent target selection for servo tracking when multiple objects
are detected. Uses priority scoring and maintains persistence to avoid bouncing
between targets.
"""

from typing import List, Dict, Optional, Tuple
import time


class TargetSelector:
    """
    Selects the "most important" target from multiple detected objects.
    
    Priority factors:
    1. Currently tracked target (persistence)
    2. Object class (person > cell phone > book > others)
    3. Size (larger = more important)
    4. Confidence (higher = more reliable)
    5. Proximity to center (closer to center = easier to track)
    
    Once a target is selected, it persists until:
    - The target is lost (not detected for N frames)
    - A significantly better target appears (priority score difference > threshold)
    """
    
    # Class priority weights (higher = more important)
    CLASS_PRIORITIES = {
        "person": 100,
        "cell phone": 50,
        "book": 40,
        "laptop": 35,
        "mouse": 30,
        "keyboard": 25,
        "remote": 20,
    }
    
    DEFAULT_CLASS_PRIORITY = 10  # For unknown classes
    
    def __init__(
        self,
        persistence_frames: int = 10,
        priority_threshold: float = 20.0,
        center_weight: float = 0.3,
        size_weight: float = 0.4,
        confidence_weight: float = 0.2,
        class_weight: float = 0.1,
    ):
        """
        Initialize TargetSelector.
        
        Args:
            persistence_frames: Number of frames to keep tracking a target after it's lost
            priority_threshold: Minimum priority difference to switch targets
            center_weight: Weight for center proximity in scoring (0-1)
            size_weight: Weight for bounding box size in scoring (0-1)
            confidence_weight: Weight for detection confidence in scoring (0-1)
            class_weight: Weight for class priority in scoring (0-1)
        """
        self.persistence_frames = persistence_frames
        self.priority_threshold = priority_threshold
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.confidence_weight = confidence_weight
        self.class_weight = class_weight
        
        # Current target state
        self.current_target_id: Optional[int] = None
        self.frames_since_seen: int = 0
        self.current_target_priority: float = 0.0
    
    def select_target(
        self,
        tracked_objects: List[Dict],
        frame_width: int,
        frame_height: int
    ) -> Optional[Dict]:
        """
        Select the best target from tracked objects.
        
        Args:
            tracked_objects: List of detected objects with fields:
                - "id": track_id (persistent ID)
                - "cls": class name (e.g., "person")
                - "conf": confidence score (0-1)
                - "bbox": (x1, y1, x2, y2) bounding box
            frame_width: Width of the frame
            frame_height: Height of the frame
        
        Returns:
            Selected target object dict, or None if no suitable target
        """
        if not tracked_objects:
            # No objects detected - increment lost counter
            self.frames_since_seen += 1
            # If we've lost the target for too long, clear it
            if self.frames_since_seen > self.persistence_frames:
                self.current_target_id = None
                self.current_target_priority = 0.0
            return None
        
        # Calculate priority scores for all objects
        scored_targets = []
        current_target_found = False
        
        for obj in tracked_objects:
            score = self._calculate_priority(obj, frame_width, frame_height)
            scored_targets.append((score, obj))
            
            # Check if this is our current target
            if self.current_target_id is not None and obj.get("id") == self.current_target_id:
                current_target_found = True
                self.frames_since_seen = 0
        
        # If current target not found, increment lost counter
        if not current_target_found and self.current_target_id is not None:
            self.frames_since_seen += 1
        
        # Sort by priority (highest first)
        scored_targets.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_targets:
            return None
        
        best_score, best_target = scored_targets[0]
        
        # Decision logic:
        # 1. If we have a current target and it's still visible, prefer it (with bonus)
        if self.current_target_id is not None:
            for score, obj in scored_targets:
                if obj.get("id") == self.current_target_id:
                    # Current target found - give it persistence bonus
                    persistence_bonus = 15.0  # Bonus to keep current target
                    adjusted_score = score + persistence_bonus
                    
                    # Only switch if new target is significantly better
                    if best_target.get("id") != self.current_target_id:
                        if best_score > adjusted_score + self.priority_threshold:
                            # New target is much better - switch
                            self.current_target_id = best_target.get("id")
                            self.current_target_priority = best_score
                            return best_target
                        else:
                            # Keep current target
                            return obj
                    else:
                        # Current target is still the best
                        self.current_target_priority = adjusted_score
                        return obj
        
        # 2. If current target is lost for too long, select new best target
        if self.frames_since_seen > self.persistence_frames:
            self.current_target_id = best_target.get("id")
            self.current_target_priority = best_score
            self.frames_since_seen = 0
            return best_target
        
        # 3. If no current target, select best one
        if self.current_target_id is None:
            self.current_target_id = best_target.get("id")
            self.current_target_priority = best_score
            self.frames_since_seen = 0
            return best_target
        
        # 4. Current target lost but within persistence window - return None
        # (servos will hold last position)
        return None
    
    def _calculate_priority(
        self,
        obj: Dict,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculate priority score for an object.
        
        Higher score = more important target.
        
        Args:
            obj: Object dict with "cls", "conf", "bbox"
            frame_width: Frame width
            frame_height: Frame height
        
        Returns:
            Priority score (higher = better)
        """
        # Get object properties
        cls_name = obj.get("cls", "").lower()
        confidence = obj.get("conf", 0.0)
        x1, y1, x2, y2 = obj.get("bbox", (0, 0, 0, 0))
        
        # Calculate bounding box area (size)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame_width * frame_height
        size_ratio = area / frame_area if frame_area > 0 else 0
        
        # Calculate center proximity (distance from frame center)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        
        # Normalized distance from center (0 = center, 1 = corner)
        dx = abs(center_x - frame_center_x) / (frame_width / 2.0) if frame_width > 0 else 1.0
        dy = abs(center_y - frame_center_y) / (frame_height / 2.0) if frame_height > 0 else 1.0
        center_distance = (dx + dy) / 2.0
        center_proximity = 1.0 - center_distance  # Higher = closer to center
        
        # Get class priority
        class_priority = self.CLASS_PRIORITIES.get(cls_name, self.DEFAULT_CLASS_PRIORITY)
        normalized_class_priority = class_priority / 100.0  # Normalize to 0-1
        
        # Combine factors (all normalized to 0-1 range)
        score = (
            self.size_weight * size_ratio * 100 +  # Size: 0-100
            self.confidence_weight * confidence * 100 +  # Confidence: 0-100
            self.center_weight * center_proximity * 50 +  # Center proximity: 0-50
            self.class_weight * normalized_class_priority * 50  # Class: 0-50
        )
        
        return score
    
    def reset(self):
        """Reset target selector state (clear current target)."""
        self.current_target_id = None
        self.frames_since_seen = 0
        self.current_target_priority = 0.0
    
    def get_current_target_id(self) -> Optional[int]:
        """Get the ID of the currently tracked target."""
        return self.current_target_id

