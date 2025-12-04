"""
Configuration Manager for Hardware Settings

Manages saving and loading of hardware configuration settings,
particularly servo offset values.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class HardwareConfig:
    """Manages hardware configuration settings, particularly servo offsets."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize hardware config manager.
        
        Args:
            config_file: Path to JSON config file (default: config.json in project root)
        """
        if config_file is None:
            # Default to config.json in project root
            project_root = Path(__file__).parent.parent.parent
            config_file = str(project_root / "hardware_config.json")
        
        self.config_file = config_file
        self._ensure_config_file()
    
    def _ensure_config_file(self):
        """Ensure config file exists, create with defaults if it doesn't."""
        config_path = Path(self.config_file)
        if not config_path.exists():
            # Create default config
            default_config = {
                "pan_offset": 0,
                "tilt_offset": 0
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
    
    def get_offsets(self) -> Tuple[int, int]:
        """
        Get saved pan and tilt offset values.
        
        Returns:
            (pan_offset, tilt_offset) tuple
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            pan_offset = config.get("pan_offset", 0)
            tilt_offset = config.get("tilt_offset", 0)
            return (int(pan_offset), int(tilt_offset))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return (0, 0)
    
    def save_offsets(self, pan_offset: int, tilt_offset: int):
        """
        Save pan and tilt offset values to config file.
        
        Args:
            pan_offset: Pan offset in degrees
            tilt_offset: Tilt offset in degrees
        """
        try:
            # Load existing config or create new
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                config = {}
            
            # Update offsets
            config["pan_offset"] = int(pan_offset)
            config["tilt_offset"] = int(tilt_offset)
            
            # Save back to file
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save offsets to config: {e}")

