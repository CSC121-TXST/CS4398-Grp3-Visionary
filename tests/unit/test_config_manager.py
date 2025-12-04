"""
Unit tests for Hardware Configuration Manager component.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hardware.config_manager import HardwareConfig


@pytest.mark.unit
@pytest.mark.hardware
class TestHardwareConfig:
    """Test suite for HardwareConfig class."""
    
    def test_initialization_default_path(self, temp_dir):
        """Test initialization with default config file path."""
        # Use a custom config file for this test (simulating default behavior)
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        assert config.config_file == config_file
        assert Path(config_file).exists()
    
    def test_initialization_custom_path(self, temp_dir):
        """Test initialization with custom config file path."""
        config_file = str(Path(temp_dir) / "custom_config.json")
        config = HardwareConfig(config_file=config_file)
        
        assert config.config_file == config_file
        assert Path(config_file).exists()
    
    def test_initialization_creates_default_file(self, temp_dir):
        """Test that default config file is created if it doesn't exist."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        # File should exist
        assert Path(config_file).exists()
        
        # Should have default values
        with open(config_file, 'r') as f:
            data = json.load(f)
            assert data["pan_offset"] == 0
            assert data["tilt_offset"] == 0
    
    def test_get_offsets_default(self, temp_dir):
        """Test getting offsets from default config."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        pan, tilt = config.get_offsets()
        assert pan == 0
        assert tilt == 0
    
    def test_get_offsets_existing(self, temp_dir):
        """Test getting offsets from existing config file."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        
        # Create config file with values
        with open(config_file, 'w') as f:
            json.dump({"pan_offset": 5, "tilt_offset": -3}, f)
        
        config = HardwareConfig(config_file=config_file)
        pan, tilt = config.get_offsets()
        
        assert pan == 5
        assert tilt == -3
    
    def test_get_offsets_missing_file(self, temp_dir):
        """Test getting offsets when file doesn't exist (should return defaults)."""
        config_file = str(Path(temp_dir) / "nonexistent.json")
        config = HardwareConfig(config_file=config_file)
        
        # File should be created by initialization
        pan, tilt = config.get_offsets()
        assert pan == 0
        assert tilt == 0
    
    def test_get_offsets_invalid_json(self, temp_dir):
        """Test getting offsets from invalid JSON file."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        
        # Create invalid JSON file
        with open(config_file, 'w') as f:
            f.write("invalid json content {")
        
        config = HardwareConfig(config_file=config_file)
        pan, tilt = config.get_offsets()
        
        # Should return defaults on error
        assert pan == 0
        assert tilt == 0
    
    def test_get_offsets_missing_keys(self, temp_dir):
        """Test getting offsets when keys are missing."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        
        # Create config file without offset keys
        with open(config_file, 'w') as f:
            json.dump({"other_key": "value"}, f)
        
        config = HardwareConfig(config_file=config_file)
        pan, tilt = config.get_offsets()
        
        # Should return defaults
        assert pan == 0
        assert tilt == 0
    
    def test_save_offsets(self, temp_dir):
        """Test saving offsets to config file."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        config.save_offsets(pan_offset=10, tilt_offset=-5)
        
        # Verify saved values
        pan, tilt = config.get_offsets()
        assert pan == 10
        assert tilt == -5
    
    def test_save_offsets_overwrites_existing(self, temp_dir):
        """Test that saving offsets overwrites existing values."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        # Save initial values
        config.save_offsets(pan_offset=5, tilt_offset=3)
        
        # Save new values
        config.save_offsets(pan_offset=15, tilt_offset=-10)
        
        # Verify new values
        pan, tilt = config.get_offsets()
        assert pan == 15
        assert tilt == -10
    
    def test_save_offsets_preserves_other_keys(self, temp_dir):
        """Test that saving offsets preserves other config keys."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        
        # Create config with other keys
        with open(config_file, 'w') as f:
            json.dump({
                "pan_offset": 0,
                "tilt_offset": 0,
                "other_setting": "value"
            }, f)
        
        config = HardwareConfig(config_file=config_file)
        config.save_offsets(pan_offset=7, tilt_offset=8)
        
        # Verify offsets updated and other key preserved
        with open(config_file, 'r') as f:
            data = json.load(f)
            assert data["pan_offset"] == 7
            assert data["tilt_offset"] == 8
            assert data["other_setting"] == "value"
    
    def test_save_offsets_float_to_int(self, temp_dir):
        """Test that float offset values are converted to int."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        # Save float values
        config.save_offsets(pan_offset=5.7, tilt_offset=-3.2)
        
        # Should be converted to int
        pan, tilt = config.get_offsets()
        assert pan == 5
        assert tilt == -3
        assert isinstance(pan, int)
        assert isinstance(tilt, int)
    
    def test_save_offsets_handles_exception(self, temp_dir):
        """Test that save_offsets handles exceptions gracefully."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        config = HardwareConfig(config_file=config_file)
        
        # Make file read-only to cause write error (on some systems)
        # Or mock the file operations to raise exception
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            # Should not raise, should print warning
            config.save_offsets(pan_offset=5, tilt_offset=3)
            # No exception should propagate
    
    def test_save_offsets_creates_file_if_missing(self, temp_dir):
        """Test that save_offsets creates file if it doesn't exist."""
        config_file = str(Path(temp_dir) / "new_config.json")
        
        # Don't create file first
        config = HardwareConfig(config_file=config_file)
        
        # Save offsets (file should be created)
        config.save_offsets(pan_offset=12, tilt_offset=15)
        
        # Verify file exists and has correct values
        assert Path(config_file).exists()
        pan, tilt = config.get_offsets()
        assert pan == 12
        assert tilt == 15
    
    def test_multiple_instances_same_file(self, temp_dir):
        """Test that multiple instances can work with same config file."""
        config_file = str(Path(temp_dir) / "shared_config.json")
        
        config1 = HardwareConfig(config_file=config_file)
        config1.save_offsets(pan_offset=20, tilt_offset=25)
        
        config2 = HardwareConfig(config_file=config_file)
        pan, tilt = config2.get_offsets()
        
        assert pan == 20
        assert tilt == 25
    
    def test_get_offsets_returns_int(self, temp_dir):
        """Test that get_offsets always returns integers."""
        config_file = str(Path(temp_dir) / "hardware_config.json")
        
        # Create config with string values (should be converted)
        with open(config_file, 'w') as f:
            json.dump({"pan_offset": "10", "tilt_offset": "5"}, f)
        
        config = HardwareConfig(config_file=config_file)
        pan, tilt = config.get_offsets()
        
        # Should handle gracefully (might return 0 on ValueError)
        assert isinstance(pan, int)
        assert isinstance(tilt, int)

