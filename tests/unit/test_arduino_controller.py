"""
Unit tests for Arduino Controller component.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hardware.arduino_controller import ArduinoController


@pytest.mark.unit
@pytest.mark.hardware
class TestArduinoController:
    """Test suite for ArduinoController class."""
    
    def test_initialization_default(self):
        """Test ArduinoController initialization with default parameters."""
        controller = ArduinoController()
        assert controller.baudrate == 115200
        assert controller.timeout == 2
        assert controller.port is None
        assert controller.arduino is None
    
    def test_initialization_custom(self):
        """Test ArduinoController initialization with custom parameters."""
        controller = ArduinoController(baudrate=9600, timeout=5, port="COM3")
        assert controller.baudrate == 9600
        assert controller.timeout == 5
        assert controller.port == "COM3"
    
    def test_is_connected_not_connected(self):
        """Test is_connected when not connected."""
        controller = ArduinoController()
        assert controller.is_connected() == False
    
    def test_is_connected_connected(self):
        """Test is_connected when connected."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.is_open = True
        controller.arduino = mock_arduino
        assert controller.is_connected() == True
    
    @patch('hardware.arduino_controller.serial.Serial')
    def test_connect_with_explicit_port(self, mock_serial_class):
        """Test connecting with explicit port."""
        mock_arduino = MagicMock()
        mock_arduino.is_open = True
        mock_serial_class.return_value = mock_arduino
        
        controller = ArduinoController(port="COM3")
        result = controller.connect()
        
        assert result == True
        mock_serial_class.assert_called_once_with("COM3", 115200, timeout=2)
        assert controller.arduino == mock_arduino
    
    @patch('hardware.arduino_controller.serial.Serial')
    @patch('hardware.arduino_controller.serial.tools.list_ports.comports')
    @patch('time.sleep')
    def test_connect_auto_discover_arduino(self, mock_sleep, mock_comports, mock_serial_class):
        """Test auto-discovery of Arduino port."""
        mock_port = MagicMock()
        mock_port.description = "Arduino Uno"
        mock_port.device = "COM4"
        mock_comports.return_value = [mock_port]
        
        mock_arduino = MagicMock()
        mock_arduino.is_open = True
        mock_serial_class.return_value = mock_arduino
        
        controller = ArduinoController()
        result = controller.connect()
        
        assert result == True
        mock_serial_class.assert_called_once_with("COM4", 115200, timeout=2)
        assert controller.arduino == mock_arduino
    
    @patch('hardware.arduino_controller.serial.Serial')
    @patch('hardware.arduino_controller.serial.tools.list_ports.comports')
    @patch('time.sleep')
    def test_connect_auto_discover_ch340(self, mock_sleep, mock_comports, mock_serial_class):
        """Test auto-discovery of CH340 device."""
        mock_port = MagicMock()
        mock_port.description = "CH340"
        mock_port.device = "COM5"
        mock_comports.return_value = [mock_port]
        
        mock_arduino = MagicMock()
        mock_arduino.is_open = True
        mock_serial_class.return_value = mock_arduino
        
        controller = ArduinoController()
        result = controller.connect()
        
        assert result == True
        mock_serial_class.assert_called_once_with("COM5", 115200, timeout=2)
    
    @patch('hardware.arduino_controller.serial.tools.list_ports.comports')
    def test_connect_auto_discover_no_device(self, mock_comports):
        """Test auto-discovery when no Arduino device found."""
        mock_comports.return_value = []
        
        controller = ArduinoController()
        result = controller.connect()
        
        assert result == False
        assert controller.arduino is None
    
    def test_connect_already_connected(self):
        """Test connect when already connected."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.is_open = True
        controller.arduino = mock_arduino
        
        result = controller.connect()
        assert result == True
    
    def test_send_command_not_connected(self):
        """Test send_command when not connected raises error."""
        controller = ArduinoController()
        with pytest.raises(RuntimeError, match="Arduino not connected"):
            controller.send_command("TEST")
    
    def test_send_command_connected(self):
        """Test send_command when connected."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.send_command("LED_ON")
        mock_arduino.write.assert_called_once_with(b"LED_ON\n")
    
    @patch('time.sleep')
    def test_blink(self, mock_sleep):
        """Test blink LED command."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.blink(times=2, interval_s=0.1)
        
        # Should call LED_ON and LED_OFF twice
        assert mock_arduino.write.call_count == 4
        calls = [str(call) for call in mock_arduino.write.call_args_list]
        assert b"LED_ON\n" in [call[0][0] for call in mock_arduino.write.call_args_list]
        assert b"LED_OFF\n" in [call[0][0] for call in mock_arduino.write.call_args_list]
    
    def test_blink_not_connected(self):
        """Test blink when not connected raises error."""
        controller = ArduinoController()
        with pytest.raises(RuntimeError, match="Arduino not connected"):
            controller.blink()
    
    def test_laser_on(self):
        """Test laser on command."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.laser_on()
        mock_arduino.write.assert_called_once_with(b"L,1\n")
    
    def test_laser_off(self):
        """Test laser off command."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.laser_off()
        mock_arduino.write.assert_called_once_with(b"L,0\n")
    
    def test_laser_toggle_on(self):
        """Test laser toggle to on."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.laser_toggle(True)
        mock_arduino.write.assert_called_once_with(b"L,1\n")
    
    def test_laser_toggle_off(self):
        """Test laser toggle to off."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.laser_toggle(False)
        mock_arduino.write.assert_called_once_with(b"L,0\n")
    
    def test_servo_auto_track(self):
        """Test servo auto-tracking command."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.servo_auto_track(pan=90, tilt=75)
        mock_arduino.write.assert_called_once_with(b"A,90,75\n")
    
    def test_servo_auto_track_clamping(self):
        """Test servo auto-tracking with angle clamping."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        # Test values outside valid range
        controller.servo_auto_track(pan=-10, tilt=200)
        mock_arduino.write.assert_called_once_with(b"A,0,180\n")
        
        # Test float values
        controller.servo_auto_track(pan=90.7, tilt=75.3)
        assert mock_arduino.write.call_count == 2
        # Check last call
        last_call = mock_arduino.write.call_args_list[-1]
        assert last_call[0][0] == b"A,90,75\n"
    
    def test_servo_manual_control(self):
        """Test servo manual control command."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.servo_manual_control(pan=100, tilt=80)
        mock_arduino.write.assert_called_once_with(b"A,100,80\n")
    
    def test_set_auto_tracking_mode_enabled(self):
        """Test enabling auto-tracking mode."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.set_auto_tracking_mode(True)
        mock_arduino.write.assert_called_once_with(b"M,1\n")
    
    def test_set_auto_tracking_mode_disabled(self):
        """Test disabling auto-tracking mode."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.set_auto_tracking_mode(False)
        mock_arduino.write.assert_called_once_with(b"M,0\n")
    
    def test_set_offset(self):
        """Test setting servo offsets."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.set_offset(pan_offset=5, tilt_offset=-3)
        mock_arduino.write.assert_called_once_with(b"O,5,-3\n")
    
    def test_set_offset_float_values(self):
        """Test setting offsets with float values (should be converted to int)."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.set_offset(pan_offset=5.7, tilt_offset=-3.2)
        mock_arduino.write.assert_called_once_with(b"O,5,-3\n")
    
    def test_read_offset_not_connected(self):
        """Test reading offsets when not connected."""
        controller = ArduinoController()
        result = controller.read_offset()
        assert result == (0, 0)
    
    def test_read_offset_no_data(self):
        """Test reading offsets when no data available."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.in_waiting = 0
        controller.arduino = mock_arduino
        
        result = controller.read_offset()
        assert result == (0, 0)
    
    def test_read_offset_success(self):
        """Test successfully reading offsets."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.in_waiting = 10
        mock_arduino.readline.return_value = b"OFFSETS,5,-3\n"
        controller.arduino = mock_arduino
        
        result = controller.read_offset()
        assert result == (5, -3)
    
    def test_read_offset_invalid_format(self):
        """Test reading offsets with invalid format."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.in_waiting = 10
        mock_arduino.readline.return_value = b"INVALID_FORMAT\n"
        controller.arduino = mock_arduino
        
        result = controller.read_offset()
        assert result == (0, 0)
    
    def test_read_offset_multiple_lines(self):
        """Test reading offsets when multiple lines in buffer."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        # First call returns non-matching line, second returns OFFSETS
        mock_arduino.in_waiting = 20
        mock_arduino.readline.side_effect = [
            b"DEBUG: Starting\n",
            b"OFFSETS,10,20\n"
        ]
        controller.arduino = mock_arduino
        
        result = controller.read_offset()
        assert result == (10, 20)
    
    def test_servo_set_angle_legacy(self):
        """Test legacy servo_set_angle method."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.servo_set_angle(90)
        mock_arduino.write.assert_called_once_with(b"A,90,90\n")
    
    def test_disconnect(self):
        """Test disconnecting Arduino."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        controller.arduino = mock_arduino
        
        controller.disconnect()
        mock_arduino.close.assert_called_once()
        assert controller.arduino is None
    
    def test_disconnect_with_exception(self):
        """Test disconnect when close raises exception."""
        controller = ArduinoController()
        mock_arduino = MagicMock()
        mock_arduino.close.side_effect = Exception("Close error")
        controller.arduino = mock_arduino
        
        # Should not raise, should set arduino to None
        controller.disconnect()
        assert controller.arduino is None

