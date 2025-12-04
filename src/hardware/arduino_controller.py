"""
Arduino Controller Module

This module handles serial communication with the Arduino hardware controller.
The Arduino manages servo motors (pan/tilt) and laser control for the Visionary system.

Communication Protocol:
    - Commands are sent as newline-terminated strings (e.g., "A,90,90\n")
    - Arduino responds with status messages (e.g., "OFFSETS,0,0")
    - Baud rate: 115200 (must match Arduino sketch)

Supported Commands:
    - A,pan,tilt: Auto-tracking servo angles (0-180)
    - M,0/1: Enable/disable auto-tracking mode
    - L,0/1: Laser on/off
    - O,pan_offset,tilt_offset: Set servo offsets
    - LED_ON/LED_OFF: LED control (for testing)

Hardware Configuration:
    - Pan Servo (Pin 10): Left-right movement, 0-180 degrees
    - Tilt Servo (Pin 9): Up-down movement, 0-180 degrees
    - Laser (Pin 13): Controlled via transistor
"""

import time
import serial
import serial.tools.list_ports
from typing import Optional, Tuple


class ArduinoController:
    """
    Manages serial communication with Arduino hardware controller.
    
    Handles:
    - Servo motor control (pan/tilt for camera gimbal)
    - Laser on/off control
    - Auto-tracking mode enable/disable
    - Servo offset calibration
    - Connection management with error recovery
    """
    
    def __init__(self, baudrate: int = 115200, timeout: float = 2.0, port: Optional[str] = None):
        """
        Initialize Arduino controller.
        
        Args:
            baudrate: Serial communication baud rate (must match Arduino sketch, default 115200)
            timeout: Read timeout in seconds (default 2.0)
            port: Optional explicit COM port (e.g., "COM3"). If None, auto-discovers Arduino
        """
        self.arduino: Optional[serial.Serial] = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.port = port

    def send_command(self, cmd: str):
        """
        Send a newline-terminated command to Arduino.
        
        Commands must match the Arduino sketch's command parser format.
        The Arduino reads commands using readStringUntil('\\n').
        
        Args:
            cmd: Command string to send (without newline, will be added automatically)
        
        Raises:
            RuntimeError: If Arduino is not connected
            serial.SerialException: If serial write fails
        """
        if not self.is_connected():
            raise RuntimeError("Arduino not connected. Call connect() first.")
        
        try:
            # Encode command as UTF-8 and append newline
            command_bytes = (cmd + "\n").encode("utf-8")
            self.arduino.write(command_bytes)
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to send command to Arduino: {e}")

    def is_connected(self) -> bool:
        """
        Check if Arduino is currently connected.
        
        Returns:
            True if serial port is open and connection is active, False otherwise
        """
        return bool(self.arduino) and getattr(self.arduino, "is_open", False)

    def connect(self) -> bool:
        """
        Open serial connection to Arduino.
        
        If port is specified, connects to that port. Otherwise, auto-discovers
        Arduino by searching for devices with "Arduino" or "CH340" in description
        (common Arduino USB-to-serial chips).
        
        Returns:
            True if connection successful, False if no Arduino found or connection failed
        
        Note:
            Waits 2 seconds after opening port to allow Arduino to reset and initialize.
        """
        # === Already Connected ===
        if self.is_connected():
            return True

        # === Connect to Specified Port ===
        if self.port:
            try:
                self.arduino = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                time.sleep(2)  # Wait for Arduino reset
                return True
            except (serial.SerialException, ValueError) as e:
                # Port doesn't exist, is in use, or invalid parameters
                if self.arduino:
                    try:
                        self.arduino.close()
                    except Exception:
                        pass
                    self.arduino = None
                return False

        # === Auto-Discover Arduino Port ===
        # Search for Arduino or CH340 devices (common USB-to-serial chips)
        for port_info in serial.tools.list_ports.comports():
            description = port_info.description or ""
            if "Arduino" in description or "CH340" in description:
                try:
                    self.arduino = serial.Serial(
                        port_info.device, 
                        self.baudrate, 
                        timeout=self.timeout
                    )
                    time.sleep(2)  # Wait for Arduino reset
                    self.port = port_info.device  # Remember the port for future reference
                    return True
                except (serial.SerialException, ValueError):
                    # Port might be in use or invalid, try next port
                    continue
        
        # No Arduino found
        return False

    def blink(self, times: int = 3, interval_s: float = 0.3):
        """
        Convenience helper: blink the LED for testing/status indication.
        
        Args:
            times: Number of blink cycles (default 3)
            interval_s: Time between LED on/off states in seconds (default 0.3)
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        if not self.is_connected():
            raise RuntimeError("Arduino not connected. Call connect() first.")
        
        for _ in range(times):
            self.send_command("LED_ON")
            time.sleep(interval_s)
            self.send_command("LED_OFF")
            time.sleep(interval_s)

    def disconnect(self):
        """
        Close serial connection to Arduino.
        
        Safely closes the connection and cleans up resources.
        Safe to call even if not connected.
        """
        if self.arduino:
            try:
                self.arduino.close()
            except Exception:
                # Ignore errors during cleanup
                pass
            finally:
                self.arduino = None

    # === Laser Control (Independent of Servo Tracking) ===
    
    def laser_on(self):
        """
        Turn laser on.
        
        Laser is controlled via Pin 13 on Arduino through a transistor circuit.
        This is independent of servo tracking and can be toggled separately.
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        self.send_command("L,1")

    def laser_off(self):
        """
        Turn laser off.
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        self.send_command("L,0")

    def laser_toggle(self, state: bool):
        """
        Toggle laser on or off based on boolean state.
        
        Args:
            state: True to turn laser on, False to turn off
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        if state:
            self.laser_on()
        else:
            self.laser_off()

    # === Servo Control Methods ===
    
    def servo_auto_track(self, pan: int, tilt: int):
        """
        Set servo angles in auto-tracking mode.
        
        This command only works if auto-tracking mode is enabled (see set_auto_tracking_mode).
        The Arduino automatically applies safety offsets to aim lower than center mass
        to avoid face/eye contact.
        
        Args:
            pan: Pan angle (0-180 degrees, left-right movement, Pin 10)
            tilt: Tilt angle (0-180 degrees, up-down movement, Pin 9)
        
        Note:
            Angles are automatically clamped to valid range (0-180).
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        # Clamp angles to valid servo range
        pan = max(0, min(180, int(pan)))
        tilt = max(0, min(180, int(tilt)))
        self.send_command(f"A,{pan},{tilt}")

    def servo_manual_control(self, pan: int, tilt: int):
        """
        Manual servo control (uses auto-tracking command format).
        
        This is a convenience method that uses the same command format as auto-tracking.
        Note: Auto-tracking mode must be enabled for this to work.
        
        Args:
            pan: Pan angle (0-180 degrees, left-right movement, Pin 10)
            tilt: Tilt angle (0-180 degrees, up-down movement, Pin 9)
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        # Clamp angles to valid servo range
        pan = max(0, min(180, int(pan)))
        tilt = max(0, min(180, int(tilt)))
        # Uses same command format as auto-tracking (requires auto mode enabled)
        self.send_command(f"A,{pan},{tilt}")

    def set_auto_tracking_mode(self, enabled: bool):
        """
        Enable or disable auto-tracking mode on Arduino.
        
        When disabled, auto-tracking commands (A,pan,tilt) are ignored by the Arduino.
        This allows manual control or prevents accidental servo movement.
        
        Args:
            enabled: True to enable auto-tracking, False to disable
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        # CompleteSketch uses M,0/1 for mode toggle
        mode_value = 1 if enabled else 0
        self.send_command(f"M,{mode_value}")

    # === Servo Offset Calibration Methods ===
    
    def set_offset(self, pan_offset: int, tilt_offset: int):
        """
        Set pan and tilt offset values on Arduino.
        
        Offsets are applied to all servo commands to compensate for camera/laser
        mounting misalignment. These are calibration values that adjust the "zero"
        position of the servos.
        
        Args:
            pan_offset: Pan offset in degrees (+ right, - left)
            tilt_offset: Tilt offset in degrees (+ down, - up)
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        pan_offset = int(pan_offset)
        tilt_offset = int(tilt_offset)
        self.send_command(f"O,{pan_offset},{tilt_offset}")

    def read_offset(self) -> Tuple[int, int]:
        """
        Read current offset values from Arduino.
        
        The Arduino sends "OFFSETS,pan,tilt" messages on startup and after
        offset changes. This method reads from the serial buffer to retrieve
        the current offsets.
        
        Returns:
            Tuple of (pan_offset, tilt_offset), or (0, 0) if unable to read
            or Arduino not connected
        
        Note:
            This method is non-blocking and may return (0, 0) if no offset
            message is available in the buffer.
        """
        if not self.is_connected():
            return (0, 0)
        
        try:
            # Read multiple lines to catch startup message or recent updates
            for _ in range(10):  # Read up to 10 lines
                if self.arduino.in_waiting > 0:
                    # Read a line (Arduino sends newline-terminated messages)
                    line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("OFFSETS,"):
                        # Parse OFFSETS,pan,tilt format
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                pan_offset = int(parts[1])
                                tilt_offset = int(parts[2])
                                return (pan_offset, tilt_offset)
                            except ValueError:
                                # Invalid integer in offset values
                                pass
                else:
                    # No more data available in buffer
                    break
        except (serial.SerialException, UnicodeDecodeError):
            # Serial read error or decode error - return default
            pass
        
        # Default: no offset (0, 0)
        return (0, 0)

    # === Legacy Methods (Backwards Compatibility) ===
    # These methods are kept for compatibility with older code but are not
    # actively used by the current CompleteSketch Arduino firmware.
    
    def servo_start(self):
        """
        Legacy method: not used by CompleteSketch firmware.
        
        Kept for backwards compatibility with older code.
        """
        pass

    def servo_stop(self):
        """
        Legacy method: not used by CompleteSketch firmware.
        
        Kept for backwards compatibility with older code.
        """
        pass

    def servo_set_angle(self, angle: int):
        """
        Legacy method: sets both servos to the same angle.
        
        This is a convenience method for backwards compatibility.
        Sets both pan and tilt servos to the same angle.
        
        Args:
            angle: Angle in degrees (0-180) for both servos
        
        Raises:
            RuntimeError: If Arduino is not connected
        """
        angle = max(0, min(180, int(angle)))
        # Use manual control to set both servos to same angle
        self.servo_manual_control(angle, angle)
