# src/hardware/arduino_controller.py
import time
import serial
import serial.tools.list_ports

class ArduinoController:
    def __init__(self, baudrate=115200, timeout=2, port=None):
        """
        baudrate: must match Serial.begin(...) in Arduino sketch (115200)
        timeout: read timeout in seconds
        port: optional explicit COM port (e.g., "COM3"); if None we auto-discover
        """
        self.arduino = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.port = port

    def send_command(self, cmd: str):
        """Send a newline-terminated command (matches readStringUntil('\\n') on Arduino)."""
        if not self.arduino:
            raise RuntimeError("Arduino not connected")
        self.arduino.write((cmd + "\n").encode("utf-8"))

    def is_connected(self) -> bool:
        return bool(self.arduino) and getattr(self.arduino, "is_open", False)

    def connect(self) -> bool:
        """Open the serial port (auto-discover if port is not specified)."""
        if self.is_connected():
            return True

        if self.port:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            return True

        for p in serial.tools.list_ports.comports():
            if "Arduino" in p.description or "CH340" in p.description:
                self.arduino = serial.Serial(p.device, self.baudrate, timeout=self.timeout)
                time.sleep(2)
                return True
        return False

    def blink(self, times: int = 3, interval_s: float = 0.3):
        """Convenience helper: blink the LED via commands."""
        if not self.is_connected():
            raise RuntimeError("Arduino not connected")
        for _ in range(times):
            self.send_command("LED_ON")
            time.sleep(interval_s)
            self.send_command("LED_OFF")
            time.sleep(interval_s)

    def disconnect(self):
        if self.arduino:
            try:
                self.arduino.close()
            finally:
                self.arduino = None

    # --- Laser control (independent of servo tracking) ---
    def laser_on(self):
        """Turn laser on (Pin 13, via transistor)."""
        self.send_command("L,1")

    def laser_off(self):
        """Turn laser off."""
        self.send_command("L,0")

    def laser_toggle(self, state: bool):
        """Toggle laser on or off based on state."""
        if state:
            self.laser_on()
        else:
            self.laser_off()

    # --- Servo control methods for CompleteSketch ---
    def servo_auto_track(self, pan: int, tilt: int):
        """
        Set servo angles in auto-tracking mode (only works if auto mode enabled).
        Offsets are applied automatically on Arduino side to aim lower than center mass.
        
        Args:
            pan: Pan angle (0-180, left-right, Pin 10)
            tilt: Tilt angle (0-180, up-down, Pin 9)
        """
        pan = max(0, min(180, int(pan)))
        tilt = max(0, min(180, int(tilt)))
        self.send_command(f"A,{pan},{tilt}")

    def servo_manual_control(self, pan: int, tilt: int):
        """
        Manual override: set servo angles using auto-tracking command.
        Note: This requires auto mode to be enabled. If disabled, enable it first.
        
        Args:
            pan: Pan angle (0-180, left-right, Pin 10)
            tilt: Tilt angle (0-180, up-down, Pin 9)
        """
        pan = max(0, min(180, int(pan)))
        tilt = max(0, min(180, int(tilt)))
        # Use A,pan,tilt command - it works when auto mode is enabled
        self.send_command(f"A,{pan},{tilt}")

    def set_auto_tracking_mode(self, enabled: bool):
        """
        Enable or disable auto-tracking mode.
        When disabled, auto-tracking commands (A,pan,tilt) are ignored.
        
        Args:
            enabled: True to enable auto-tracking, False to disable
        """
        # CompleteSketch uses M,0/1 for mode toggle (not AUTO,)
        self.send_command(f"M,{1 if enabled else 0}")

    # --- Offset control methods ---
    def set_offset(self, pan_offset: int, tilt_offset: int):
        """
        Set pan and tilt offset values on Arduino.
        These offsets are applied to all servo commands to compensate for
        camera/laser mounting misalignment.
        
        Args:
            pan_offset: Pan offset in degrees (+ right, - left)
            tilt_offset: Tilt offset in degrees (+ down, - up)
        """
        pan_offset = int(pan_offset)
        tilt_offset = int(tilt_offset)
        self.send_command(f"O,{pan_offset},{tilt_offset}")

    def read_offset(self) -> tuple[int, int]:
        """
        Read current offset values from Arduino.
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        
        This method attempts to read any pending OFFSETS message from the serial buffer.
        It reads multiple lines to catch the startup message.
        
        Returns:
            (pan_offset, tilt_offset) tuple, or (0, 0) if unable to read
        """
        if not self.is_connected():
            return (0, 0)
        
        try:
            # Read multiple lines to catch startup message
            for _ in range(10):  # Read up to 10 lines
                if self.arduino.in_waiting > 0:
                    # Read a line (Arduino sends newline-terminated messages)
                    line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("OFFSETS,"):
                        # Parse OFFSETS,pan,tilt
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                pan_offset = int(parts[1])
                                tilt_offset = int(parts[2])
                                return (pan_offset, tilt_offset)
                            except ValueError:
                                pass
                else:
                    break  # No more data available
        except Exception:
            pass
        
        return (0, 0)

    # --- Legacy servo helpers for backwards compatibility ---
    def servo_start(self):
        """Legacy: not used by CompleteSketch, but kept for compatibility."""
        pass

    def servo_stop(self):
        """Legacy: not used by CompleteSketch, but kept for compatibility."""
        pass

    def servo_set_angle(self, angle: int):
        """Legacy: sets both servos to same angle (for backwards compatibility)."""
        angle = max(0, min(180, int(angle)))
        # Use manual control as fallback
        self.servo_manual_control(angle, angle)
