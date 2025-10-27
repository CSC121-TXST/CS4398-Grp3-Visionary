# src/hardware/arduino_controller.py
import time
import serial
import serial.tools.list_ports

class ArduinoController:
    def __init__(self, baudrate=9600, timeout=2, port=None):
        """
        baudrate: must match Serial.begin(...) in Arduino sketch (9600)
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
