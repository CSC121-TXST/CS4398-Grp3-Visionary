# src/test_arduino.py
from hardware.arduino_controller import ArduinoController
import time

def main():
    ard = ArduinoController()
    if not ard.connect():
        print("No Arduino found. Tip: check Tools->Port in Arduino IDE for the COM port.")
        return

    print("Turning LED ON for 2 seconds...")
    ard.send_command("LED_ON")
    time.sleep(2)

    print("Turning LED OFF...")
    ard.send_command("LED_OFF")

    ard.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()
