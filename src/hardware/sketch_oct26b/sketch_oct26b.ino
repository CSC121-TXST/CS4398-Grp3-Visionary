#include <Servo.h>

Servo s;
const int SERVO_PIN = 9;
bool running = false;

void setup() {
  Serial.begin(9600);            // <-- matches ArduinoController default
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // Read one line (blocking a little is fine at 9600; adjust if you prefer non-blocking)
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length()) handleCommand(line);
  }

  // If running, do a gentle sweep so you can see motion hands-free
  if (running) {
    static unsigned long t0 = 0;
    static int pos = 0, dir = 1;
    if (millis() - t0 > 20) {
      t0 = millis();
      pos += dir;
      if (pos >= 180) { pos = 180; dir = -1; }
      if (pos <= 0)   { pos = 0;   dir = 1;  }
      s.write(pos);
    }
  }
}

void handleCommand(const String& cmd) {
  if (cmd == "LED_ON") {
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.println("OK LED_ON");
  } else if (cmd == "LED_OFF") {
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("OK LED_OFF");
  } else if (cmd == "START") {
    if (!s.attached()) s.attach(SERVO_PIN);
    running = true;
    Serial.println("OK START");
  } else if (cmd == "STOP") {
    running = false;
    if (s.attached()) s.detach();
    Serial.println("OK STOP");
  } else if (cmd.startsWith("ANGLE:")) {
    int angle = cmd.substring(6).toInt(); // ANGLE:0..180
    angle = constrain(angle, 0, 180);
    if (!s.attached()) s.attach(SERVO_PIN);
    running = false;            // stop sweeping if you set a fixed angle
    s.write(angle);
    Serial.println("OK ANGLE");
  } else {
    Serial.println("ERR UNKNOWN");
  }
}
