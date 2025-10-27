const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
  Serial.println("Visionary ready");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "LED_ON") {
      digitalWrite(LED_PIN, HIGH);
      Serial.println("LED turned ON");
    } else if (cmd == "LED_OFF") {
      digitalWrite(LED_PIN, LOW);
      Serial.println("LED turned OFF");
    } else {
      Serial.println("Unknown command: " + cmd);
    }
  }
}
