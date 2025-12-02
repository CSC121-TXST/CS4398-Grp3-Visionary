int laserPin = 13;   // Signal wire to the transistor base (through resistor)

void setup() {
  pinMode(laserPin, OUTPUT);
}

void loop() {
  digitalWrite(laserPin, HIGH);  // Turn laser ON
  delay(1000);                   // 1 second ON

  digitalWrite(laserPin, LOW);   // Turn laser OFF
  delay(1000);                   // 1 second OFF
}
