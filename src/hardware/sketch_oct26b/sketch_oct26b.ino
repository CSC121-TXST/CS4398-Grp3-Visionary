#include <Servo.h>

Servo servo1;
Servo servo2;

void setup() {
  servo1.attach(9);
  servo2.attach(10);
}

void loop() {
  // Servo 1 sweeps slowly
  for (int pos = 0; pos <= 180; pos++) {
    servo1.write(pos);
    delay(10);
  }
  for (int pos = 180; pos >= 0; pos--) {
    servo1.write(pos);
    delay(10);
  }

  // Servo 2 sweeps independently afterward
  for (int pos = 0; pos <= 180; pos++) {
    servo2.write(pos);
    delay(10);
  }
  for (int pos = 180; pos >= 0; pos--) {
    servo2.write(pos);
    delay(10);
  }
}
