#include <Servo.h>

// Pin assignments
const int SERVO_TILT_PIN = 9;   // Top servo: up/down
const int SERVO_PAN_PIN  = 10;  // Bottom servo: left/right
const int LASER_PIN      = 13;  // Laser control (through transistor)

// Servo objects
Servo tiltServo;
Servo panServo;

// Current state
int panAngle  = 90;   // Center-ish
int tiltAngle = 90;   // Center-ish
bool laserOn  = false;
bool autoMoveEnabled = true;  // Can toggle via serial

// Limits (tune as needed so you don't smash your LEGO rig)
const int PAN_MIN  = 30;
const int PAN_MAX  = 150;
const int TILT_MIN = 40;
const int TILT_MAX = 140;

// Offsets (to compensate for camera/laser not perfectly aligned)
// You'll tweak these after a bit of live testing.
int panOffsetDegrees  = 0;   // + right, - left
int tiltOffsetDegrees = 5;   // + down, - up (aim a bit lower to avoid eyes)

void setup() {
  Serial.begin(115200);

  panServo.attach(SERVO_PAN_PIN);
  tiltServo.attach(SERVO_TILT_PIN);
  pinMode(LASER_PIN, OUTPUT);

  // Initialize positions
  applyServoAngles();
  digitalWrite(LASER_PIN, LOW);
}

void loop() {
  // Handle any incoming serial commands
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      handleCommand(line);
    }
  }

  // You could add small smoothing/slow return here if you want.

  // Tiny delay just to not hammer the servos
  delay(5);
}

void handleCommand(const String& cmd) {
  // Commands:
  //  "A,pan,tilt"  -> set pan/tilt angles (if autoMoveEnabled)
  //  "L,0" / "L,1" -> laser off/on
  //  "M,0" / "M,1" -> autoMove off/on

  if (cmd.startsWith("A,")) {
    if (!autoMoveEnabled) return;

    // Parse A,pan,tilt
    int firstComma = cmd.indexOf(',');
    int secondComma = cmd.indexOf(',', firstComma + 1);
    if (secondComma == -1) return;  // malformed

    String panStr  = cmd.substring(firstComma + 1, secondComma);
    String tiltStr = cmd.substring(secondComma + 1);

    int targetPan  = panStr.toInt();
    int targetTilt = tiltStr.toInt();

    // Apply offsets so we correct for camera/laser mounting + aim a bit low
    targetPan  += panOffsetDegrees;
    targetTilt += tiltOffsetDegrees;

    // Clamp to safe ranges
    panAngle  = constrain(targetPan,  PAN_MIN,  PAN_MAX);
    tiltAngle = constrain(targetTilt, TILT_MIN, TILT_MAX);

    applyServoAngles();
  }
  else if (cmd.startsWith("L,")) {
    // Laser toggle
    if (cmd.endsWith("1")) {
      laserOn = true;
      digitalWrite(LASER_PIN, HIGH);
    } else if (cmd.endsWith("0")) {
      laserOn = false;
      digitalWrite(LASER_PIN, LOW);
    }
  }
  else if (cmd.startsWith("M,")) {
    // Auto movement toggle
    if (cmd.endsWith("1")) {
      autoMoveEnabled = true;
    } else if (cmd.endsWith("0")) {
      autoMoveEnabled = false;
    }
  }
}

void applyServoAngles() {
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}
