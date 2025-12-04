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

// Limits (tune as needed so we don't smash the LEGO rig)
const int PAN_MIN  = 30;
const int PAN_MAX  = 150;
const int TILT_MIN = 40;
const int TILT_MAX = 140;

// Offsets (to compensate for camera/laser not perfectly aligned)
// These can now be changed LIVE from the app via the "O,panOffset,tiltOffset" command.
int panOffsetDegrees  = 0;   // + right, - left (logical)
int tiltOffsetDegrees = 0;   // + down, - up (logical)

void setup() {
  Serial.begin(115200);

  panServo.attach(SERVO_PAN_PIN);
  tiltServo.attach(SERVO_TILT_PIN);
  pinMode(LASER_PIN, OUTPUT);

  // Initialize positions
  applyServoAngles();
  digitalWrite(LASER_PIN, LOW);

  Serial.println("Visionary Servo Controller Ready");
  Serial.print("OFFSETS,");
  Serial.print(panOffsetDegrees);
  Serial.print(",");
  Serial.println(tiltOffsetDegrees);
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

  // Tiny delay just to not hammer the servos
  delay(5);
}

void handleCommand(const String& cmd) {
  // Commands:
  //  "A,pan,tilt"      -> set pan/tilt angles (if autoMoveEnabled)
  //  "L,0" / "L,1"     -> laser off/on
  //  "M,0" / "M,1"     -> autoMove off/on
  //  "O,pan,tilt"      -> set panOffsetDegrees, tiltOffsetDegrees (in degrees)
  //
  // Example:
  //  O,4,0     -> panOffset = +4°, tiltOffset = 0°
  //  O,-2,3    -> panOffset = -2°, tiltOffset = +3°

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

    // Apply offsets so we correct for camera/laser mounting + safety tilt
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
  else if (cmd.startsWith("O,")) {
    // Offset update: O,panOffset,tiltOffset

    int firstComma = cmd.indexOf(',');
    int secondComma = cmd.indexOf(',', firstComma + 1);

    // If only one value is provided, treat it as panOffset only
    if (secondComma == -1) {
      String panStr = cmd.substring(firstComma + 1);
      panOffsetDegrees = panStr.toInt();
    } else {
      String panStr  = cmd.substring(firstComma + 1, secondComma);
      String tiltStr = cmd.substring(secondComma + 1);

      panOffsetDegrees  = panStr.toInt();
      tiltOffsetDegrees = tiltStr.toInt();
    }

    // Echo back for debugging
    Serial.print("OFFSETS,");
    Serial.print(panOffsetDegrees);
    Serial.print(",");
    Serial.println(tiltOffsetDegrees);
  }
}

void applyServoAngles() {
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}
