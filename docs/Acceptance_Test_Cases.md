# Acceptance Test Cases
## Visionary System – CS4398 Group 3

**Project:** Visionary – Autonomous Object Tracking System  
**Purpose of this document:**  
Describe how we executed a small set of acceptance test cases for the Visionary system, including the main end-to-end flow (launch → detect → track → aim → shutdown) and the key hardware behaviors, supported by screenshots of the main windows and pop-ups.

---

## 1. Test Environment

**Hardware**
- USB webcam connected to the computer
- Arduino Uno with the Visionary firmware uploaded
- Two servos connected (Pin 9: Tilt, Pin 10: Pan)
- Laser module connected through a transistor circuit (Pin 13)
- USB cable between Arduino and computer

**Software**
- Python 3.8
- All project dependencies installed (`pip install -r requirements.txt`)
- YOLO and DeepSORT models in place

Test objects: a person that can be detected.

---

## 2. Acceptance Test Cases (Overview)

We executed three main acceptance test cases:

1. **1: End-to-End System Flow**  
   Launch app → start camera → enable tracking → track a target → adjust offsets → toggle laser → shutdown.

2. **2: Hardware Connection & Servo/Laser Control**  
   Connect/disconnect Arduino, manually move servos, and toggle laser.

3. **3: Error Handling & Clean Shutdown**  
   Handle loss of Arduino/camera gracefully and close the application without crashing.

Each test case below includes the steps, expected results, and a short screenshot checklist.

---

## 3. Test Case 1: End-to-End System Flow

**Goal:**  
Verify that Visionary can go from a cold start to fully tracking a target with the laser aligned (after offsets), and then shut down cleanly.

**Preconditions:**
- Webcam and Arduino are plugged in.
- Servos and laser are wired correctly.
- Models and dependencies are installed.

### Steps

1. **Launch Application**
   - Run `python src/main.py` (or use your launcher).
   - Wait for the main Visionary window to appear.

   **Expected:**  
   - Main window opens without errors.  
   - Dark theme (or default theme) is visible.  
   - Status bar shows initial status (e.g., camera OFF, Arduino not connected).

2. **Start Camera**
   - Click the button to start the camera (e.g., “Start Camera”).
   - Wait for the live video feed to appear.

   **Expected:**  
   - Live camera feed is visible in the video panel.  
   - FPS or status indicator updates to show the camera is active.

3. **Connect Arduino**
   - Click “Connect Arduino” (or equivalent) in the hardware/controls panel.

   **Expected:**  
   - Connection succeeds, no error dialog.  
   - Status shows “Arduino: Connected” (or similar).  
   - Hardware controls (servo/laser/auto-tracking) become enabled.

4. **Enable Detection and Tracking**
   - Turn on detection/tracking (e.g., toggle “Enable Tracking”).
   - Place yourself or an object in front of the camera.

   **Expected:**  
   - Bounding box appears around the detected object.  
   - The tracked target is clearly highlighted (label, ID, etc.).  
   - Status shows something like “Tracking: 1 object”.

5. **Enable Auto-Tracking (Servo Movement)**
   - Turn on auto-tracking / hardware tracking.
   - Move the object slowly around the frame.

   **Expected:**  
   - Pan/tilt servos move to keep the target near the center of the camera view.  
   - Movement is smooth and stays within safe angle limits.

6. **Adjust Offsets (Calibration)**
   - Open the offset controls in the GUI.  
   - Adjust pan/tilt offsets until the laser spot is close to the center of the tracked target in the video feed.

   **Expected:**  
   - Offsets take effect without restarting the program.  
   - The physical laser converges toward the point the UI considers “center” of the target.

7. **Toggle Laser**
   - Turn the laser ON via the UI.  
   - Confirm the laser is on, aimed at the tracked target.  
   - Turn the laser OFF again.

   **Expected:**  
   - Laser turns on and off reliably.  
   - While ON, the laser follows the same target the servos are tracking.

8. **Stop Tracking and Shut Down**
   - Turn off auto-tracking and detection (if separate).  
   - Stop the camera.  
   - Close the application using the window close button or menu.

   **Expected:**  
   - Camera feed stops.  
   - Serial connection to Arduino is closed cleanly.  
   - No crash or error dialog during shutdown.  
   - Process exits completely.

### Screenshot Summary:
1. **1_1_MainWindow.png** – Main window right after launch.  
2. **1_2_CameraFeed.png** – Live video feed running.  
3. **1_3_TrackingActive.png** – Target detected with bounding box, tracking enabled.  
4. **1_4_AutoTrackingServo.png** – Target near center while servos are clearly tracking.  
5. **1_5_OffsetControls.png** – Offset/ calibration UI visible with some non-zero values.  
6. **1_6_LaserOn.png** – Laser ON while tracking (photo of hardware + GUI, or split).  
7. **1_7_BeforeShutdown.png** – App right before closing, with status showing things are stopped.

---

## 4. Test Case 2: Hardware Connection & Control

**Goal:**  
Verify that the app can connect to the Arduino, move the servos manually, and toggle the laser without errors.

**Preconditions:**
- App is running.  
- Camera is optional for this test.

### Steps

1. **Connect Arduino**
   - Click “Connect Arduino”.

   **Expected:**  
   - Status indicates “Connected”.  
   - No error dialogs.

2. **Manual Servo Movement**
   - Use any manual servo controls (sliders/inputs) to set pan and tilt (for example, both to 90°).  

   **Expected:**  
   - Servos move to the requested angles.  
   - Movements stay within safe limits.

3. **Toggle Laser**
   - Click “Laser ON”, then “Laser OFF”.

   **Expected:**  
   - Laser responds to commands.  
   - Status text (if present) reflects ON/OFF state.

4. **Disconnect Arduino**
   - Click “Disconnect Arduino”.  

   **Expected:**  
   - Status shows “Not Connected”.  
   - Hardware controls are disabled or show clearly that Arduino is not connected.

### Screenshot Summary:
1. **2_1_HardwareDisconnected.png** – Hardware panel before connection (showing “Connect Arduino”).  
2. **2_2_HardwareConnected.png** – Hardware panel after connection.  
3. **2_3_ServoControl.png** – Manual servo controls visible with some angle entered.  
4. **2_4_LaserControl.png** – Laser ON/OFF controls visible (laser ON if safe to show).  

---

## 5. Test Case 3: Error Handling & Clean Shutdown

**Goal:**  
Show that the system fails gracefully when hardware is missing and that it can be shut down cleanly.

### Steps

1. **Arduino Not Connected Scenario**
   - Start the app without plugging in the Arduino.  
   - Try to enable auto-tracking or send a hardware command (servo or laser).

   **Expected:**  
   - User is informed that Arduino is not connected (status message or popup).  
   - The app does **not** crash.

2. **Camera Failure / Not Available (Optional)**
   - Start the app with no webcam available *or* unplug the camera and then try to start the feed.

   **Expected:**  
   - A clear error or status message is shown.  
   - The app can keep running or be closed normally.

3. **Shutdown After Error**
   - After an error scenario (no camera or no Arduino), close the app.

   **Expected:**  
   - App shuts down cleanly without hanging or crashing.

### Screenshot Summary:


---

## 6. Test Results Summary

| Test ID | Description                     | Status (Pass/Fail) | Notes                                      |
|--------:|---------------------------------|--------------------|--------------------------------------------|
| AT-01   | End-to-End System Flow          |        Pass        |                                            |
| AT-02   | Hardware Connection & Control   |        Pass        |                                            |
| AT-03   | Error Handling & Clean Shutdown |        Pass        | Some discrepancies given Arduino version   |

---

All referenced screenshots should be saved under a `screenshots/` folder (e.g., `screenshots/AT01_1_MainWindow.png`, etc.) and included in the project archive.
