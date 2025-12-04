# Visionary System Architecture

This document provides a comprehensive overview of the Visionary codebase architecture, component relationships, and data flow.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Module Dependencies](#module-dependencies)
5. [Key Algorithms](#key-algorithms)
6. [File Structure](#file-structure)

---

## System Overview

Visionary is an autonomous object tracking system that combines:
- **Computer Vision**: YOLOv8 object detection + DeepSORT multi-object tracking
- **Hardware Control**: Arduino-based servo motors and laser control
- **AI Narration**: OpenAI-powered scene description
- **GUI**: Tkinter-based user interface

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionaryApp (Main Window)                │
│                                                               │
│  ┌──────────────┐         ┌─────────────────────────────┐  │
│  │ VideoPanel   │         │  Sidebar (Controls)          │  │
│  │              │         │  - ControlPanel             │  │
│  │  Camera ────►│         │  - NarrationPanel           │  │
│  │  Tracker     │         │  - Theme Toggle             │  │
│  │  Display     │         └─────────────────────────────┘  │
│  └──────────────┘                                         │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              StatusBar (Footer)                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Vision Pipeline

**Location**: `src/vision/`

#### ObjectTracker (`tracking.py`)
- **Purpose**: Integrates YOLOv8 detection with DeepSORT tracking
- **Key Methods**:
  - `process_frame()`: Main entry point for frame processing
  - `_filter_and_prepare_detections()`: Filters detections by class and confidence
  - `_update_tracker()`: Updates DeepSORT tracker with new detections
  - `_draw_tracking_results()`: Draws bounding boxes and track IDs

**Data Flow**:
```
Camera Frame (BGR) 
  → YOLOv8 Detection 
  → Class Filtering 
  → DeepSORT Tracking 
  → Annotated Frame + Tracked Objects List
```

#### SimpleCamera (`camera_control.py`)
- **Purpose**: Manages webcam capture and frame scheduling
- **Key Components**:
  - `SimpleCamera`: Handles OpenCV VideoCapture and frame processing
  - `CanvasRenderer`: Renders frames onto Tkinter Canvas

**Frame Processing Pipeline**:
```
Camera → OpenCV VideoCapture → Mirroring → Frame Processor → Renderer → Display
```

### 2. Hardware Control Pipeline

**Location**: `src/hardware/`

#### ArduinoController (`arduino_controller.py`)
- **Purpose**: Serial communication with Arduino hardware
- **Commands**:
  - `A,pan,tilt`: Auto-tracking servo angles
  - `M,0/1`: Enable/disable auto-tracking mode
  - `L,0/1`: Laser on/off
  - `O,pan_offset,tilt_offset`: Set servo offsets

#### ServoTracker (`servo_tracking.py`)
- **Purpose**: Converts camera coordinates to servo angles
- **Key Algorithm**:
  - Calculates center of bounding box
  - Applies vertical offset (aims below center mass for safety)
  - Converts pixel coordinates to normalized coordinates
  - Maps to servo angle range (0-180 degrees)

**Coordinate Transformation**:
```
Image Coordinates (x1, y1, x2, y2)
  → Center Calculation
  → Vertical Offset Application
  → Normalized Coordinates (-0.5 to 0.5)
  → Angle Offset Calculation
  → Servo Angles (pan, tilt)
```

#### TargetSelector (`target_selector.py`)
- **Purpose**: Intelligent target selection from multiple detected objects
- **Priority Factors**:
  1. Currently tracked target (persistence bonus)
  2. Object class (person > cell phone > book > others)
  3. Size (larger = more important)
  4. Confidence (higher = more reliable)
  5. Proximity to center (closer = easier to track)

**Selection Algorithm**:
```
All Detected Objects
  → Calculate Priority Score (class + size + confidence + center proximity)
  → Apply Persistence Bonus to Current Target
  → Select Best Target (with threshold for switching)
  → Return Selected Target or None
```

### 3. AI Narration System

**Location**: `src/ai/`

#### VisionNarrator (`vision_narrator.py`)
- **Purpose**: Generates natural language descriptions of detected scenes
- **Workflow**:
  1. Collects detections during a "narration period"
  2. Formats detections for LLM
  3. Sends to OpenAI API for description generation
  4. Returns formatted description

**Data Flow**:
```
Detections (List[Dict]) 
  → Format for LLM 
  → OpenAI API Call 
  → Natural Language Description
```

### 4. User Interface

**Location**: `src/ui/`

#### VisionaryApp (`main_interface.py`)
- **Purpose**: Main application window orchestrating all components
- **Key Responsibilities**:
  - Initialize all system components
  - Handle UI callbacks and state management
  - Coordinate data flow between components
  - Manage application lifecycle

#### VideoPanel (`interface_layout/video_panel.py`)
- **Purpose**: Displays camera feed with tracking overlay
- **Integration**:
  - Receives frames from SimpleCamera
  - Processes through ObjectTracker
  - Displays annotated frames
  - Triggers servo tracking callbacks

#### ControlPanel (`interface_layout/control_panel.py`)
- **Purpose**: Hardware and system controls
- **Sections**:
  - Camera Controls (start/stop, resolution)
  - Hardware Controls (Arduino connection, servo control, laser)
  - Tracking Controls (enable/disable tracking)
  - Narration Controls (start/stop narration period)

---

## Data Flow

### Complete Tracking Pipeline

```
┌─────────┐
│ Camera  │
└────┬────┘
     │ BGR Frame
     ▼
┌─────────────────┐
│ SimpleCamera    │ (Mirroring, Scheduling)
└────┬────────────┘
     │ BGR Frame
     ▼
┌─────────────────┐
│ ObjectTracker   │
│  - YOLOv8        │ → Detections
│  - Class Filter  │ → Filtered Detections
│  - DeepSORT      │ → Tracked Objects
└────┬────────────┘
     │ Annotated Frame + Tracked Objects
     ▼
┌─────────────────┐
│ VideoPanel      │ → Display
└────┬────────────┘
     │ Tracked Objects
     ▼
┌─────────────────┐
│ TargetSelector  │ → Selected Target
└────┬────────────┘
     │ Target BBox
     ▼
┌─────────────────┐
│ ServoTracker    │ → Pan/Tilt Angles
└────┬────────────┘
     │ Servo Angles
     ▼
┌─────────────────┐
│ ArduinoController│ → Serial Command
└────┬────────────┘
     │
     ▼
┌─────────┐
│ Arduino │ → Servo Motors + Laser
└─────────┘
```

### Narration Pipeline

```
┌─────────────────┐
│ ObjectTracker   │ → Detections
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ VisionNarrator  │
│  - Collect      │ → Detection History
│  - Format       │ → LLM Prompt
│  - Generate     │ → Description
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ NarrationPanel  │ → Display Description
└─────────────────┘
```

---

## Module Dependencies

### Dependency Graph

```
VisionaryApp (main_interface.py)
├── VideoPanel
│   ├── SimpleCamera (camera_control.py)
│   │   └── CanvasRenderer (camera_control.py)
│   └── ObjectTracker (tracking.py)
│       ├── YOLOv8 (ultralytics)
│       └── DeepSORT (deep_sort_pytorch)
│
├── ControlPanel
│   ├── ArduinoController (arduino_controller.py)
│   └── SimpleCamera
│
├── ServoTracker (servo_tracking.py)
│   └── (Standalone - no dependencies)
│
├── TargetSelector (target_selector.py)
│   └── (Standalone - no dependencies)
│
└── VisionNarrator (vision_narrator.py)
    ├── LLM Integration (llm_integration.py)
    └── Event Logger (event_logger.py)
```

### DeepSORT Integration

**Location**: `src/vision/deep_sort_pytorch/deep_sort_pytorch/`

**Used Components**:
- `deep_sort/deep_sort.py`: Main DeepSORT class
- `deep_sort/deep/feature_extractor.py`: Feature extraction for tracking
- `deep_sort/deep/model.py`: Neural network model
- `deep_sort/sort/tracker.py`: Multi-object tracker
- `deep_sort/sort/kalman_filter.py`: Kalman filtering for motion prediction
- `deep_sort/sort/nn_matching.py`: Nearest neighbor matching
- `deep_sort/sort/linear_assignment.py`: Hungarian algorithm for assignment
- `deep_sort/sort/iou_matching.py`: IoU-based matching
- `deep_sort/sort/detection.py`: Detection data structure
- `deep_sort/sort/track.py`: Track data structure

## Key Algorithms

### 1. DeepSORT Tracking Algorithm
**Location**: `src/vision/deep_sort_pytorch/`

**Process**:
1. **Feature Extraction**: Extract appearance features from detected bounding boxes
2. **Kalman Prediction**: Predict track positions using Kalman filtering
3. **Matching Cascade**: Match detections to tracks using:
   - IoU matching (for confirmed tracks)
   - Feature matching (cosine distance)
   - Hungarian algorithm for optimal assignment
4. **Track Management**: 
   - Update matched tracks
   - Create new tracks for unmatched detections
   - Delete tracks that haven't been seen for `max_age` frames

### 2. Target Selection Algorithm
**Location**: `src/hardware/target_selector.py`

**Priority Score Calculation**:
```python
score = (
    size_weight * size_ratio * 100 +
    confidence_weight * confidence * 100 +
    center_weight * center_proximity * 50 +
    class_weight * normalized_class_priority * 50
)
```

**Selection Logic**:
1. Calculate priority scores for all detected objects
2. If current target exists and is visible:
   - Apply persistence bonus (+15.0)
   - Only switch if new target score > (current + bonus + threshold)
3. If current target lost:
   - Keep tracking for `persistence_frames` frames
   - After persistence window, select new best target
4. Return selected target or None

### 3. Servo Angle Calculation
**Location**: `src/hardware/servo_tracking.py`

**Algorithm**:
1. Calculate bounding box center: `(cx, cy) = ((x1+x2)/2, (y1+y2)/2)`
2. Apply vertical offset: `target_y = cy + (bbox_height * offset_vertical)`
3. Normalize coordinates: `norm_x = (cx - img_w/2) / img_w`
4. Convert to angle offsets: `pan_offset = -norm_x * fov_horizontal`
5. Apply to center angles: `pan_angle = pan_center + pan_offset`
6. Clamp to safe range: `pan_angle = clip(pan_angle, pan_min, pan_max)`

---

## File Structure

```
src/
├── main.py                    # Application entry point
│
├── vision/                    # Computer vision components
│   ├── tracking.py            # ObjectTracker (YOLOv8 + DeepSORT)
│   ├── camera_control.py      # SimpleCamera + CanvasRenderer
│   └── deep_sort_pytorch/     # DeepSORT implementation
│       └── deep_sort_pytorch/
│           └── deep_sort/     # Core tracking algorithms
│
├── hardware/                  # Hardware control components
│   ├── arduino_controller.py  # Serial communication with Arduino
│   ├── servo_tracking.py      # Camera-to-servo coordinate conversion
│   ├── target_selector.py     # Intelligent target selection
│   └── config_manager.py      # Hardware configuration management
│
├── ui/                        # User interface components
│   ├── main_interface.py      # Main application window
│   ├── style.py               # Theme and styling
│   └── interface_layout/      # UI layout components
│       ├── video_panel.py     # Video display panel
│       ├── control_panel.py   # Control buttons panel
│       ├── narration_panel.py # AI narration display
│       └── ...
│
└── ai/                        # AI and narration components
    ├── vision_narrator.py     # Scene description generation
    ├── llm_integration.py      # OpenAI API integration
    └── event_logger.py        # Event logging
```

---

## Component Communication

### Callback Chain
1. **Camera Frame Processing**:
   ```
   SimpleCamera._read_one()
     → frame_processor (VideoPanel._process_frame)
       → ObjectTracker.process_frame()
         → on_tracking_update (status bar update)
         → on_narration_detection (if narration active)
         → on_servo_tracking (if auto-tracking enabled)
   ```

2. **Servo Tracking**:
   ```
   VideoPanel._process_frame()
     → on_servo_tracking()
       → VisionaryApp._on_servo_tracking()
         → TargetSelector.select_target()
         → ServoTracker.bbox_to_angles()
         → ArduinoController.servo_auto_track()
   ```

3. **Narration**:
   ```
   VideoPanel._process_frame()
     → on_narration_detection()
       → VisionaryApp._on_narration_detection()
         → VisionNarrator.add_detection_to_period()
   ```

---

## Configuration

### Key Configuration Files
- `hardware_config.json`: Hardware settings (servo ranges, offsets, etc.)
- `yolov8n.pt`: YOLOv8 model weights (downloaded automatically)
- `src/vision/deep_sort_pytorch/.../checkpoint/ckpt.t7`: DeepSORT feature extractor weights

## Performance Considerations

### Frame Processing Optimization
- **Frame Skipping**: Process every Nth frame (configurable via `process_interval`)
- **Image Size**: Smaller images = faster processing (configurable via `imgsz`)
- **Confidence Threshold**: Higher threshold = fewer detections = faster tracking

### Memory Management
- **Frame Caching**: Last processed frame cached for skipped frames
- **DeepSORT Feature Budget**: Limited to `nn_budget` features per track
- **Track Cleanup**: Old tracks automatically deleted after `max_age` frames


## References
- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/nwojke/deep_sort
- Original Implementation: https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking

