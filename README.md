# CS4398 - Grp3 - Visionary

## Overview
**Visionary** is an innovative Software/Hardware integration project developed for our CS4398 Software Engineering Project class. This project combines computer vision, robotics, and artificial intelligence to create an intelligent camera system capable of autonomous object detection, tracking, and interaction.

## Project Description
Visionary is a comprehensive system that integrates a USB webcam with Arduino-controlled servo motors, enabling the camera to pan, tilt, and automatically point at detected objects. The system incorporates advanced computer vision algorithms and machine learning capabilities to identify and track objects in real-time, with an integrated LLM (Large Language Model) for enhanced communication and learning capabilities.

## Key Features

### Hardware Components
- **USB Webcam**: Primary visual input device
- **Arduino Controller**: Manages servo motor movements
- **Servo Motors**: Enable camera pan and tilt functionality
- **Laser Pointer**: Automated pointing system for object indication

### Software Capabilities
- **Real-time Object Detection**: Advanced computer vision algorithms to identify objects
- **Autonomous Tracking**: Automatic camera movement to follow detected objects
- **LLM Integration**: Built-in language model for enhanced interaction and learning
- **Intelligent Pointing**: Laser-guided object indication system
- **User Interface**: Intuitive control and monitoring system

## Development Goals

### Primary Objectives
1. **Seamless Hardware Integration**: Create a robust connection between software and hardware components
2. **Computer Vision**: Implement object detection and tracking algorithms
3. **Autonomous Operation**: Develop a system that can operate with minimal human intervention
4. **Learning Capabilities**: Integrate LLM for adaptive behavior and improved performance over time
5. **User Experience**: Design an intuitive interface for system control and monitoring

### Technical Milestones
- [X] Establish stable USB webcam communication
- [X] Implement Arduino-servo motor control system
- [X] Develop object detection algorithms (YOLOv8)
- [X] Create autonomous tracking functionality (DeepSORT)
- [X] Integrate laser pointing system
- [ ] Implement LLM for enhanced interaction (OpenAI Narrator, partial Implementation)
- [X] Design and develop user interface
- [X] Conduct comprehensive testing and optimization

## System Architecture

```
┌─────────────┐
│ USB Webcam  │
└──────┬──────┘
       │ BGR Frames
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Visionary Software (Python)                     │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ YOLOv8       │───►│ DeepSORT     │───►│ Target       │  │
│  │ Detection    │    │ Tracking     │    │ Selection    │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                   │          │
│  ┌──────────────┐    ┌──────────────┐           │          │
│  │ Servo        │◄───│ Coordinate   │◄──────────┘          │
│  │ Tracker      │    │ Conversion   │                       │
│  └──────┬───────┘    └──────────────┘                       │
│         │                                                      │
│  ┌──────▼───────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Arduino      │    │ Vision       │    │ Tkinter      │  │
│  │ Controller   │    │ Narrator     │    │ UI           │  │
│  └──────┬───────┘    │ (OpenAI)     │    └──────────────┘  │
└─────────┼────────────┴──────────────┴───────────────────────┘
          │ Serial Commands
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
│   Arduino       │───►│   Servo Motors  │    │   Laser      │
│   Controller    │    │   (Pan/Tilt)    │    │   Pointer    │
└─────────────────┘    └─────────────────┘    └──────────────┘
```

For detailed architecture documentation, see [ArchitectureGuide.md].

## Technology Stack

### Software
- **Programming Language**: Python 3.8+ (primary), C++ (Arduino)
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Object Tracking**: DeepSORT (multi-object tracking with Kalman filtering)
- **Hardware Communication**: PySerial for Arduino communication
- **GUI Framework**: Tkinter
- **Testing**: pytest with coverage reporting

### Hardware (If you want to test the REAL fun part)
- **Microcontroller**: Arduino Uno R3
- **Servo Motors**: Standard hobby servos for pan/tilt functionality
- **Camera**: USB webcam with good resolution and frame rate
- **Laser Module**: Low-power laser pointer for object indication

## Project Structure

```
Visionary/
├── README.md
├── ArchitectureGuide.md          # Detailed architecture documentation
├── requirements.txt
├── pytest.ini                    # Pytest configuration
├── hardware_config.json          # Hardware configuration
├── Visionary.bat                 # Quick launch script
├── yolov8n.pt                    # YOLOv8 model weights
│
├── src/
│   ├── main.py                   # Application entry point
│   │
│   ├── vision/                   # Computer vision components
│   │   ├── tracking.py           # ObjectTracker (YOLOv8 + DeepSORT)
│   │   ├── camera_control.py     # SimpleCamera + CanvasRenderer
│   │   └── deep_sort_pytorch/    # DeepSORT implementation
│   │       └── deep_sort_pytorch/
│   │           └── deep_sort/    # Core tracking algorithms
│   │
│   ├── hardware/                 # Hardware control
│   │   ├── arduino_controller.py # Serial communication with Arduino
│   │   ├── servo_tracking.py     # Camera-to-servo coordinate conversion
│   │   ├── target_selector.py    # Intelligent target selection
│   │   ├── config_manager.py     # Hardware configuration management
│   │   └── CompleteSketch/       # Arduino firmware
│   │       └── CompleteSketch.ino
│   │
│   ├── ai/                       # AI and narration
│   │   ├── vision_narrator.py    # Scene description generation
│   │   ├── llm_integration.py    # OpenAI API integration
│   │   ├── event_logger.py       # Event logging
│   │   └── tts_engine.py         # Text-to-speech (optional)
│   │
│   └── ui/                       # User interface
│       ├── main_interface.py     # Main application window
│       ├── style.py               # Theme and styling
│       └── interface_layout/     # UI components
│           ├── video_panel.py    # Video display
│           ├── control_panel.py   # Control buttons
│           ├── narration_panel.py # AI narration display
│           ├── menubar.py        # Menu bar
│           ├── status_bar.py     # Status indicators
│           └── control_buttons/  # Button components
│
├── docs/                         # Documentation
│   ├── setup_guide.md
│   ├── user_manual.md
│   └── technical_specs.md
│
└── tests/                        # Test suite
    ├── README.md                 # Test documentation
    ├── conftest.py               # Pytest fixtures
    ├── run_tests.py              # Test runner
    ├── unit/                     # Unit tests
    └── integration/              # Integration tests
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Arduino IDE
- USB webcam
- Arduino microcontroller
- Servo motors
- Laser pointer module

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CS4398-Grp3-Visionary
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API (optional, for narration features)**
   - Create a `.env` file in the `src/` directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - If not set, narration features will be disabled but the app will still run

4. **Upload Arduino firmware**
   - Open `src/hardware/CompleteSketch/CompleteSketch.ino` in Arduino IDE
   - Upload to your Arduino Uno
   - Ensure baud rate is set to 115200

5. **Connect hardware components**
   - Connect USB webcam to computer
   - Connect Arduino via USB
   - Connect servo motors to pins 9 (tilt) and 10 (pan)
   - Connect laser to pin 13 (via transistor)

6. **Run the application**
   ```bash
   python src/main.py
   ```
   Or double-click `Visionary.bat` for quick launch (Windows)

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

See [tests/README.md](tests/README.md) for detailed testing documentation.


## Group Members
| Net ID |       Name        |
|--------|-------------------|
| csc121 | Christopher Clark |
| oaa54  | Caden Dengel      |
| d_e39  | Dillon Emberson   |
| yus2   | Dylan Fennell     |

## Project Timeline

- **Sprint 1**: Hardware setup and basic communication (Weeks 7-8)
- **Sprint 2**: Object detection implementation (Weeks 9-10)
- **Sprint 3**: Tracking and servo control (Weeks 11-12)
- **Sprint 4**: LLM integration and learning features (Weeks 13-14)
- **Sprint 5**: Testing, Final Integration, and Documentation (Week 15)

## Documentation

- **[ArchitectureGuide.md](ArchitectureGuide.md)**: Comprehensive architecture documentation
- **[tests/README.md](tests/README.md)**: Testing documentation

## Contributing

This is a class project for CS4398 Software Engineering. All development follows standard software engineering practices including version control, testing, and documentation.

## License

This project is developed for educational purposes as part of CS4398 Software Engineering Project class.

## Contact

For questions or collaboration, please contact the project team through course communication channels.