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
- [ ] Establish stable USB webcam communication
- [ ] Implement Arduino-servo motor control system
- [ ] Develop object detection algorithms
- [ ] Create autonomous tracking functionality
- [ ] Integrate laser pointing system
- [ ] Implement LLM for enhanced interaction
- [ ] Design and develop user interface
- [ ] Conduct comprehensive testing and optimization

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   USB Webcam    │    │   Arduino       │    │   Servo Motors  │
│                 │◄──►│   Controller    │◄──►│   (Pan/Tilt)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Visionary Software                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐             │
│  │   Computer  │  │   Object    │  │   LLM        │             │
│  │   Vision    │  │   Tracking  │  │   Integration│             │
│  └─────────────┘  └─────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Software
- **Programming Language**: Python (primary), C++ (Arduino)
- **Computer Vision**: OpenCV, YOLO, or similar object detection frameworks
- **Machine Learning**: TensorFlow/PyTorch for LLM integration
- **Hardware Communication**: PySerial for Arduino communication
- **GUI Framework**: Tkinter, PyQt, or web-based interface

### Hardware
- **Microcontroller**: Arduino Uno/Nano
- **Servo Motors**: Standard hobby servos for pan/tilt functionality
- **Camera**: USB webcam with good resolution and frame rate
- **Laser Module**: Low-power laser pointer for object indication

## Project Structure

```
Visionary/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── vision/
│   │   ├── object_detection.py
│   │   ├── tracking.py
│   │   └── camera_control.py
│   ├── hardware/
│   │   ├── arduino_controller.py
│   │   ├── servo_control.py
│   │   └── laser_control.py
│   ├── ai/
│   │   ├── llm_integration.py
│   │   └── learning_system.py
│   └── ui/
│       ├── main_interface.py
│       └── control_panel.py
├── arduino/
│   └── visionary_controller.ino
├── docs/
│   ├── setup_guide.md
│   ├── user_manual.md
│   └── technical_specs.md
└── tests/
    ├── unit_tests/
    └── integration_tests/
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
1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Upload Arduino code to microcontroller
4. Connect hardware components
5. Run the main application: `python src/main.py`

## Group Members

| Net ID |       Name        |                   "Role"                    |
|--------|-------------------|---------------------------------------------|
| csc121 | Christopher Clark |                                             |
| oaa54  | Caden Dengel      | Hardware Integration / Arduino Development  |
| d_e39  | Dillon Emberson   |                                             |
| yus2   | Dylan Fennell     |                                             |

Internal Note: "Role" may not be necessary, but may highlight some of the ways we want to divide the work? I.E. Caden taking on a lot of the CAD design and actual hardware aspect. We can remove this though if we want.

## Project Timeline

- **Sprint 1**: Hardware setup and basic communication (Weeks 7-8)
- **Sprint 2**: Object detection implementation (Weeks 9-10)
- **Sprint 3**: Tracking and servo control (Weeks 11-12)
- **Sprint 4**: LLM integration and learning features (Weeks 13-14)
- **Sprint 5**: Testing, Final Integration, and Documentation (Week 15)

## Contributing

This is a class project for CS4398 Software Engineering. All development follows standard software engineering practices including version control, testing, and documentation.

## License

This project is developed for educational purposes as part of CS4398 Software Engineering Project class.

## Contact

For questions or collaboration, please contact the project team through course communication channels.