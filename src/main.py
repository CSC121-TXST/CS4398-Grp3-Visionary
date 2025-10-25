#!/usr/bin/env python3
"""
Visionary Project - Main Application Entry Point

This is the main entry point for the Visionary project, which integrates
computer vision, hardware control, and AI capabilities for autonomous
object detection and tracking.

"""

from ui.main_interface import VisionaryApp

# from Vision --> Camera Control
# from Vision --> Object Detection
# from Hardware --> Hardware Control? Specific to Arduino vs Other Hardware?

if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()