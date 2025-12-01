"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from hardware.arduino_controller import ArduinoController

# Tracking Module
from vision.tracking import ObjectTracker


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Debug flag
        self.debug_enabled = False

        # Initialize ObjectTracker before building UI
        # This creates the tracker once and reuses it throughout the app
        self.tracker = ObjectTracker(
            model_path="yolov8n.pt",
            conf=0.35,
            target_classes=["person", "cell phone"]
        )
        # Ensure tracker starts with debug disabled
        if hasattr(self.tracker, "set_debug"):
            self.tracker.set_debug(self.debug_enabled)

        # Build UI 
        build_menubar(
            self,
            on_exit=self.on_exit,
            on_about=self._show_about,
            on_toggle_debug=self._on_toggle_debug,
<<<<<<< HEAD
            on_change_detection_classes=self._on_change_detection_classes
=======
            on_set_performance=self._on_set_performance
>>>>>>> Performance_Optimization
        )
        build_title(self)

        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Create VideoPanel with tracker
        # Pass the tracker so it can process frames
        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}"),
            tracker=self.tracker  # Pass tracker to VideoPanel
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        # Create ControlPanel with tracking toggle callback
        # Only pass on_toggle_tracking if tracker is available
        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            on_toggle_tracking=self._on_toggle_tracking if self.tracker is not None else None
        )
        self._control_widget.pack(expand=True, fill="both")
    
    def _on_toggle_tracking(self, enabled: bool):
        """
        Callback when the tracking toggle is changed in the ControlPanel.
        
        This method enables or disables the ObjectTracker and updates the status bar.
        
        Args:
            enabled: True to enable tracking, False to disable
        """
        if self.tracker is not None:
            self.tracker.set_enabled(enabled)
            status_text = "Detection: ON" if enabled else "Detection: OFF"
            self.status.var_status.set(f"Status: {status_text}")

    def _on_toggle_debug(self, enabled: bool):
        """Toggle debug prints across the app."""
        self.debug_enabled = enabled
        if hasattr(self.tracker, "set_debug"):
            self.tracker.set_debug(enabled)

<<<<<<< HEAD
    def _on_change_detection_classes(self, classes):
        """Update tracker target classes from the Settings menu."""
        if not hasattr(self, "tracker") or self.tracker is None:
            return
        # If empty list, treat as ALL classes
        target = classes if classes else None
        self.tracker.set_target_classes(target)
        self.status.var_status.set(
            f"Status: Classes = {', '.join(classes) if classes else 'ALL'}"
        )
=======
    def _on_set_performance(self, mode: str):
        """Apply a performance profile to the tracker at runtime."""
        if not hasattr(self, "tracker") or self.tracker is None:
            return
        mode = (mode or "").strip().lower()
        if mode == "high_fps":
            # Emphasize FPS: more skipping, smaller input, moderate conf
            self.tracker.set_process_interval(4)
            self.tracker.set_imgsz(480)
            self.tracker.set_conf(0.35)
            self.status.var_status.set("Status: Performance = High FPS")
        elif mode == "high_accuracy":
            # Emphasize accuracy: no skipping, normal input size, higher conf
            self.tracker.set_process_interval(1)
            self.tracker.set_imgsz(640)
            self.tracker.set_conf(0.50)
            self.status.var_status.set("Status: Performance = High Accuracy")
        else:
            # Balanced default
            self.tracker.set_process_interval(2)
            self.tracker.set_imgsz(640)
            self.tracker.set_conf(0.35)
            self.status.var_status.set("Status: Performance = Balanced")
>>>>>>> Performance_Optimization

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary 0.2 UI Release + SOLID principles are lit – CS4398 Group 3\n\n"
        )

    def on_exit(self):
        """Safely close the application."""
        try:
            if hasattr(self, "video_panel"):
                cam = getattr(self.video_panel, "camera", None)
                if cam and cam.is_running():
                    cam.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "arduino") and self.arduino:
                self.arduino.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()
