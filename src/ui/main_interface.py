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
from ai.vision_narrator import VisionNarrator
from vision.object_detection import get_current_detections


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build UI 
        build_menubar(self, on_exit=self.on_exit, on_about=self._show_about)
        build_title(self)

        # Initialize Vision Narrator
        self.narrator = VisionNarrator()
        
        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Start periodic narration timer (every 60 seconds)
        self._start_narration_timer()

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}")
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew",
                              padx=(10, 5), pady=(0, 10))

        # Hardware
        self.arduino = ArduinoController()

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew",
                                padx=(5, 10), pady=(0, 10))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            narrator=self.narrator,
            on_narrate_request=self._trigger_narration
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Set up narrator callback to update UI (the narration panel should have already set this,
        # but we ensure it's set correctly here)
        if hasattr(self._control_widget, 'narration_panel'):
            # The narration panel should have already set up the callback, but we can verify
            pass

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        self.status = StatusBar(self)
        self.status.pack(side=tk.BOTTOM, fill="x")
    
    def _start_narration_timer(self, interval_ms=60000):
        """
        Start the periodic narration timer.
        
        Args:
            interval_ms: Interval in milliseconds (default: 60000 = 1 minute)
        """
        self._trigger_narration()
        # Schedule next narration
        self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _schedule_next_narration(self, interval_ms):
        """Schedule the next periodic narration."""
        if self.winfo_exists():  # Check if window still exists
            self._trigger_narration()
            self.after(interval_ms, lambda: self._schedule_next_narration(interval_ms))
    
    def _trigger_narration(self):
        """Trigger narration generation from current detections."""
        try:
            # Get current detections (placeholder for now)
            detections = get_current_detections()
            
            # Generate narration
            narration = self.narrator.narrate(detections)
            
            # Update UI if narration panel exists
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(narration)
        except Exception as e:
            print(f"Error generating narration: {e}")
            # Update UI with error message
            if hasattr(self, '_control_widget') and hasattr(self._control_widget, 'narration_panel'):
                self._control_widget.narration_panel.update_narration(
                    f"Error generating narration: {str(e)}"
                )

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
        try:
            # Cancel any pending narration timers
            if hasattr(self, "_narration_timer_id"):
                self.after_cancel(self._narration_timer_id)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()
