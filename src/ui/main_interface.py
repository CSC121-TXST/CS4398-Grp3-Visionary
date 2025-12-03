"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from ui.style import apply_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from ui.interface_layout.narration_panel import NarrationPanel
from hardware.arduino_controller import ArduinoController

# Tracking Module
from vision.tracking import ObjectTracker

# AI Module
from ai import VisionNarrator


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
            conf=0.20,  # Lower confidence (0.20) for better detection of small objects like phones/books
            target_classes=["person", "cell phone", "book"]  # Default includes common objects
        )
        # Ensure tracker starts with debug disabled
        if hasattr(self.tracker, "set_debug"):
            self.tracker.set_debug(self.debug_enabled)

        # Initialize Vision Narrator (requires OpenAI API key)
        try:
            self.narrator = VisionNarrator()
            self.narrator_available = True
            print("DEBUG: Vision Narrator initialized successfully")
        except Exception as e:
            print(f"ERROR: Vision Narrator initialization failed: {e}")
            print(f"ERROR: Please ensure OPENAI_API_KEY is set in src/.env file")
            self.narrator = None
            self.narrator_available = False

        # Build UI 
        build_menubar(
            self,
            on_exit=self.on_exit,
            on_about=self._show_about,
            on_toggle_debug=self._on_toggle_debug,
            on_change_detection_classes=self._on_change_detection_classes,
            on_set_performance=self._on_set_performance
        )
        build_title(self)

        self._build_main_area()      # video + control panel
        self._build_statusbar()      # footer
        
        # Sync UI checkboxes with initial tracker classes after UI is built
        # This ensures the menu reflects the actual tracker state (person, cell phone, book)
        self.after(100, self._sync_detection_classes_ui)

    def _build_main_area(self):
        # Main Layout
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True, padx=15, pady=15)
        
        # Grid Configuration
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Video Panel
        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}"),
            tracker=self.tracker,
            on_tracking_update=lambda count: self.status.var_tracking.set(f"Tracking: {count} objects"),
            on_narration_detection=self._on_narration_detection if self.narrator_available else None
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=0)

        # Hardware
        self.arduino = ArduinoController()

        # Sidebar Container
        sidebar = ttk.Frame(main)
        sidebar.grid(row=0, column=1, sticky="nsew")

        # Control Panel
        self.control_frame = ttk.LabelFrame(sidebar, text="Control Panel", padding=10)
        self.control_frame.pack(side=tk.TOP, fill="x", expand=False, pady=(0, 15))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            on_toggle_tracking=self._on_toggle_tracking if self.tracker is not None else None,
            on_start_narration=self._on_start_narration_period if self.narrator_available else None,
            on_end_narration=self._on_end_narration_period if self.narrator_available else None
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Narration Panel
        if self.narrator_available:
            self.narration_panel = NarrationPanel(sidebar)
            self.narration_panel.pack(side=tk.TOP, fill="both", expand=True)
    
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

    def _sync_detection_classes_ui(self):
        """Sync UI checkboxes with tracker's current target classes."""
        if not hasattr(self, "tracker") or self.tracker is None:
            return
        
        # Get current tracker classes
        tracker_classes = self.tracker.target_classes if hasattr(self.tracker, 'target_classes') and self.tracker.target_classes else set()
        
        # Update checkboxes to match tracker state
        if hasattr(self, "_var_cls_person"):
            self._var_cls_person.set("person" in tracker_classes)
        if hasattr(self, "_var_cls_cell_phone"):
            self._var_cls_cell_phone.set("cell phone" in tracker_classes)
        if hasattr(self, "_var_cls_book"):
            self._var_cls_book.set("book" in tracker_classes)
        if hasattr(self, "_var_cls_dog"):
            self._var_cls_dog.set("dog" in tracker_classes)
        if hasattr(self, "_var_cls_cat"):
            self._var_cls_cat.set("cat" in tracker_classes)
        if hasattr(self, "_var_cls_backpack"):
            self._var_cls_backpack.set("backpack" in tracker_classes)
    
    def _on_change_detection_classes(self, classes):
        """Update tracker target classes from the Settings menu."""
        if not hasattr(self, "tracker") or self.tracker is None:
            return
        # If empty list, treat as ALL classes
        target = classes if classes else None
        self.tracker.set_target_classes(target)
        
        # Update status with current classes
        if target:
            self.status.var_status.set(
                f"Status: Classes = {', '.join(classes)}"
            )
        else:
            self.status.var_status.set("Status: Classes = ALL")
        
        # Debug output
        if self.debug_enabled:
            print(f"DEBUG: Updated target classes to: {target}")
            print(f"DEBUG: Tracker target_classes set: {self.tracker.target_classes}")

    def _on_set_performance(self, mode: str):
        """Apply a performance profile to the tracker at runtime."""
        if not hasattr(self, "tracker") or self.tracker is None:
            return
        mode = (mode or "").strip().lower()
        if mode == "ultra_fast":
            # Maximum FPS: heavy skipping, very small input, lower conf
            self.tracker.set_process_interval(6)
            self.tracker.set_imgsz(416)
            self.tracker.set_conf(0.30)
            self.status.var_status.set("Status: Performance = Ultra Fast")
        elif mode == "high_fps":
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
        elif mode == "max_quality":
            # Maximum quality: process every frame, larger input, highest conf
            self.tracker.set_process_interval(1)
            self.tracker.set_imgsz(832)
            self.tracker.set_conf(0.60)
            self.status.var_status.set("Status: Performance = Maximum Quality")
        else:
            # Balanced default
            self.tracker.set_process_interval(2)
            self.tracker.set_imgsz(640)
            self.tracker.set_conf(0.35)
            self.status.var_status.set("Status: Performance = Balanced")

    def _on_narration_detection(self, tracked_objects):
        """
        Callback when detections are available during narration period.
        
        Args:
            tracked_objects: List of detected objects from ObjectTracker
        """
        if self.narrator and self.narrator.is_recording_period_active():
            self.narrator.add_detection_to_period(tracked_objects)
    
    def _on_start_narration_period(self):
        """Start a narration recording period."""
        if not self.narrator or not hasattr(self, 'narration_panel'):
            messagebox.showwarning("Narration Unavailable", "Vision Narrator is not available.")
            return
        
        self.narrator.start_recording_period()
        self.narration_panel.update_status("Recording...")
        self.narration_panel.update_narration("Started narration period. Detections are being collected...")
        self.status.var_status.set("Status: Narration period active")
    
    def _on_end_narration_period(self):
        """End the narration period and generate summary."""
        if not self.narrator or not hasattr(self, 'narration_panel'):
            return
        
        if not self.narrator.is_recording_period_active():
            messagebox.showinfo("No Active Period", "No narration period is currently active.")
            return
        
        # Update UI to show processing
        self.narration_panel.update_status("Processing...")
        if hasattr(self, '_control_widget'):
            self._control_widget.update_narration_status("Processing...", "#FF9800")
        self.status.var_status.set("Status: Generating narration...")
        
        # Generate summary (this may take a moment)
        # Run in a separate thread to avoid blocking UI, then update UI in main thread
        def run_in_thread():
            try:
                description = self.narrator.stop_recording_period()
                # Schedule UI update in main thread
                self.after(0, lambda: update_ui_with_description(description))
            except Exception as e:
                # Schedule error update in main thread
                self.after(0, lambda: update_ui_with_error(str(e)))
        
        def update_ui_with_description(description):
            if description:
                self.narration_panel.update_narration(description)
                self.narration_panel.update_status("Complete")
                if hasattr(self, '_control_widget'):
                    self._control_widget.update_narration_status("Complete", "#4CAF50")
                self.status.var_status.set("Status: Narration complete")
            else:
                self.narration_panel.update_narration("No objects were detected during this period.")
                self.narration_panel.update_status("No detections")
                if hasattr(self, '_control_widget'):
                    self._control_widget.update_narration_status("No detections", "#9aa4b1")
                self.status.var_status.set("Status: No detections in period")
            
            # Reset button states
            if hasattr(self, '_control_widget'):
                self._control_widget.reset_narration_buttons()
        
        def update_ui_with_error(error_msg):
            self.narration_panel.update_narration(f"Error generating narration: {error_msg}")
            self.narration_panel.update_status("Error")
            if hasattr(self, '_control_widget'):
                self._control_widget.update_narration_status("Error", "#F44336")
            self.status.var_status.set("Status: Narration error")
            messagebox.showerror("Narration Error", f"Error generating narration: {error_msg}")
            
            # Reset button states
            if hasattr(self, '_control_widget'):
                self._control_widget.reset_narration_buttons()
        
        # Start generation in background thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

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
