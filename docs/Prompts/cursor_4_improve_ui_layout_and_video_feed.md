# Improve UI layout and video feed
_Exported on 12/3/2025 at 23:24:49 CST from Cursor (2.1.46)_

---

**User**

Alright I've just merged all of my code into the most recent UI improvements branch. I would like to focus on improving the UI entirely at this point, minimizing the size of many of the buttons we use, making sure all of the buttons stay on screen. 



Right now when you connect the hardware it forces the additional options off screen below.



Additionally, we can get rid of the grid overlay on the Video Feed, and potentially increase the size of the video feed if we're capturing screen revenue back from the control panel section.



Begin making those improvements with the following goals:

1) All Buttons should always remain on screen, if for any reason a UI change pushes them further below, a scroll wheel should appear to accomodate.



2) The status bar should remain at the bottom of the screen providing information about the system



3) The Video feed should be clean and clear.





---

**Cursor**


Reviewing the current UI structure to understand the layout.



Examining the control panel structure and related files:



Creating the initial todo list and starting implementation:



Implementing the improvements:


```python
# Central CSS adjacent theme + HUD utilities (this is pure Tkinter/ttk)

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

# Palette Constants
SEC_BG   = "#464646"
SEC_SURF = "#2b2b2b"
SEC_LINE = "#3a3a3a"
ACCENT   = "#00c2c7"
ACCENT_2 = "#15ffa8"
TEXT     = "#e6edf3"
MUTED    = "#9aa4b1"

# Theme State
_is_dark_mode = True

def font(primary=True, size=10, weight="normal"):
    # Font Selection
    family = "Segoe UI" if primary else "Calibri" 
    return (family, size, weight)

def apply_theme(root: tk.Tk):
    # Initial Theme Setup
    style = ThemedStyle(root)
    try:
        style.set_theme("equilux")
    except Exception:
        style.theme_use("clam")
    
    _configure_custom_styles(style, root)

def toggle_theme(root: tk.Tk):
    # Theme Toggle Logic
    global _is_dark_mode
    style = ThemedStyle(root)
    
    if _is_dark_mode:
        style.set_theme("arc") 
        _is_dark_mode = False
    else:
        style.set_theme("equilux") 
        _is_dark_mode = True
        
    _configure_custom_styles(style, root)
    return _is_dark_mode

def _configure_custom_styles(style, root):
    # Global Defaults
    style.configure(".", font=font(True, 10))
    
    # Button Style
    style.configure("TButton", padding=6, font=font(True, 10, "bold"))

    # Labelframe Style
    style.configure("TLabelframe", font=font(True, 10, "bold"))
    style.configure("TLabelframe.Label", font=font(True, 10, "bold"))

    # Header Style
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    
    # Status Bar Style
    style.configure("Status.TLabel", font=("Consolas", 9), background=style.lookup("TFrame", "background"))

    # Window Background
    bg_color = style.lookup("TFrame", "background")
    root.configure(bg=bg_color)

def style_menu(menu: tk.Menu):
    # Menu Styling
    menu.configure(
        tearoff=0,
        bg="#464646", fg="#e6edf3",
        activebackground="#5e5e5e", activeforeground="#ffffff",
        bd=0, relief="flat"
    )

def add_video_grid(parent: ttk.LabelFrame):
    # Clean Canvas Setup (grid overlay removed for cleaner video feed)
    canvas = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    return canvas
```

```python
# Central CSS adjacent theme + HUD utilities (this is pure Tkinter/ttk)

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

# Palette Constants
SEC_BG   = "#464646"
SEC_SURF = "#2b2b2b"
SEC_LINE = "#3a3a3a"
ACCENT   = "#00c2c7"
ACCENT_2 = "#15ffa8"
TEXT     = "#e6edf3"
MUTED    = "#9aa4b1"

# Theme State
_is_dark_mode = True

def font(primary=True, size=10, weight="normal"):
    # Font Selection
    family = "Segoe UI" if primary else "Calibri" 
    return (family, size, weight)

def apply_theme(root: tk.Tk):
    # Initial Theme Setup
    style = ThemedStyle(root)
    try:
        style.set_theme("equilux")
    except Exception:
        style.theme_use("clam")
    
    _configure_custom_styles(style, root)

def toggle_theme(root: tk.Tk):
    # Theme Toggle Logic
    global _is_dark_mode
    style = ThemedStyle(root)
    
    if _is_dark_mode:
        style.set_theme("arc") 
        _is_dark_mode = False
    else:
        style.set_theme("equilux") 
        _is_dark_mode = True
        
    _configure_custom_styles(style, root)
    return _is_dark_mode

def _configure_custom_styles(style, root):
    # Global Defaults
    style.configure(".", font=font(True, 10))
    
    # Button Style (reduced padding for smaller buttons)
    style.configure("TButton", padding=4, font=font(True, 9, "normal"))

    # Labelframe Style
    style.configure("TLabelframe", font=font(True, 10, "bold"))
    style.configure("TLabelframe.Label", font=font(True, 10, "bold"))

    # Header Style
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    
    # Status Bar Style
    style.configure("Status.TLabel", font=("Consolas", 9), background=style.lookup("TFrame", "background"))

    # Window Background
    bg_color = style.lookup("TFrame", "background")
    root.configure(bg=bg_color)

def style_menu(menu: tk.Menu):
    # Menu Styling
    menu.configure(
        tearoff=0,
        bg="#464646", fg="#e6edf3",
        activebackground="#5e5e5e", activeforeground="#ffffff",
        bd=0, relief="flat"
    )

def add_video_grid(parent: ttk.LabelFrame):
    # Clean Canvas Setup (grid overlay removed for cleaner video feed)
    canvas = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    return canvas
```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from ui.style import apply_theme, toggle_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from ui.interface_layout.narration_panel import NarrationPanel
from hardware.arduino_controller import ArduinoController

# Tracking Module
from vision.tracking import ObjectTracker

# Servo Tracking Module
from hardware.servo_tracking import ServoTracker
from hardware.target_selector import TargetSelector

# AI Module
from ai import VisionNarrator

class ToggleSwitch(tk.Canvas):
    # Custom Toggle Widget
    def __init__(self, parent, command=None, *args, **kwargs):
        super().__init__(parent, width=50, height=24, highlightthickness=0, bd=0, *args, **kwargs)
        self.command = command
        self.is_on = True
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _toggle(self, event=None):
        # Click Handler
        self.is_on = not self.is_on
        self._draw()
        if self.command:
            self.command()

    def _draw(self):
        # Drawing Logic
        self.delete("all")
        bg = "#00c2c7" if self.is_on else "#777777"
        fg = "#ffffff"
        
        # Pill Shape
        self.create_oval(2, 2, 22, 22, fill=bg, outline=bg)
        self.create_oval(28, 2, 48, 22, fill=bg, outline=bg)
        self.create_rectangle(12, 2, 38, 22, fill=bg, outline=bg)
        
        # Circle Indicator
        cx = 38 if self.is_on else 12
        self.create_oval(cx-8, 4, cx+8, 20, fill=fg, outline=fg)

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
        # Main Layout Container
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True, padx=15, pady=15)
        
        # Grid Configuration (increased video feed size: 5:2 ratio)
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Video Panel
        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}"),
            tracker=self.tracker,
            on_tracking_update=lambda count: self.status.var_tracking.set(f"Tracking: {count} objects"),
            on_narration_detection=self._on_narration_detection if self.narrator_available else None,
            on_servo_tracking=self._on_servo_tracking  # Callback for servo tracking
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=0)

        # Hardware Controller
        self.arduino = ArduinoController()
        
        # Initialize ServoTracker for automatic target tracking
        # Default image size will be updated from actual camera frames
        self.servo_tracker = ServoTracker(
            image_width=640,
            image_height=480,
            pan_range=(30, 150),
            tilt_range=(40, 140),
            offset_vertical=0.15  # Aim 15% below center mass for safety
        )
        
        # Initialize TargetSelector for intelligent target selection
        self.target_selector = TargetSelector(
            persistence_frames=10,  # Keep tracking target for 10 frames after it's lost
            priority_threshold=20.0,  # Need 20+ priority difference to switch targets
            center_weight=0.3,  # 30% weight for center proximity
            size_weight=0.4,  # 40% weight for size
            confidence_weight=0.2,  # 20% weight for confidence
            class_weight=0.1,  # 10% weight for class priority
        )
        
        self._auto_tracking_active = True  # Enabled by default when auto-tracking mode is on

        # Sidebar Container
        sidebar = ttk.Frame(main)
        sidebar.grid(row=0, column=1, sticky="nsew")

        # Theme Header
        theme_frame = ttk.Frame(sidebar)
        theme_frame.pack(side=tk.TOP, fill="x", pady=(0, 10))
        
        lbl_mode = ttk.Label(theme_frame, text="Dark Mode", font=("Segoe UI", 10, "bold"))
        lbl_mode.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Theme Toggle Logic
        def on_toggle():
            is_dark = toggle_theme(self)
            lbl_mode.config(text="Dark Mode" if is_dark else "Light Mode")
            
            new_bg = self.cget("bg")
            self.theme_switch.config(bg=new_bg)
            
        self.theme_switch = ToggleSwitch(theme_frame, command=on_toggle, bg=self.cget('bg'))
        self.theme_switch.pack(side=tk.RIGHT)

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
            on_end_narration=self._on_end_narration_period if self.narrator_available else None,
            on_auto_tracking_toggle=self._on_auto_tracking_toggle
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

    def _on_auto_tracking_toggle(self, enabled: bool):
        """
        Callback when auto-tracking toggle is changed in HardwareControls.
        
        This enables/disables automatic servo tracking of detected objects.
        
        Args:
            enabled: True to enable servo auto-tracking, False to disable
        """
        self._auto_tracking_active = enabled
        if enabled:
            # Reset target selector when re-enabling to start fresh
            if hasattr(self, 'target_selector'):
                self.target_selector.reset()
        if self.debug_enabled:
            print(f"Servo auto-tracking: {'enabled' if enabled else 'disabled'}")

    def _on_servo_tracking(self, tracked_objects, frame_shape):
        """
        Callback when objects are detected and need to be tracked with servos.
        
        This method uses TargetSelector to intelligently choose the best target
        and converts its coordinates to servo angles for automatic tracking.
        
        Args:
            tracked_objects: List of detected objects with bbox coordinates
            frame_shape: (width, height) tuple of frame dimensions
        """
        # Only track if Arduino is connected and auto-tracking is enabled
        if not self.arduino or not self.arduino.is_connected():
            return
        
        if not self._auto_tracking_active:
            return
        
        try:
            # Update servo tracker with actual frame dimensions
            width, height = frame_shape
            self.servo_tracker.update_image_size(width, height)
            
            # Use TargetSelector to choose the best target
            # This handles priority scoring, persistence, and prevents bouncing
            target = self.target_selector.select_target(
                tracked_objects if tracked_objects else [],
                width,
                height
            )
            
            if target:
                # Convert bounding box to servo angles
                pan_angle, tilt_angle = self.servo_tracker.bbox_to_angles(
                    target["bbox"],
                    image_width=width,
                    image_height=height
                )
                
                # Send auto-tracking command to Arduino
                self.arduino.servo_auto_track(pan_angle, tilt_angle)
                
                # Debug output
                if self.debug_enabled:
                    target_id = target.get("id", "?")
                    target_cls = target.get("cls", "?")
                    print(f"Tracking target ID:{target_id} ({target_cls}) -> Pan:{pan_angle}°, Tilt:{tilt_angle}°")
            else:
                # No target selected (target lost but within persistence window)
                # Servos will hold last position
                if self.debug_enabled:
                    print("No target selected (holding last position)")
                
        except Exception as e:
            # Silently fail to avoid spamming errors during tracking
            if self.debug_enabled:
                print(f"Servo tracking error: {e}")

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

```


Adding a scrollable frame to the control panel. Creating a utility class and updating the main interface:



Creating a scrollable frame utility and updating the sidebar to use it:


```python
"""
Scrollable Frame Utility

Provides a scrollable container for widgets that may overflow their available space.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame with a scrollbar that appears when content exceeds available space.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_canvas_configure(self, event):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        # Only scroll if mouse is over the canvas
        if self.canvas.winfo_containing(event.x_root, event.y_root) == self.canvas:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def get_content_frame(self):
        """Return the frame where widgets should be placed."""
        return self.scrollable_frame

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from ui.style import apply_theme, toggle_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from ui.interface_layout.narration_panel import NarrationPanel
from ui.interface_layout.scrollable_frame import ScrollableFrame
from hardware.arduino_controller import ArduinoController

# Tracking Module
from vision.tracking import ObjectTracker

# Servo Tracking Module
from hardware.servo_tracking import ServoTracker
from hardware.target_selector import TargetSelector

# AI Module
from ai import VisionNarrator

class ToggleSwitch(tk.Canvas):
    # Custom Toggle Widget
    def __init__(self, parent, command=None, *args, **kwargs):
        super().__init__(parent, width=50, height=24, highlightthickness=0, bd=0, *args, **kwargs)
        self.command = command
        self.is_on = True
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _toggle(self, event=None):
        # Click Handler
        self.is_on = not self.is_on
        self._draw()
        if self.command:
            self.command()

    def _draw(self):
        # Drawing Logic
        self.delete("all")
        bg = "#00c2c7" if self.is_on else "#777777"
        fg = "#ffffff"
        
        # Pill Shape
        self.create_oval(2, 2, 22, 22, fill=bg, outline=bg)
        self.create_oval(28, 2, 48, 22, fill=bg, outline=bg)
        self.create_rectangle(12, 2, 38, 22, fill=bg, outline=bg)
        
        # Circle Indicator
        cx = 38 if self.is_on else 12
        self.create_oval(cx-8, 4, cx+8, 20, fill=fg, outline=fg)

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
        # Main Layout Container
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True, padx=15, pady=15)
        
        # Grid Configuration (increased video feed size: 5:2 ratio)
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Video Panel
        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}"),
            tracker=self.tracker,
            on_tracking_update=lambda count: self.status.var_tracking.set(f"Tracking: {count} objects"),
            on_narration_detection=self._on_narration_detection if self.narrator_available else None,
            on_servo_tracking=self._on_servo_tracking  # Callback for servo tracking
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=0)

        # Hardware Controller
        self.arduino = ArduinoController()
        
        # Initialize ServoTracker for automatic target tracking
        # Default image size will be updated from actual camera frames
        self.servo_tracker = ServoTracker(
            image_width=640,
            image_height=480,
            pan_range=(30, 150),
            tilt_range=(40, 140),
            offset_vertical=0.15  # Aim 15% below center mass for safety
        )
        
        # Initialize TargetSelector for intelligent target selection
        self.target_selector = TargetSelector(
            persistence_frames=10,  # Keep tracking target for 10 frames after it's lost
            priority_threshold=20.0,  # Need 20+ priority difference to switch targets
            center_weight=0.3,  # 30% weight for center proximity
            size_weight=0.4,  # 40% weight for size
            confidence_weight=0.2,  # 20% weight for confidence
            class_weight=0.1,  # 10% weight for class priority
        )
        
        self._auto_tracking_active = True  # Enabled by default when auto-tracking mode is on

        # Sidebar Container
        sidebar = ttk.Frame(main)
        sidebar.grid(row=0, column=1, sticky="nsew")

        # Theme Header
        theme_frame = ttk.Frame(sidebar)
        theme_frame.pack(side=tk.TOP, fill="x", pady=(0, 10))
        
        lbl_mode = ttk.Label(theme_frame, text="Dark Mode", font=("Segoe UI", 10, "bold"))
        lbl_mode.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Theme Toggle Logic
        def on_toggle():
            is_dark = toggle_theme(self)
            lbl_mode.config(text="Dark Mode" if is_dark else "Light Mode")
            
            new_bg = self.cget("bg")
            self.theme_switch.config(bg=new_bg)
            
        self.theme_switch = ToggleSwitch(theme_frame, command=on_toggle, bg=self.cget('bg'))
        self.theme_switch.pack(side=tk.RIGHT)

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
            on_end_narration=self._on_end_narration_period if self.narrator_available else None,
            on_auto_tracking_toggle=self._on_auto_tracking_toggle
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

    def _on_auto_tracking_toggle(self, enabled: bool):
        """
        Callback when auto-tracking toggle is changed in HardwareControls.
        
        This enables/disables automatic servo tracking of detected objects.
        
        Args:
            enabled: True to enable servo auto-tracking, False to disable
        """
        self._auto_tracking_active = enabled
        if enabled:
            # Reset target selector when re-enabling to start fresh
            if hasattr(self, 'target_selector'):
                self.target_selector.reset()
        if self.debug_enabled:
            print(f"Servo auto-tracking: {'enabled' if enabled else 'disabled'}")

    def _on_servo_tracking(self, tracked_objects, frame_shape):
        """
        Callback when objects are detected and need to be tracked with servos.
        
        This method uses TargetSelector to intelligently choose the best target
        and converts its coordinates to servo angles for automatic tracking.
        
        Args:
            tracked_objects: List of detected objects with bbox coordinates
            frame_shape: (width, height) tuple of frame dimensions
        """
        # Only track if Arduino is connected and auto-tracking is enabled
        if not self.arduino or not self.arduino.is_connected():
            return
        
        if not self._auto_tracking_active:
            return
        
        try:
            # Update servo tracker with actual frame dimensions
            width, height = frame_shape
            self.servo_tracker.update_image_size(width, height)
            
            # Use TargetSelector to choose the best target
            # This handles priority scoring, persistence, and prevents bouncing
            target = self.target_selector.select_target(
                tracked_objects if tracked_objects else [],
                width,
                height
            )
            
            if target:
                # Convert bounding box to servo angles
                pan_angle, tilt_angle = self.servo_tracker.bbox_to_angles(
                    target["bbox"],
                    image_width=width,
                    image_height=height
                )
                
                # Send auto-tracking command to Arduino
                self.arduino.servo_auto_track(pan_angle, tilt_angle)
                
                # Debug output
                if self.debug_enabled:
                    target_id = target.get("id", "?")
                    target_cls = target.get("cls", "?")
                    print(f"Tracking target ID:{target_id} ({target_cls}) -> Pan:{pan_angle}°, Tilt:{tilt_angle}°")
            else:
                # No target selected (target lost but within persistence window)
                # Servos will hold last position
                if self.debug_enabled:
                    print("No target selected (holding last position)")
                
        except Exception as e:
            # Silently fail to avoid spamming errors during tracking
            if self.debug_enabled:
                print(f"Servo tracking error: {e}")

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

```

```python
"""
Main Interface module/window that assembles the pieces.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from ui.style import apply_theme, toggle_theme
from ui.interface_layout.menubar import build_menubar
from ui.interface_layout.title_bar import build_title
from ui.interface_layout.status_bar import StatusBar
from ui.interface_layout.video_panel import VideoPanel

from ui.interface_layout.control_panel import ControlPanel
from ui.interface_layout.narration_panel import NarrationPanel
from ui.interface_layout.scrollable_frame import ScrollableFrame
from hardware.arduino_controller import ArduinoController

# Tracking Module
from vision.tracking import ObjectTracker

# Servo Tracking Module
from hardware.servo_tracking import ServoTracker
from hardware.target_selector import TargetSelector

# AI Module
from ai import VisionNarrator

class ToggleSwitch(tk.Canvas):
    # Custom Toggle Widget
    def __init__(self, parent, command=None, *args, **kwargs):
        super().__init__(parent, width=50, height=24, highlightthickness=0, bd=0, *args, **kwargs)
        self.command = command
        self.is_on = True
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _toggle(self, event=None):
        # Click Handler
        self.is_on = not self.is_on
        self._draw()
        if self.command:
            self.command()

    def _draw(self):
        # Drawing Logic
        self.delete("all")
        bg = "#00c2c7" if self.is_on else "#777777"
        fg = "#ffffff"
        
        # Pill Shape
        self.create_oval(2, 2, 22, 22, fill=bg, outline=bg)
        self.create_oval(28, 2, 48, 22, fill=bg, outline=bg)
        self.create_rectangle(12, 2, 38, 22, fill=bg, outline=bg)
        
        # Circle Indicator
        cx = 38 if self.is_on else 12
        self.create_oval(cx-8, 4, cx+8, 20, fill=fg, outline=fg)

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
        # Main Layout Container
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True, padx=15, pady=15)
        
        # Grid Configuration (increased video feed size: 5:2 ratio)
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Video Panel
        self.video_panel = VideoPanel(
            parent=main,
            on_fps=lambda f: self.status.var_fps.set(f"FPS: {f:4.1f}"),
            tracker=self.tracker,
            on_tracking_update=lambda count: self.status.var_tracking.set(f"Tracking: {count} objects"),
            on_narration_detection=self._on_narration_detection if self.narrator_available else None,
            on_servo_tracking=self._on_servo_tracking  # Callback for servo tracking
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=0)

        # Hardware Controller
        self.arduino = ArduinoController()
        
        # Initialize ServoTracker for automatic target tracking
        # Default image size will be updated from actual camera frames
        self.servo_tracker = ServoTracker(
            image_width=640,
            image_height=480,
            pan_range=(30, 150),
            tilt_range=(40, 140),
            offset_vertical=0.15  # Aim 15% below center mass for safety
        )
        
        # Initialize TargetSelector for intelligent target selection
        self.target_selector = TargetSelector(
            persistence_frames=10,  # Keep tracking target for 10 frames after it's lost
            priority_threshold=20.0,  # Need 20+ priority difference to switch targets
            center_weight=0.3,  # 30% weight for center proximity
            size_weight=0.4,  # 40% weight for size
            confidence_weight=0.2,  # 20% weight for confidence
            class_weight=0.1,  # 10% weight for class priority
        )
        
        self._auto_tracking_active = True  # Enabled by default when auto-tracking mode is on

        # Sidebar Container
        sidebar = ttk.Frame(main)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_rowconfigure(1, weight=1)  # Make scrollable area expandable

        # Theme Header (fixed at top)
        theme_frame = ttk.Frame(sidebar)
        theme_frame.pack(side=tk.TOP, fill="x", pady=(0, 10))
        
        lbl_mode = ttk.Label(theme_frame, text="Dark Mode", font=("Segoe UI", 10, "bold"))
        lbl_mode.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Theme Toggle Logic
        def on_toggle():
            is_dark = toggle_theme(self)
            lbl_mode.config(text="Dark Mode" if is_dark else "Light Mode")
            
            new_bg = self.cget("bg")
            self.theme_switch.config(bg=new_bg)
            
        self.theme_switch = ToggleSwitch(theme_frame, command=on_toggle, bg=self.cget('bg'))
        self.theme_switch.pack(side=tk.RIGHT)

        # Scrollable container for control panel and narration
        scrollable_container = ScrollableFrame(sidebar)
        scrollable_container.pack(side=tk.TOP, fill="both", expand=True)
        content_frame = scrollable_container.get_content_frame()

        # Control Panel (inside scrollable frame)
        self.control_frame = ttk.LabelFrame(content_frame, text="Control Panel", padding=10)
        self.control_frame.pack(side=tk.TOP, fill="x", expand=False, pady=(0, 15))

        self._control_widget = ControlPanel(
            self.control_frame,
            camera=self.video_panel.camera,
            arduino=self.arduino,
            on_status=lambda s: self.status.var_status.set(f"Status: {s}"),
            on_laser=lambda on: self.status.var_laser.set(f"Laser: {'ON' if on else 'OFF'}"),
            on_toggle_tracking=self._on_toggle_tracking if self.tracker is not None else None,
            on_start_narration=self._on_start_narration_period if self.narrator_available else None,
            on_end_narration=self._on_end_narration_period if self.narrator_available else None,
            on_auto_tracking_toggle=self._on_auto_tracking_toggle
        )
        self._control_widget.pack(expand=True, fill="both")
        
        # Narration Panel (inside scrollable frame)
        if self.narrator_available:
            self.narration_panel = NarrationPanel(content_frame)
            self.narration_panel.pack(side=tk.TOP, fill="both", expand=True, pady=(0, 10))
    
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

    def _on_auto_tracking_toggle(self, enabled: bool):
        """
        Callback when auto-tracking toggle is changed in HardwareControls.
        
        This enables/disables automatic servo tracking of detected objects.
        
        Args:
            enabled: True to enable servo auto-tracking, False to disable
        """
        self._auto_tracking_active = enabled
        if enabled:
            # Reset target selector when re-enabling to start fresh
            if hasattr(self, 'target_selector'):
                self.target_selector.reset()
        if self.debug_enabled:
            print(f"Servo auto-tracking: {'enabled' if enabled else 'disabled'}")

    def _on_servo_tracking(self, tracked_objects, frame_shape):
        """
        Callback when objects are detected and need to be tracked with servos.
        
        This method uses TargetSelector to intelligently choose the best target
        and converts its coordinates to servo angles for automatic tracking.
        
        Args:
            tracked_objects: List of detected objects with bbox coordinates
            frame_shape: (width, height) tuple of frame dimensions
        """
        # Only track if Arduino is connected and auto-tracking is enabled
        if not self.arduino or not self.arduino.is_connected():
            return
        
        if not self._auto_tracking_active:
            return
        
        try:
            # Update servo tracker with actual frame dimensions
            width, height = frame_shape
            self.servo_tracker.update_image_size(width, height)
            
            # Use TargetSelector to choose the best target
            # This handles priority scoring, persistence, and prevents bouncing
            target = self.target_selector.select_target(
                tracked_objects if tracked_objects else [],
                width,
                height
            )
            
            if target:
                # Convert bounding box to servo angles
                pan_angle, tilt_angle = self.servo_tracker.bbox_to_angles(
                    target["bbox"],
                    image_width=width,
                    image_height=height
                )
                
                # Send auto-tracking command to Arduino
                self.arduino.servo_auto_track(pan_angle, tilt_angle)
                
                # Debug output
                if self.debug_enabled:
                    target_id = target.get("id", "?")
                    target_cls = target.get("cls", "?")
                    print(f"Tracking target ID:{target_id} ({target_cls}) -> Pan:{pan_angle}°, Tilt:{tilt_angle}°")
            else:
                # No target selected (target lost but within persistence window)
                # Servos will hold last position
                if self.debug_enabled:
                    print("No target selected (holding last position)")
                
        except Exception as e:
            # Silently fail to avoid spamming errors during tracking
            if self.debug_enabled:
                print(f"Servo tracking error: {e}")

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

```


Verifying the status bar and reducing button sizes in hardware controls:


```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=4)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 8))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(2, 4))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(2, 8))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 8))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(2, 4))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(2, 8))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=4)
```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(2, 4))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(2, 8))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(2, 4))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(2, 8))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(2, 8))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=8)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(4, 8))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(4, 2))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(2, 1))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(2, 4))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(2, 1))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(1, 2))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(4, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(2, 1))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(1, 2))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(2, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=8)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(2, 1))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(1, 2))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(2, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=4)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=2)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```

```python
# ui/controls/camera_controls.py
import tkinter as tk
from tkinter import ttk, messagebox

class CameraControls(ttk.Frame):
    """Start/Stop camera buttons. Probably want to change this to build/rebuild depending on state."""

    def __init__(self, parent, camera, on_status, status_text: tk.StringVar):
        super().__init__(parent)
        self.camera = camera
        self.on_status = on_status
        self.status_text = status_text

        ttk.Button(
            self, text="Start Camera",
            style="Accent.TButton", command=self._start_camera
        ).pack(fill="x", pady=2)

        ttk.Button(
            self, text="Stop Camera",
            style="Accent.TButton", command=self._stop_camera
        ).pack(fill="x", pady=2)

    def _start_camera(self):
        if not self.camera:
            messagebox.showerror("Visionary", "Camera not available.")
            return
        try:
            self.camera.start()
            self.status_text.set("Camera started.")
            self.on_status("Camera: ON")
        except Exception as e:
            messagebox.showerror("Visionary", f"Failed to start camera:\n{e}")

    def _stop_camera(self):
        if not self.camera:
            return
        self.camera.stop()
        self.status_text.set("Camera stopped.")
        self.on_status("Camera: OFF")

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 4))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 4))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 4))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 6))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 12, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 6))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 6))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 6))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 6))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 4))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 10))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```

```python
"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(
        self, 
        master=None, 
        camera=None, 
        arduino=None, 
        on_status=None, 
        on_laser=None,
        on_toggle_tracking: Optional[Callable[[bool], None]] = None,
        on_start_narration: Optional[Callable[[], None]] = None,
        on_end_narration: Optional[Callable[[], None]] = None,
        on_auto_tracking_toggle: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize ControlPanel.
        
        Args:
            camera: object with start(), stop(), is_running()
            arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
            on_status: callback(str) to update status bar
            on_laser: callback(bool) to update laser label
            on_toggle_tracking: callback(bool) called when tracking toggle is changed
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.on_toggle_tracking = on_toggle_tracking
        self.on_start_narration = on_start_narration
        self.on_end_narration = on_end_narration
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 8))

        # Camera controls
        CameraControls(
            parent=btns,
            camera=self.camera,
            on_status=self.on_status,
            status_text=self._status_var()
        ).pack(fill="x", pady=(0, 2))

        # Hardware controls
        HardwareControls(
            parent=btns,
            arduino=self.arduino,
            on_status=self.on_status,
            on_laser=self.on_laser,
            status_text=self._status_var(),
            on_auto_tracking_toggle=self.on_auto_tracking_toggle
        ).pack(fill="x", pady=(0, 2))

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Status labels
        ttk.Label(
            self, text="System Status",
            font=("Consolas", 11, "bold"), foreground=ACCENT_2
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self._status_var(), foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Object Detection Toggle
        if self.on_toggle_tracking is not None:
            ttk.Label(
                self, text="Object Detection",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # BooleanVar to track checkbox state
            self.tracking_var = tk.BooleanVar(value=False)  # Disabled by default
            
            # Checkbutton for enabling/disabling object tracking
            tracking_check = ttk.Checkbutton(
                self,
                text="Enable Detection",
                variable=self.tracking_var,
                command=self._on_tracking_toggle
            )
            tracking_check.pack(anchor="w", pady=(0, 6))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)

        # Narration Period Controls
        if self.on_start_narration is not None or self.on_end_narration is not None:
            ttk.Label(
                self, text="Vision Narration",
                font=("Consolas", 11, "bold"), foreground=ACCENT_2
            ).pack(anchor="w", pady=(0, 4))
            
            # Narration period buttons frame
            narration_frame = ttk.Frame(self)
            narration_frame.pack(fill="x", pady=(0, 4))
            
            if self.on_start_narration is not None:
                self.start_btn = ttk.Button(
                    narration_frame,
                    text="Start Narration Period",
                    command=self._on_start_narration,
                    style="Accent.TButton"
                )
                self.start_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
            
            if self.on_end_narration is not None:
                self.end_btn = ttk.Button(
                    narration_frame,
                    text="End Narration Period",
                    command=self._on_end_narration,
                    state=tk.DISABLED  # Disabled until period starts
                )
                self.end_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
            
            # Narration status label
            self.narration_status = ttk.Label(
                self,
                text="Not recording",
                foreground="#9aa4b1",
                font=("Arial", 9)
            )
            self.narration_status.pack(anchor="w", pady=(0, 6))
            
            ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)
    
    def _on_start_narration(self):
        """Callback when Start Narration Period button is clicked."""
        if self.on_start_narration:
            self.on_start_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.DISABLED)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.NORMAL)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Recording...", foreground="#4CAF50")
    
    def _on_end_narration(self):
        """Callback when End Narration Period button is clicked."""
        if self.on_end_narration:
            self.on_end_narration()
            # Update button states
            if hasattr(self, 'start_btn'):
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'end_btn'):
                self.end_btn.config(state=tk.DISABLED)
            if hasattr(self, 'narration_status'):
                self.narration_status.config(text="Processing...", foreground="#FF9800")
    
    def update_narration_status(self, status_text: str, color: str = "#9aa4b1"):
        """Update the narration status label."""
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text=status_text, foreground=color)
    
    def reset_narration_buttons(self):
        """Reset narration buttons to initial state."""
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'end_btn'):
            self.end_btn.config(state=tk.DISABLED)
        if hasattr(self, 'narration_status'):
            self.narration_status.config(text="Not recording", foreground="#9aa4b1")
    
    def _on_tracking_toggle(self):
        """
        Callback when the detection checkbox is toggled.
        Calls the on_toggle_tracking callback with the new state.
        """
        if self.on_toggle_tracking is not None:
            enabled = self.tracking_var.get()
            self.on_toggle_tracking(enabled)

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text

```


Fixing the separator issue:


```python
# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from hardware.config_manager import HardwareConfig

class HardwareControls(ttk.Frame):
    """
    Small widget for Arduino actions:
    - Connect
    - LED ON / LED OFF / Blink (x3)
    - Disconnect

    Expects:
      arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
      on_status: callback(str) to update status bar
      on_laser:  callback(bool) to update 'Laser: ON/OFF' footer label
      status_text: StringVar to show local panel status
      on_auto_tracking_toggle: Optional callback(bool) when auto-tracking toggle changes
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar, on_auto_tracking_toggle=None):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text
        self.on_auto_tracking_toggle = on_auto_tracking_toggle

        self._connected_controls = None
        self._servo_running = False
        self._angle_after_id = None  # for slider debounce
        self._laser_on = False
        self._auto_tracking_enabled = True
        self._pan_after_id = None
        self._tilt_after_id = None
        
        # Initialize config manager for saving/loading offsets
        self.config = HardwareConfig()
        
        # Load saved offsets from config
        self._pan_offset, self._tilt_offset = self.config.get_offsets()
        self._offset_after_id = None  # for offset change debounce

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=2)

        # Holder frame for controls that only make sense when connected
        self._connected_controls = None

    # ---- actions ----
    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
            return

        if self.arduino.is_connected():
            self.status_text.set("Arduino already connected.")
            self.on_status("Hardware: connected")
            self._show_connected_controls()
            return

        self.status_text.set("Connecting to Arduino…")
        self.on_status("Hardware: connecting…")
        self.connect_btn.configure(state="disabled")
        self.after(50, self._do_connect)

    def _do_connect(self):
        try:
            ok = self.arduino.connect()
        except Exception as e:
            ok = False
            messagebox.showerror("Visionary", f"Arduino connect failed:\n{e}")

        if ok:
            self.status_text.set("Arduino connected.")
            self.on_status("Hardware: connected")
            # Proof-of-life: LED on ~0.8s then off
            self._led_on()
            self.after(800, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _laser_toggle(self):
        """Toggle laser on/off (independent of servo tracking)."""
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            if hasattr(self.arduino, "laser_toggle"):
                self._laser_on = not self._laser_on
                self.arduino.laser_toggle(self._laser_on)
                self.on_laser(self._laser_on)
                status = "ON" if self._laser_on else "OFF"
                self.status_text.set(f"Laser: {status}")
                self.on_status(f"Laser: {status}")
                # Update button text
                if hasattr(self, "_laser_btn"):
                    self._laser_btn.configure(text=f"Laser: {status}")
            else:
                # Fallback: use direct commands
                if self._laser_on:
                    self.arduino.send_command("L,0")
                    self._laser_on = False
                    self.on_laser(False)
                else:
                    self.arduino.send_command("L,1")
                    self._laser_on = True
                    self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser toggle failed:\n{e}")

    def _led_on(self):
        """Legacy: kept for backwards compatibility - turns laser on."""
        try:
            if hasattr(self.arduino, "laser_on"):
                self.arduino.laser_on()
            else:
                self.arduino.send_command("L,1")
            self._laser_on = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser ON failed:\n{e}")

    def _led_off(self):
        """Legacy: kept for backwards compatibility - turns laser off."""
        try:
            if hasattr(self.arduino, "laser_off"):
                self.arduino.laser_off()
            else:
                self.arduino.send_command("L,0")
            self._laser_on = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"Laser OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")

        # re-enable connect button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # show connected-only controls
        if not self._connected_controls:
            self._show_connected_controls()
        
        # Initialize auto mode on Arduino (ensures A,pan,tilt commands work)
        try:
            if self.arduino and self.arduino.is_connected():
                if hasattr(self.arduino, "set_auto_tracking_mode"):
                    self.arduino.set_auto_tracking_mode(True)
                
                # Apply saved offsets to Arduino first
                if hasattr(self.arduino, "set_offset"):
                    self.arduino.set_offset(self._pan_offset, self._tilt_offset)
                
                # Try to read offsets from Arduino (it sends OFFSETS,pan,tilt on startup)
                # Wait a bit for Arduino to send the startup message
                self.after(200, self._try_read_arduino_offsets)
        except Exception:
            pass  # Ignore errors during initialization

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # quick call; this sleeps inside blink(), so for buttery UI
            # you can thread it later if you want
            self.after(10, self._safe_blink)
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _safe_blink(self):
        try:
            self.arduino.blink(times=3, interval_s=0.25)
            self.status_text.set("Blink complete.")
            self.on_status("LED: idle")
        except Exception as e:
            messagebox.showerror("Visionary", f"Blink failed:\n{e}")

    def _disconnect_hardware(self):
        if self.arduino and self.arduino.is_connected():
            try:
                self.arduino.disconnect()
            except Exception as e:
                messagebox.showerror("Visionary", f"Disconnect failed:\n{e}")
                return

        self.status_text.set("Arduino disconnected.")
        self.on_status("Hardware: disconnected")
        self._laser_on = False
        self.on_laser(False)

        # remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        if getattr(self, "connect_btn", None) is None:
            self.connect_btn = ttk.Button(
                self, text="Connect Hardware",
                style="Accent.TButton", command=self._connect_hardware
            )
        self.connect_btn.pack(fill="x", pady=4)

    # ---- UI builder for connected state ----
    def _show_connected_controls(self):
        if self._connected_controls:
            # visible already; nothing to rebuild
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # remove the connect button after connect
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        # Laser Control Section
        laser_frame = ttk.LabelFrame(self._connected_controls, text="Laser Control")
        laser_frame.pack(fill="x", pady=(0, 8))
        
        laser_status = "ON" if self._laser_on else "OFF"
        self._laser_btn = ttk.Button(
            laser_frame, text=f"Laser: {laser_status}",
            style="Accent.TButton", command=self._laser_toggle
        )
        self._laser_btn.pack(fill="x", pady=2)

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=4)

        # Servo Control Section
        servo_frame = ttk.LabelFrame(self._connected_controls, text="Servo Control")
        servo_frame.pack(fill="x", pady=(0, 4))

        # Auto-Tracking Toggle
        auto_frame = ttk.Frame(servo_frame)
        auto_frame.pack(fill="x", pady=(2, 4))
        ttk.Label(auto_frame, text="Auto-Tracking:").pack(side="left", padx=(0, 8))
        self._auto_tracking_var = tk.BooleanVar(value=self._auto_tracking_enabled)
        self._auto_tracking_check = ttk.Checkbutton(
            auto_frame, variable=self._auto_tracking_var,
            command=self._on_auto_tracking_toggle
        )
        self._auto_tracking_check.pack(side="left")

        # Manual Pan Control
        pan_row = ttk.Frame(servo_frame)
        pan_row.pack(fill="x", pady=(1, 2))
        ttk.Label(pan_row, text="Pan (L-R):").pack(side="left")
        self._pan_var = tk.IntVar(value=90)
        self._pan_slider = ttk.Scale(
            pan_row, from_=30, to=150, orient="horizontal",
            variable=self._pan_var, command=self._on_pan_drag
        )
        self._pan_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._pan_lbl = ttk.Label(pan_row, text="90°")
        self._pan_lbl.pack(side="left")

        # Manual Tilt Control
        tilt_row = ttk.Frame(servo_frame)
        tilt_row.pack(fill="x", pady=(1, 4))
        ttk.Label(tilt_row, text="Tilt (U-D):").pack(side="left")
        self._tilt_var = tk.IntVar(value=90)
        self._tilt_slider = ttk.Scale(
            tilt_row, from_=40, to=140, orient="horizontal",
            variable=self._tilt_var, command=self._on_tilt_drag
        )
        self._tilt_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._tilt_lbl = ttk.Label(tilt_row, text="90°")
        self._tilt_lbl.pack(side="left")

        ttk.Separator(servo_frame, orient="horizontal").pack(fill="x", pady=4)

        # Offset Adjustment Section
        offset_frame = ttk.LabelFrame(servo_frame, text="Offset Adjustment")
        offset_frame.pack(fill="x", pady=(2, 4))
        
        # Pan Offset Control
        pan_offset_row = ttk.Frame(offset_frame)
        pan_offset_row.pack(fill="x", pady=(2, 1))
        ttk.Label(pan_offset_row, text="Pan Offset:").pack(side="left", padx=(0, 8))
        self._pan_offset_var = tk.IntVar(value=self._pan_offset)
        pan_offset_spin = ttk.Spinbox(
            pan_offset_row, from_=-30, to=30, width=8,
            textvariable=self._pan_offset_var, command=self._on_offset_changed
        )
        pan_offset_spin.pack(side="left", padx=(0, 4))
        pan_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(pan_offset_row, text="° (+ right, - left)").pack(side="left")
        
        # Tilt Offset Control
        tilt_offset_row = ttk.Frame(offset_frame)
        tilt_offset_row.pack(fill="x", pady=(1, 2))
        ttk.Label(tilt_offset_row, text="Tilt Offset:").pack(side="left", padx=(0, 8))
        self._tilt_offset_var = tk.IntVar(value=self._tilt_offset)
        tilt_offset_spin = ttk.Spinbox(
            tilt_offset_row, from_=-30, to=30, width=8,
            textvariable=self._tilt_offset_var, command=self._on_offset_changed
        )
        tilt_offset_spin.pack(side="left", padx=(0, 4))
        tilt_offset_spin.bind('<KeyRelease>', lambda e: self._on_offset_changed())
        ttk.Label(tilt_offset_row, text="° (+ down, - up)").pack(side="left")
        
        # Save Offset Button
        ttk.Button(
            offset_frame, text="Save Offsets",
            command=self._save_offsets
        ).pack(fill="x", pady=(2, 0))

        ttk.Separator(self._connected_controls, orient="horizontal").pack(fill="x", pady=4)

        # Disconnect Button
        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=2)

    # ---- Servo actions ----
    def _on_auto_tracking_toggle(self):
        """Toggle auto-tracking mode for servos.
        
        This controls whether automatic object tracking sends servo commands.
        Manual control always works by ensuring auto mode is enabled.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            enabled = self._auto_tracking_var.get()
            self._auto_tracking_enabled = enabled
            # Always keep Arduino auto mode enabled for manual control to work
            # The toggle only controls whether automatic tracking sends commands
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                # Keep auto mode enabled on Arduino so A,pan,tilt commands always work
                self.arduino.set_auto_tracking_mode(True)
            status = "ON" if enabled else "OFF"
            self.status_text.set(f"Auto-Tracking: {status}")
            self.on_status(f"Auto-Tracking: {status}")
            # Notify main interface if callback provided
            if self.on_auto_tracking_toggle:
                self.on_auto_tracking_toggle(enabled)
        except Exception as e:
            messagebox.showerror("Visionary", f"Auto-tracking toggle failed:\n{e}")

    def _on_pan_drag(self, val_str):
        """Handle pan slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._pan_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._pan_after_id:
            try:
                self.after_cancel(self._pan_after_id)
            except Exception:
                pass
        
        self._pan_after_id = self.after(150, self._send_manual_pan_tilt)

    def _on_tilt_drag(self, val_str):
        """Handle tilt slider drag (manual control)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        angle = int(float(val_str))
        self._tilt_lbl.config(text=f"{angle}°")
        
        # Debounce to avoid spamming the serial line
        if self._tilt_after_id:
            try:
                self.after_cancel(self._tilt_after_id)
            except Exception:
                pass
        
        self._tilt_after_id = self.after(150, self._send_manual_pan_tilt)

    def _send_manual_pan_tilt(self):
        """Send manual pan/tilt commands to Arduino.
        
        Ensures auto mode is enabled on Arduino so A,pan,tilt commands work.
        """
        self._pan_after_id = None
        self._tilt_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        try:
            pan = int(self._pan_var.get())
            tilt = int(self._tilt_var.get())
            
            # Ensure auto mode is enabled on Arduino for A,pan,tilt commands to work
            if hasattr(self.arduino, "set_auto_tracking_mode"):
                self.arduino.set_auto_tracking_mode(True)
            
            # Send servo command (uses A,pan,tilt internally)
            if hasattr(self.arduino, "servo_manual_control"):
                self.arduino.servo_manual_control(pan, tilt)
            else:
                # Fallback: use auto_track method
                self.arduino.servo_auto_track(pan, tilt)
            self.status_text.set(f"Servo: Pan={pan}°, Tilt={tilt}°")
            self.on_status(f"Servo: Pan={pan}°, Tilt={tilt}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set servo angle failed:\n{e}")
    
    # ---- Offset control methods ----
    def _try_read_arduino_offsets(self):
        """Try to read offset values from Arduino serial output.
        
        Arduino sends "OFFSETS,pan,tilt" on startup and after offset changes.
        This method attempts to parse that response using the ArduinoController's read_offset method.
        """
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            # Use ArduinoController's read_offset method if available
            if hasattr(self.arduino, "read_offset"):
                pan_offset, tilt_offset = self.arduino.read_offset()
                if pan_offset != 0 or tilt_offset != 0:  # Only update if we got non-zero values
                    # Update UI and internal state
                    self._pan_offset = pan_offset
                    self._tilt_offset = tilt_offset
                    if hasattr(self, "_pan_offset_var"):
                        self._pan_offset_var.set(pan_offset)
                    if hasattr(self, "_tilt_offset_var"):
                        self._tilt_offset_var.set(tilt_offset)
                    # Update config to match Arduino
                    self.config.save_offsets(pan_offset, tilt_offset)
        except Exception:
            pass  # Ignore errors when reading serial
    
    def _on_offset_changed(self):
        """Handle offset value changes (debounced)."""
        if not self.arduino or not self.arduino.is_connected():
            return
        
        # Debounce to avoid spamming the serial line
        if self._offset_after_id:
            try:
                self.after_cancel(self._offset_after_id)
            except Exception:
                pass
        
        self._offset_after_id = self.after(300, self._apply_offset_changes)
    
    def _apply_offset_changes(self):
        """Apply offset changes to Arduino."""
        self._offset_after_id = None
        if not self.arduino or not self.arduino.is_connected():
            return
        
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values to reasonable range
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            # Send to Arduino
            if hasattr(self.arduino, "set_offset"):
                self.arduino.set_offset(pan_offset, tilt_offset)
            
            self.status_text.set(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offset: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Set offset failed:\n{e}")
    
    def _save_offsets(self):
        """Save current offset values to config file."""
        try:
            pan_offset = int(self._pan_offset_var.get())
            tilt_offset = int(self._tilt_offset_var.get())
            
            # Clamp values
            pan_offset = max(-30, min(30, pan_offset))
            tilt_offset = max(-30, min(30, tilt_offset))
            
            # Save to config
            self.config.save_offsets(pan_offset, tilt_offset)
            
            # Update internal state
            self._pan_offset = pan_offset
            self._tilt_offset = tilt_offset
            
            self.status_text.set(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
            self.on_status(f"Offsets saved: Pan={pan_offset}°, Tilt={tilt_offset}°")
        except Exception as e:
            messagebox.showerror("Visionary", f"Save offsets failed:\n{e}")

```


Fixing mousewheel scrolling for Windows in the scrollable frame:


```python
"""
Scrollable Frame Utility

Provides a scrollable container for widgets that may overflow their available space.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame with a scrollbar that appears when content exceeds available space.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling (Windows uses <MouseWheel>, Linux uses <Button-4> and <Button-5>)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_canvas_configure(self, event):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        # Only scroll if mouse is over the canvas
        if self.canvas.winfo_containing(event.x_root, event.y_root) == self.canvas:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def get_content_frame(self):
        """Return the frame where widgets should be placed."""
        return self.scrollable_frame


```

```python
"""
Scrollable Frame Utility

Provides a scrollable container for widgets that may overflow their available space.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame with a scrollbar that appears when content exceeds available space.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling (Windows uses <MouseWheel>, Linux uses <Button-4> and <Button-5>)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_canvas_configure(self, event):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling (Windows)."""
        # Only scroll if mouse is over the canvas or scrollable frame
        widget = self.canvas.winfo_containing(event.x_root, event.y_root)
        if widget in (self.canvas, self.scrollable_frame) or self._is_child_of(widget, self.scrollable_frame):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_mousewheel_linux(self, event):
        """Handle mousewheel scrolling (Linux)."""
        widget = self.canvas.winfo_containing(event.x_root, event.y_root)
        if widget in (self.canvas, self.scrollable_frame) or self._is_child_of(widget, self.scrollable_frame):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
    
    def _is_child_of(self, widget, parent):
        """Check if widget is a child of parent."""
        if widget is None:
            return False
        current = widget
        while current:
            if current == parent:
                return True
            current = current.master
        return False
    
    def get_content_frame(self):
        """Return the frame where widgets should be placed."""
        return self.scrollable_frame


```


Reducing label frame font sizes to match the smaller button theme:


```python
# Central CSS adjacent theme + HUD utilities (this is pure Tkinter/ttk)

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

# Palette Constants
SEC_BG   = "#464646"
SEC_SURF = "#2b2b2b"
SEC_LINE = "#3a3a3a"
ACCENT   = "#00c2c7"
ACCENT_2 = "#15ffa8"
TEXT     = "#e6edf3"
MUTED    = "#9aa4b1"

# Theme State
_is_dark_mode = True

def font(primary=True, size=10, weight="normal"):
    # Font Selection
    family = "Segoe UI" if primary else "Calibri" 
    return (family, size, weight)

def apply_theme(root: tk.Tk):
    # Initial Theme Setup
    style = ThemedStyle(root)
    try:
        style.set_theme("equilux")
    except Exception:
        style.theme_use("clam")
    
    _configure_custom_styles(style, root)

def toggle_theme(root: tk.Tk):
    # Theme Toggle Logic
    global _is_dark_mode
    style = ThemedStyle(root)
    
    if _is_dark_mode:
        style.set_theme("arc") 
        _is_dark_mode = False
    else:
        style.set_theme("equilux") 
        _is_dark_mode = True
        
    _configure_custom_styles(style, root)
    return _is_dark_mode

def _configure_custom_styles(style, root):
    # Global Defaults
    style.configure(".", font=font(True, 10))
    
    # Button Style (reduced padding for smaller buttons)
    style.configure("TButton", padding=4, font=font(True, 9, "normal"))

    # Labelframe Style (reduced font size)
    style.configure("TLabelframe", font=font(True, 9, "bold"))
    style.configure("TLabelframe.Label", font=font(True, 9, "bold"))

    # Header Style
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    
    # Status Bar Style
    style.configure("Status.TLabel", font=("Consolas", 9), background=style.lookup("TFrame", "background"))

    # Window Background
    bg_color = style.lookup("TFrame", "background")
    root.configure(bg=bg_color)

def style_menu(menu: tk.Menu):
    # Menu Styling
    menu.configure(
        tearoff=0,
        bg="#464646", fg="#e6edf3",
        activebackground="#5e5e5e", activeforeground="#ffffff",
        bd=0, relief="flat"
    )

def add_video_grid(parent: ttk.LabelFrame):
    # Clean Canvas Setup (grid overlay removed for cleaner video feed)
    canvas = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    return canvas
```


Verifying the status bar is correctly positioned:



## Summary of UI Improvements

### 1. Removed grid overlay from video feed
- Removed the grid lines and crosshair overlay from the video canvas
- Video feed now displays without overlay

### 2. Made buttons smaller
- Reduced button padding from `6` to `4` pixels
- Changed button font from `10pt bold` to `9pt normal`
- Reduced spacing between buttons and sections (pady values reduced)
- Reduced label frame font sizes to match

### 3. Added scrollable frame to control panel
- Created a new `ScrollableFrame` utility class
- Wrapped the control panel and narration panel in a scrollable container
- Scrollbar appears automatically when content exceeds available space
- Supports mousewheel scrolling on Windows and Linux
- All buttons remain accessible even when hardware is connected

### 4. Increased video feed size
- Changed grid column weights from `3:2` to `5:2` ratio
- Video feed now takes up more screen space

### 5. Status bar remains at bottom
- Verified status bar is correctly positioned using `pack(side=tk.BOTTOM)`
- Status bar stays fixed at the bottom of the window

All changes are complete and the code passes linting. The UI should now be more compact, with all controls accessible via scrolling when needed, and a larger, cleaner video feed.

---

**User**

Cool, currently the scrollbar is very slow, and it causes the entire control panel to kind of flicker as it loads in new assets, is that normal to ttkthemes?

---

**Cursor**


Investigating the slow scrolling and flicker. Likely due to frequent scroll region updates and canvas redraws. Optimizing the scrollable frame:



Optimizing the scrollable frame to address slow scrolling and flicker. Issues:
1. Scroll speed is too low (1 unit per event)
2. Scrollregion updates on every Configure event, causing flicker

Applying fixes:


```python
"""
Scrollable Frame Utility

Provides a scrollable container for widgets that may overflow their available space.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame with a scrollbar that appears when content exceeds available space.
    Optimized to reduce flickering and improve scroll speed.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Debounce variables for scrollregion updates
        self._update_after_id = None
        self._last_height = 0
        
        # Configure scrollable frame with debounced update
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling (Windows uses <MouseWheel>, Linux uses <Button-4> and <Button-5>)
        # Use bind instead of bind_all to avoid conflicts with other widgets
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)
        # Also bind to scrollable frame for better coverage
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<Button-4>", self._on_mousewheel_linux)
        self.scrollable_frame.bind("<Button-5>", self._on_mousewheel_linux)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_frame_configure(self, event):
        """Debounced scrollregion update to reduce flickering."""
        # Only update if height actually changed
        if event.height != self._last_height:
            self._last_height = event.height
            # Cancel any pending update
            if self._update_after_id:
                self.after_cancel(self._update_after_id)
            # Schedule update after a short delay to batch multiple configure events
            self._update_after_id = self.after(10, self._update_scrollregion)
    
    def _update_scrollregion(self):
        """Update the canvas scrollregion."""
        self._update_after_id = None
        try:
            # Update scrollregion only if canvas is ready
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
        except tk.TclError:
            # Canvas might be destroyed, ignore
            pass
    
    def _on_canvas_configure(self, event):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        # Update scrollregion after width change
        self.after_idle(self._update_scrollregion)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling (Windows)."""
        # Faster scrolling: use 3 units per wheel notch instead of 1
        scroll_amount = int(-3 * (event.delta / 120))
        if scroll_amount != 0:
            self.canvas.yview_scroll(scroll_amount, "units")
    
    def _on_mousewheel_linux(self, event):
        """Handle mousewheel scrolling (Linux)."""
        # Faster scrolling: use 3 units per wheel notch
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
    
    def get_content_frame(self):
        """Return the frame where widgets should be placed."""
        return self.scrollable_frame


```


Optimized the scrollable frame to address both issues:

## Changes Made:

1. **Faster scrolling**: Increased scroll speed from 1 unit to 3 units per wheel notch
2. **Reduced flickering**:
   - Debounced scrollregion updates (10ms delay) to batch multiple configure events
   - Only update scrollregion when the frame height actually changes
   - Use `after_idle` for updates after canvas resize
3. **Better event handling**: Changed from `bind_all` to `bind` on the canvas and scrollable frame to avoid conflicts with other widgets (like the video canvas)

The flickering was caused by recalculating the scrollregion on every Configure event when widgets were added (e.g., when hardware connects). The debouncing batches these updates and only applies them when needed.

Scrolling should now be faster and smoother, with minimal flickering when the control panel content changes.

