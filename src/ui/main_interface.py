"""
Main Interface Module

Provides the main user interface for the Visionary system.

Layout:
- Custom dark header bar: File, Settings, Help
- Centered title label: "Visionary"
- Main area split: Video Feed (left), Control Panel (right)
- Footer status bar: Laser, Servo, Status, FPS

No camera, hardware, or buttons are implemented here.
These areas act as "slots" for mounting real widgets later
(e.g., ui/video_panel.py, ui/control_panel.py).
"""

import tkinter as tk
from tkinter import ttk, messagebox

# Theme
try:
    from ttkthemes import ThemedTk
    BaseTk = ThemedTk
except:
    BaseTk = tk.Tk

from ui.style import apply_theme, style_menu, DEFAULT_THEME
from ui.control_panel import ControlPanel
from vision.camera_control import SimpleCamera

class VisionaryApp(BaseTk):
    def __init__(self):
        super().__init__()

        # Window
        self.title("Visionary")
        self.geometry("1200x720")
        self.minsize(1040, 600)

        # Theme
        apply_theme(self, DEFAULT_THEME)

        # UI
        self._build_header()
        self._build_main_area()
        self._build_statusbar()

    # Header
    def _build_header(self):
        header = ttk.Frame(self, padding=(12, 10))
        header.pack(side=tk.TOP, fill="x")

        ttk.Label(header, text="Visionary", style="Title.TLabel").pack(side=tk.LEFT)

        container = ttk.Frame(header)
        container.pack(side=tk.RIGHT)

        def menu_btn(label):
            btn = ttk.Menubutton(container, text=label)
            menu = tk.Menu(btn)
            style_menu(menu)
            btn["menu"] = menu
            btn.pack(side="left", padx=8)
            return menu

        file_menu = menu_btn("File")
        file_menu.add_command(label="Exit", command=self.on_exit)

        settings_menu = menu_btn("Settings")
        settings_menu.add_command(label="Preferences...", command=self._open_settings)

        help_menu = menu_btn("Help")
        help_menu.add_command(label="About", command=self._show_about)

    # Body
    def _build_main_area(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=3)
        main.grid_rowconfigure(0, weight=1)

        video = ttk.Frame(main)
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(video, text="Video Feed", style="Section.TLabel").pack(anchor="w", padx=8, pady=(8, 6))
        wrap = ttk.Frame(video)
        wrap.pack(expand=True, fill="both", padx=8, pady=8)
        self._video_canvas = tk.Canvas(wrap, highlightthickness=0)
        self._video_canvas.pack(expand=True, fill="both")

        self.camera = SimpleCamera(canvas=self._video_canvas, index=0, mirror=True)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(controls, text="Controls", style="Section.TLabel").pack(anchor="w", padx=8, pady=(8, 6))
        panel = ttk.Frame(controls)
        panel.pack(expand=True, fill="both", padx=8, pady=8)

        self._control = ControlPanel(
            panel,
            camera=self.camera,
            on_status=self._set_status,
            on_laser=self._set_laser,
            on_servo=self._set_servo,
            on_fps=self._set_fps,
        )
        self._control.pack(fill="both", expand=True)

    # Status Bar
    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(10, 6))
        bar.pack(side=tk.BOTTOM, fill="x")

        bar.grid_columnconfigure(0, weight=1)
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_columnconfigure(2, weight=1)
        bar.grid_columnconfigure(3, weight=1)

        self.var_laser = tk.StringVar(value="Laser: OFF")
        self.var_servo = tk.StringVar(value="Servo: Pan 0°, Tilt 0°")
        self.var_status = tk.StringVar(value="Status: Idle")
        self.var_fps = tk.StringVar(value="FPS: —")

        ttk.Label(bar, textvariable=self.var_laser).grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.var_servo).grid(row=0, column=1, sticky="w")
        ttk.Label(bar, textvariable=self.var_status).grid(row=0, column=2, sticky="w")
        ttk.Label(bar, textvariable=self.var_fps).grid(row=0, column=3, sticky="e")

    # Setters
    def _set_laser(self, text): self.var_laser.set(text)
    def _set_servo(self, text): self.var_servo.set(text)
    def _set_status(self, text): self.var_status.set(text)
    def _set_fps(self, text): self.var_fps.set(text)

    # Actions
    def _open_settings(self):
        messagebox.showinfo("Settings", "Preferences go here.")

    def _show_about(self):
        messagebox.showinfo("About", "Visionary")

    def on_exit(self):
        self.destroy()

if __name__ == "__main__":
    VisionaryApp().mainloop()
