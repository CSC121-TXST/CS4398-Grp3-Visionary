"""
Main Interface Module

This module provides the main user interface for the Visionary system, integrating all 
components into a cohesive user experience.

This file defines ONLY the window layout to match the project wireframe:
- Menubar: File, Settings, Help (top)
- Centered title label: "Visionary" just under the menubar
- Main split: left Video Feed area, right Control Panel area
- Footer status bar with segments: Laser, Servo, Status, FPS

No camera, no hardware, no buttons. These areas are "slots" where real widgets can be mounted 
later (e.g., ui/video_panel.py, ui/control_panel.py).

"""
import tkinter as tk
from tkinter import ttk, messagebox

class _Mountable(ttk.Frame):
    """Fallback stub used when real panels are not present."""
    def __init__(self, master, text: str):
        super().__init__(master)
        lbl = ttk.Label(self, text=text, anchor="center")
        lbl.pack(expand=True, fill="both", padx=12, pady=12)

class VisionaryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

        self._build_menubar()

        title = ttk.Label(self, text="Visionary", font=("Segoe UI", 16, "bold"))
        title.pack(side=tk.TOP, pady=(6, 6))

        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.video_frame = ttk.LabelFrame(main, text="Video Feed")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=(0,10))

        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=(0,10))

        self._video_widget = _Mountable(self.video_frame, "Video panel goes here…")
        self._video_widget.pack(expand=True, fill="both")

        self._control_widget = _Mountable(self.control_frame, "Controls go here…")
        self._control_widget.pack(expand=True, fill="both")

        self._build_statusbar()

    def _build_menubar(self):
        mb = tk.Menu(self)

        file_menu = tk.Menu(mb, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_exit)
        mb.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(mb, tearoff=0)
        # (settings items to be added later)
        mb.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(mb, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        mb.add_cascade(label="Help", menu=help_menu)

        self.config(menu=mb)

    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(side=tk.BOTTOM, fill="x")

        self.var_laser = tk.StringVar(value="Laser: OFF")
        self.var_servo = tk.StringVar(value="Servo: Pan 0°, Tilt 0°")
        self.var_status = tk.StringVar(value="Status: Idle")
        self.var_fps = tk.StringVar(value="FPS: —")

        # Four evenly spaced labels
        for i in range(4):
            bar.grid_columnconfigure(i, weight=1)

        ttk.Label(bar, textvariable=self.var_laser, anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.var_servo, anchor="center").grid(row=0, column=1)
        ttk.Label(bar, textvariable=self.var_status, anchor="center").grid(row=0, column=2)
        ttk.Label(bar, textvariable=self.var_fps, anchor="e").grid(row=0, column=3, sticky="e")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary\n",
            "Visionary – CS4398 Group 3\n"
            "Wireframe-aligned Tkinter skeleton.\n"
            "This layout provides mount points for video and controls."
        )

    def on_exit(self):
        self.destroy()

# Local launcher for direct run
if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()
