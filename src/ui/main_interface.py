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

from ui.style import apply_theme, style_menu, add_video_grid, ACCENT

class _Mountable(ttk.Frame):
    """Simple placeholder used when real panels are not yet implemented."""
    def __init__(self, master, text: str):
        super().__init__(master)
        lbl = ttk.Label(self, text=text, anchor="center")
        lbl.pack(expand=True, fill="both", padx=12, pady=12)


class VisionaryApp(tk.Tk):
    """Main Tkinter window for the Visionary application."""
    def __init__(self):
        super().__init__()
        self.title("Visionary")
        self.geometry("1100x650")
        self.minsize(980, 560)

        # Apply dark theme before creating widgets
        apply_theme(self)

        # Build interface
        self._build_menubar()
        self._build_title()
        self._build_main_area()
        self._build_statusbar()

    def _build_menubar(self):
        header = ttk.Frame(self, padding=(10, 6))
        header.pack(side=tk.TOP, fill="x")

        def make_menu_button(parent, text):
            mb = ttk.Menubutton(parent, text=text)
            menu = tk.Menu(mb, tearoff=0)
            style_menu(menu)
            mb["menu"] = menu
            mb.pack(side="left", padx=(0, 12))
            return menu

        file_menu = make_menu_button(header, "File")
        file_menu.add_command(label="Exit", command=self.on_exit)

        settings_menu = make_menu_button(header, "Settings")
        # Chris To-Do: add settings_menu.add_checkbutton(...)

        help_menu = make_menu_button(header, "Help")
        help_menu.add_command(label="About", command=self._show_about)

    def _build_title(self):
        """Places the centered Visionary title label."""
        title = ttk.Label(
            self,
            text="Visionary",
            font=("Consolas", 18, "bold"),
            foreground=ACCENT
        )
        title.pack(side=tk.TOP, pady=(6, 6))

    def _build_main_area(self):
        """Constructs the main layout with video and control panels."""
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Left: Video Feed
        self.video_frame = ttk.LabelFrame(main, text="Video Feed")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        self._video_canvas = add_video_grid(self.video_frame)

        # Right: Control Panel
        self.control_frame = ttk.LabelFrame(main, text="Control Panel")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        self._control_widget = _Mountable(self.control_frame, "Controls go here…")
        self._control_widget.pack(expand=True, fill="both")

    def _build_statusbar(self):
        """Creates the footer status bar with key telemetry values."""
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(side=tk.BOTTOM, fill="x")

        self.var_laser = tk.StringVar(value="Laser: OFF")
        self.var_servo = tk.StringVar(value="Servo: Pan 0°, Tilt 0°")
        self.var_status = tk.StringVar(value="Status: Idle")
        self.var_fps = tk.StringVar(value="FPS: —")

        for i in range(4):
            bar.grid_columnconfigure(i, weight=1)

        ttk.Label(bar, textvariable=self.var_laser, anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.var_servo, anchor="center").grid(row=0, column=1)
        ttk.Label(bar, textvariable=self.var_status, anchor="center").grid(row=0, column=2)
        ttk.Label(bar, textvariable=self.var_fps, anchor="e").grid(row=0, column=3, sticky="e")

    def _show_about(self):
        """Displays About dialog."""
        messagebox.showinfo(
            "About Visionary",
            "Visionary – CS4398 Group 3\n\n"
            "Wireframe-aligned Tkinter skeleton (dark theme).\n"
            "This layout provides mount points for video and controls."
        )

    def on_exit(self):
        """Safely close the application."""
        self.destroy()

if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()
