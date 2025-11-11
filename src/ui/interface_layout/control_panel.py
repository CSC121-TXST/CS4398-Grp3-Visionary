"""
Control Panel Module

"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2

from ui.interface_layout.control_buttons.camera_controls import CameraControls
from ui.interface_layout.control_buttons.hardware_controls import HardwareControls
from ui.interface_layout.vision_narration_panel import VisionNarrationPanel


class ControlPanel(ttk.Frame):
    """Right-side UI composed from two small sub-widgets."""
    def __init__(self, master=None, camera=None, arduino=None, on_status=None, on_laser=None, narrator=None, on_narrate_request=None):
        """
        camera:  object with start(), stop(), is_running()
        arduino: object with connect(), disconnect(), is_connected(), send_command(), blink()
        on_status: callback(str) to update status bar
        on_laser:  callback(bool) to update laser label
        narrator: VisionNarrator instance (optional)
        on_narrate_request: callback() to trigger narration (optional)
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.arduino = arduino
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda on: None)
        self.narrator = narrator
        self.on_narrate_request = on_narrate_request

        self._build()

    def _build(self):
        # Section title
        ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        ).pack(anchor="w", pady=(0, 10))

        # Buttons container (camera + hardware)
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 15))

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
            status_text=self._status_var()
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

        # Vision Narration Panel
        if self.narrator or self.on_narrate_request:
            self.narration_panel = VisionNarrationPanel(
                self,
                narrator=self.narrator,
                on_narrate_request=self.on_narrate_request
            )
            self.narration_panel.pack(fill="both", expand=True, pady=(0, 0))

    # share one StringVar for status inside the panel
    def _status_var(self):
        if not hasattr(self, "status_text"):
            self.status_text = tk.StringVar(value="Idle")
        return self.status_text
