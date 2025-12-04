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
