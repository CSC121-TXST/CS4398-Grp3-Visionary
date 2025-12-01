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
        ).pack(fill="x", pady=4)

        ttk.Button(
            self, text="Stop Camera",
            style="Accent.TButton", command=self._stop_camera
        ).pack(fill="x", pady=4)

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
