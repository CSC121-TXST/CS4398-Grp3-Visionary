"""
Control Panel Module

Defines the right-hand control interface for the Visionary system. Includes core buttons such 
as Start/Stop Camera, Connect Hardware, and Test Laser — all currently placeholder callbacks.

This file is independent and mountable into main_interface.py.
"""

import tkinter as tk
from tkinter import ttk, messagebox

class ControlPanel(ttk.Frame):
    def __init__(self, master=None, camera=None, on_status=None, on_laser=None, on_servo=None, on_fps=None):
        super().__init__(master)
        self.camera = camera
        self.on_status = on_status or (lambda s: None)
        self.on_laser = on_laser or (lambda s: None)
        self.on_servo = on_servo or (lambda s: None)
        self.on_fps = on_fps or (lambda s: None)
        self._build()

    # Layout
    def _build(self):
        ttk.Label(self, text="System Controls", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        box = ttk.Frame(self)
        box.pack(fill="x", pady=(0, 12))

        ttk.Button(box, text="Start Camera", command=self._start).pack(fill="x", pady=4)
        ttk.Button(box, text="Stop Camera", command=self._stop).pack(fill="x", pady=4)
        ttk.Button(box, text="Connect Hardware", command=self._hardware).pack(fill="x", pady=4)
        ttk.Button(box, text="Test Laser", command=self._laser).pack(fill="x", pady=4)

        ttk.Separator(self).pack(fill="x", pady=12)

        ttk.Label(self, text="Status").pack(anchor="w")
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", pady=(4, 0))

    # Actions
    def _start(self):
        if not self.camera:
            messagebox.showerror("Error", "Camera unavailable.")
            self._emit_status("Status: Camera unavailable")
            return
        try:
            self._emit_status("Status: Starting camera…")
            self.camera.start()
            self._emit_status("Status: Camera started")
            self.on_fps("FPS: —")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._emit_status("Status: Start failed")

    def _stop(self):
        if not self.camera:
            self._emit_status("Status: No camera to stop")
            return
        try:
            self.camera.stop()
            self._emit_status("Status: Camera stopped")
            self.on_fps("FPS: —")
        except Exception as e:
            messagebox.showwarning("Warning", str(e))
            self._emit_status("Status: Stop issue")

    def _hardware(self):
        messagebox.showinfo("Hardware", "Connect request sent.")
        self._emit_status("Status: Hardware request sent")
        self.on_servo("Servo: Pan 0°, Tilt 0°")

    def _laser(self):
        self.on_laser("Laser: ON")
        messagebox.showinfo("Laser", "Laser test triggered.")
        self.on_laser("Laser: OFF")

    def _emit_status(self, text):
        self.status.set(text.replace("Status: ", ""))
        self.on_status(text)
