"""
Control Panel Module

Defines the right-hand control interface for the Visionary system. Includes core buttons such 
as Start/Stop Camera, Connect Hardware, and Test Laser — all currently placeholder callbacks.

This file is independent and mountable into main_interface.py.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from ui.style import ACCENT, ACCENT_2

class ControlPanel(ttk.Frame):
    """Control panel for Visionary's right-side UI."""
    def __init__(self, master=None, camera=None, arduino=None, on_status=None, on_laser=None):
        """
        camera: object with start(), stop(), is_running()
        on_status: optional callback(str) to update app status bar
        """
        super().__init__(master, padding=10)
        self.camera = camera
        self.on_status = on_status or (lambda s: None)
        self.arduino = arduino
        self.on_laser = on_laser or (lambda on: None)
        self._led_state = False  # for toggle button behavior
        self._build_layout()

    # Layout
    def _build_layout(self):
        # Section title
        header = ttk.Label(
            self, text="System Controls",
            font=("Consolas", 13, "bold"), foreground=ACCENT
        )
        header.pack(anchor="w", pady=(0, 10))

        # Buttons container
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(0, 15))

        self.start_btn = ttk.Button(
            btns, text="▶ Start Camera",
            style="Accent.TButton", command=self._start_camera
        )
        self.start_btn.pack(fill="x", pady=4)

        self.stop_btn = ttk.Button(
            btns, text="■ Stop Camera",
            style="Accent.TButton", command=self._stop_camera
        )
        self.stop_btn.pack(fill="x", pady=4)

        self.connect_btn = ttk.Button(
            btns, text="🔌 Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=4)

        self.laser_btn = ttk.Button(
            btns, text="💡 Test Laser",
            style="Accent.TButton", command=self._test_laser
        )
        self.laser_btn.pack(fill="x", pady=4)

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Status labels
        status_header = ttk.Label(
            self, text="System Status",
            font=("Consolas", 12, "bold"), foreground=ACCENT_2
        )
        status_header.pack(anchor="w", pady=(0, 6))

        self.status_text = tk.StringVar(value="Idle")
        ttk.Label(self, textvariable=self.status_text, foreground="#9aa4b1").pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=12)

        # Placeholder for advanced controls
        ttk.Label(self, text="[Additional controls coming soon...]", foreground="#666").pack(anchor="w", pady=(10, 0))

    # Functioning Methods
        # ---- Callbacks ----
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

    def _connect_hardware(self):
        if not self.arduino:
            messagebox.showerror("Visionary", "Arduino controller not available.")
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
            # Proof-of-life: turn the LED on for ~1.2s then off
            self._led_on()
            self.after(1200, self._led_off_then_ready)
        else:
            self.status_text.set("Arduino not found.")
            self.on_status("Hardware: not found")
            self.connect_btn.configure(state="normal")

    def _led_on(self):
        try:
            self.arduino.send_command("LED_ON")
            self._led_state = True
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"LED_ON failed:\n{e}")

    def _led_off(self):
        try:
            self.arduino.send_command("LED_OFF")
            self._led_state = False
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"LED_OFF failed:\n{e}")

    def _led_off_then_ready(self):
        self._led_off()
        self.status_text.set("Proof-of-life complete. Ready.")
        self.on_status("Hardware: ready")
        self.connect_btn.configure(state="normal")

    # Placeholder Methods
    def _test_laser(self):
        self.status_text.set("Testing laser module...")
        messagebox.showinfo("Visionary", "Laser test initiated (placeholder).")
