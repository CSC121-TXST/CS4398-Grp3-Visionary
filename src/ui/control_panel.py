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
        self._connected_controls = None  # frame that holds LED buttons after connect


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

        # If already connected, just render controls and bail
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
            # Proof-of-life: LED on ~1.2s then off
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

        # Only touch the button if it still exists
        if getattr(self, "connect_btn", None) is not None:
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass

        # Now replace the button with the LED controls (if not already shown)
        if not self._connected_controls:
            self._show_connected_controls()

    # Placeholder Methods
    def _test_laser(self):
        self.status_text.set("Testing laser module...")
        messagebox.showinfo("Visionary", "Laser test initiated (placeholder).")

    def _show_connected_controls(self):
        # Already built? just ensure visible and return
        if self._connected_controls:
            try:
                self._connected_controls.pack_forget()
            except Exception:
                pass

        # Remove the connect button once connected
        if getattr(self, "connect_btn", None):
            try:
                self.connect_btn.destroy()
            except Exception:
                pass
            self.connect_btn = None

        # Build the LED/Disconnect button group
        self._connected_controls = ttk.Frame(self)
        self._connected_controls.pack(fill="x", pady=(0, 15))

        ttk.Button(
            self._connected_controls, text="💡 LED ON",
            style="Accent.TButton", command=self._led_on
        ).pack(fill="x", pady=4)

        ttk.Button(
            self._connected_controls, text="💤 LED OFF",
            style="Accent.TButton", command=self._led_off
        ).pack(fill="x", pady=4)

        ttk.Button(
            self._connected_controls, text="✨ Blink (x3)",
            style="Accent.TButton", command=self._blink_led
        ).pack(fill="x", pady=4)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(
            self._connected_controls, text="⏏ Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)

    def _blink_led(self):
        if not self.arduino or not self.arduino.is_connected():
            messagebox.showwarning("Visionary", "Arduino not connected.")
            return
        try:
            self.status_text.set("Blinking LED…")
            self.on_status("LED: blinking")
            # non-blocking feel: do quick blink without freezing UI
            # (blink() itself sleeps; for a super-smooth UI you could thread it later)
            self.after(10, lambda: self._safe_blink())
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
        self.on_laser(False)
        self._led_state = False

        # Remove connected controls and bring back Connect button
        if self._connected_controls:
            try:
                self._connected_controls.destroy()
            except Exception:
                pass
            self._connected_controls = None

        self.connect_btn = ttk.Button(
            self, text="🔌 Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=4)
