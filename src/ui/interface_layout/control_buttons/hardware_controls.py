# ui/controls/hardware_controls.py
import tkinter as tk
from tkinter import ttk, messagebox

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

        # Connect button (appears initially; replaced after connect)
        self.connect_btn = ttk.Button(
            self, text="Connect Hardware",
            style="Accent.TButton", command=self._connect_hardware
        )
        self.connect_btn.pack(fill="x", pady=4)

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
