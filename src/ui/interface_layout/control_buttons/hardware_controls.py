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
    """
    def __init__(self, parent, arduino, on_status, on_laser, status_text: tk.StringVar):
        super().__init__(parent)
        self.arduino = arduino
        self.on_status = on_status
        self.on_laser = on_laser
        self.status_text = status_text

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

    def _led_on(self):
        try:
            self.arduino.send_command("LED_ON")
            self.on_laser(True)
        except Exception as e:
            messagebox.showerror("Visionary", f"LED_ON failed:\n{e}")

    def _led_off(self):
        try:
            self.arduino.send_command("LED_OFF")
            self.on_laser(False)
        except Exception as e:
            messagebox.showerror("Visionary", f"LED_OFF failed:\n{e}")

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

        ttk.Button(
            self._connected_controls, text="LED ON",
            style="Accent.TButton", command=self._led_on
        ).pack(fill="x", pady=4)

        ttk.Button(
            self._connected_controls, text="LED OFF",
            style="Accent.TButton", command=self._led_off
        ).pack(fill="x", pady=4)

        ttk.Button(
            self._connected_controls, text="Blink (x3)",
            style="Accent.TButton", command=self._blink_led
        ).pack(fill="x", pady=4)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(
            self._connected_controls, text="Disconnect",
            style="Accent.TButton", command=self._disconnect_hardware
        ).pack(fill="x", pady=4)
