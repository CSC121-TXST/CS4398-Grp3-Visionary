# ui/interface_layout/statusbar.py
import tkinter as tk
from tkinter import ttk

class StatusBar(ttk.Frame):
    """Bottom bar with a few text fields you can update."""
    def __init__(self, parent):
        super().__init__(parent, padding=(8, 4))

        self.var_laser = tk.StringVar(value="Laser: OFF")
        self.var_servo = tk.StringVar(value="Servo: Pan 0°, Tilt 0°")
        self.var_status = tk.StringVar(value="Status: Idle")
        self.var_fps   = tk.StringVar(value="FPS: —")

        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

        ttk.Label(self, textvariable=self.var_laser, anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.var_servo, anchor="center").grid(row=0, column=1)
        ttk.Label(self, textvariable=self.var_status, anchor="center").grid(row=0, column=2)
        ttk.Label(self, textvariable=self.var_fps,   anchor="e").grid(row=0, column=3, sticky="e")
