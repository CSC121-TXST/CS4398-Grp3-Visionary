"""
Main Interface Module

This module provides the main user interface for the Visionary system,
integrating all components into a cohesive user experience.

"""
import tkinter as tk
from tkinter import ttk, messagebox

class VisionaryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visionary – Intelligent Camera System")
        self.geometry("900x500")
        self.minsize(800, 400)

        # Menu bar
        self._build_menubar()

        # Main layout
        self._build_layout()

        # Status bar
        self._build_statusbar()

    def _build_menubar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_layout(self):
        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Placeholder frames
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed", width=500, height=400)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.control_frame = ttk.LabelFrame(main_frame, text="Control Panel", width=300)
        self.control_frame.pack(side="right", fill="y", padx=5, pady=5)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def _show_about(self):
        messagebox.showinfo(
            "About Visionary",
            "Visionary – CS4398 Group 3\n\nBasic skeleton UI ready for feature integration."
        )


# Simple launcher for now
if __name__ == "__main__":
    app = VisionaryApp()
    app.mainloop()
