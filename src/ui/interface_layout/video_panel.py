# ui/interface_layout/video_panel.py
import tkinter as tk
from tkinter import ttk
from ui.style import add_video_grid
from vision.camera_control import SimpleCamera

class VideoPanel(ttk.LabelFrame):
    """Left video area, owns canvas and camera."""
    def __init__(self, parent, on_fps):
        super().__init__(parent, text="Video Feed")
        # Canvas with grid/crosshair
        self.canvas = add_video_grid(self)
        # Camera instance that draws into the canvas
        self.camera = SimpleCamera(
            canvas=self.canvas,
            index=0,
            mirror=True,
            on_fps=on_fps
        )