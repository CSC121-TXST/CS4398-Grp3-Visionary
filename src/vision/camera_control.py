"""
Camera Control Module

This module handles camera initialization, frame capture, and
basic camera operations for the Visionary system.

Usage:
    cam = OpenCVCameraController(canvas=self._video_canvas, index=0, mirror=True,
                                 on_fps=lambda f: self.var_fps.set(f"FPS: {f:4.1f}"))
    cam.start()
    cam.stop()
"""

import time
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk


class SimpleCamera:
    def __init__(
        self,
        canvas,
        index: int = 0,
        mirror: bool = True,
        on_fps: Optional[Callable[[float], None]] = None,
    ):
        self.canvas = canvas
        self.index = index
        self.mirror = mirror
        self.on_fps = on_fps

        self._cap = None
        self._running = False
        self._after_id = None
        self._photo = None      # keep reference to prevent GC
        self._image_item = None # canvas image id

    # --- lifecycle ---
    def start(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Unable to open camera index {self.index}")
        self._running = True
        self._loop()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._image_item is not None:
            try:
                self.canvas.delete(self._image_item)
            except Exception:
                pass
            self._image_item = None
        self._photo = None

    def is_running(self) -> bool:
        return self._running

    # --- loop ---
    def _loop(self):
        if not self._running or self._cap is None:
            return

        t0 = time.time()
        ok, frame = self._cap.read()
        if ok:
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Resize to current canvas size (simple stretch – no aspect math)
            w = max(1, self.canvas.winfo_width())
            h = max(1, self.canvas.winfo_height())
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            # BGR -> RGB -> Tk image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(img)

            if self._image_item is None:
                self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                self.canvas.itemconfig(self._image_item, image=self._photo)

        # FPS callback
        if self.on_fps:
            dt = max(1e-6, time.time() - t0)
            self.on_fps(1.0 / dt)

        # ~30–60 FPS depending on camera/CPU
        self._after_id = self.canvas.after(15, self._loop)