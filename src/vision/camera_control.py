"""
Camera Control
- SimpleCamera: opens the webcam, reads frames, and schedules the next read.
- CanvasRenderer: knows how to draw a frame onto a Tk Canvas.
"""

import time
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk

class CanvasRenderer:
    """ Draws frames onto a Tkinter Canvas. This class only knows about the canvas and Tk images. """

    def __init__(self, canvas):
        self.canvas = canvas
        self._photo = None 
        self._image_item = None 

    def clear(self):
        """Remove the last drawn image from the canvas (used when stopping)."""
        if self._image_item is not None:
            try:
                self.canvas.delete(self._image_item)
            except Exception:
                pass
            self._image_item = None
        self._photo = None

    def draw(self, frame_bgr):
        """
        Accepts a BGR frame (from OpenCV), resizes it to the canvas,
        converts to Tk image, and displays it.
        """
        if frame_bgr is None:
            return
        # Resize to current canvas size (simple stretch; aspect-correct can come later)
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())

        try:
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Resize
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
            # Make Tk image and draw
            img = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(img)
            if self._image_item is None:
                self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                self.canvas.itemconfig(self._image_item, image=self._photo)
        except Exception:
            # If anything goes wrong (e.g., canvas not ready), just ignore this frame
            pass


class SimpleCamera:
    """
    Tiny webcam helper.
    - Captures frames from OpenCV.
    - Hands frames to a renderer (CanvasRenderer).

    This class does NOT know about Tk. It only expects the renderer to have a draw(frame_bgr) method.
    """

    def __init__(
        self,
        index: int = 0,
        mirror: bool = True,
        on_fps: Optional[Callable[[float], None]] = None,
        renderer: Optional[CanvasRenderer] = None,
        canvas=None,
        frame_processor: Optional[Callable] = None,
    ):
        """
        Initialize SimpleCamera.
        
        Args:
            index: Camera device index (default 0)
            mirror: Whether to horizontally flip the frame
            on_fps: Optional callback(float) for FPS updates
            renderer: Optional CanvasRenderer instance
            canvas: Optional Tkinter canvas (creates CanvasRenderer if renderer is None)
            frame_processor: Optional callable(frame_bgr) -> processed_frame_bgr
                           Called after mirroring, before rendering.
                           Used for object detection/tracking processing.
        """
        self.index = index
        self.mirror = mirror
        self.on_fps = on_fps
        self.frame_processor = frame_processor

        # Camera state
        self._cap = None
        self._running = False
        self._after_id = None

        # If a canvas is given, wrap it in a CanvasRenderer
        if renderer is None and canvas is not None:
            renderer = CanvasRenderer(canvas)
        self.renderer = renderer

        # We need a Tk scheduler for the read loop; using the canvas is the simplest place to get it.
        self._tk_owner = canvas if canvas is not None else getattr(renderer, "canvas", None)

    def start(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._safe_release()
            raise RuntimeError(f"Unable to open camera index {self.index}")
        self._running = True
        self._schedule_next()

    def stop(self):
        self._running = False
        # cancel scheduled callback
        if self._after_id is not None and self._tk_owner is not None:
            try:
                self._tk_owner.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        # release camera
        self._safe_release()

        # clear last drawn image
        if self.renderer:
            try:
                self.renderer.clear()
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._running

    def _safe_release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _schedule_next(self):
        """Schedule the next frame read (~60 FPS budget -> 16 ms)."""
        if not self._running:
            return
        delay_ms = 15  # ~30–60 FPS depending on system
        if self._tk_owner is None:
            # If no Tk owner, just stop; we rely on Tk's .after() in this simple version
            self.stop()
            return
        self._after_id = self._tk_owner.after(delay_ms, self._read_one)

    def _read_one(self):
        if not self._running or self._cap is None:
            return

        t0 = time.time()
        ok, frame_bgr = self._cap.read()

        if ok:
            # Mirror the frame if requested
            if self.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            
            # Apply frame processor (e.g., object detection/tracking)
            # This allows external processing like drawing bounding boxes
            if self.frame_processor is not None:
                try:
                    frame_bgr = self.frame_processor(frame_bgr)
                except Exception as e:
                    print(f"Frame processor error: {e}")
                    raise
            
            # Render the (possibly processed) frame
            if self.renderer:
                self.renderer.draw(frame_bgr)

        # FPS callback
        if self.on_fps:
            dt = max(1e-6, time.time() - t0)
            self.on_fps(1.0 / dt)

        # schedule another read
        self._schedule_next()
