"""
Camera Control Module

This module provides camera capture and frame rendering functionality for the Visionary system.

Components:
    - SimpleCamera: Manages webcam capture, frame processing, and scheduling
    - CanvasRenderer: Renders OpenCV frames onto Tkinter Canvas widgets

Architecture:
    Camera → OpenCV VideoCapture → Frame Processing → CanvasRenderer → Tkinter Display
    
The camera loop runs asynchronously using Tkinter's event scheduler, allowing the UI
to remain responsive while processing video frames.
"""

import time
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk


class CanvasRenderer:
    """
    Renders OpenCV BGR frames onto a Tkinter Canvas widget.
    
    This class handles the conversion from OpenCV's BGR format to Tkinter's
    PhotoImage format and manages the canvas image item lifecycle.
    
    Responsibilities:
    - Convert BGR frames to RGB for display
    - Resize frames to match canvas dimensions
    - Create/update Tkinter PhotoImage objects
    - Manage canvas image item creation and updates
    """

    def __init__(self, canvas):
        """
        Initialize CanvasRenderer.
        
        Args:
            canvas: Tkinter Canvas widget to draw frames on
        """
        self.canvas = canvas
        self._photo: Optional[ImageTk.PhotoImage] = None  # Tkinter PhotoImage object
        self._image_item = None  # Canvas image item ID

    def clear(self):
        """
        Remove the last drawn image from the canvas.
        
        Used when stopping the camera to clean up the display.
        Safe to call multiple times.
        """
        if self._image_item is not None:
            try:
                self.canvas.delete(self._image_item)
            except Exception:
                # Canvas may be destroyed, ignore errors
                pass
            self._image_item = None
        self._photo = None

    def draw(self, frame_bgr):
        """
        Draw a BGR frame onto the canvas.
        
        Converts OpenCV BGR format to RGB, resizes to canvas dimensions,
        and updates the canvas image item.
        
        Args:
            frame_bgr: OpenCV BGR frame (numpy array) from camera or processing pipeline
        
        Note:
            If frame is None or canvas is not ready, this method silently returns.
            Uses simple stretch resize (aspect ratio not preserved).
        """
        if frame_bgr is None:
            return
        
        # === Get Canvas Dimensions ===
        # Ensure minimum size to avoid division by zero
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())

        try:
            # === Convert BGR to RGB ===
            # OpenCV uses BGR, but PIL/Tkinter expect RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # === Resize to Canvas Size ===
            # Simple stretch resize (aspect ratio not preserved)
            # TODO: Could add aspect-correct resize with letterboxing
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
            
            # === Create Tkinter PhotoImage ===
            img = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(img)
            
            # === Update Canvas ===
            # Create image item on first draw, update on subsequent draws
            if self._image_item is None:
                self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                self.canvas.itemconfig(self._image_item, image=self._photo)
        except Exception:
            # If anything goes wrong (e.g., canvas destroyed, invalid frame),
            # silently ignore this frame to avoid crashing the application
            pass


class SimpleCamera:
    """
    Manages webcam capture and frame processing pipeline.
    
    This class handles:
    - Opening/closing camera via OpenCV VideoCapture
    - Reading frames from the camera
    - Applying frame processing (mirroring, object detection/tracking)
    - Scheduling frame reads using Tkinter's event loop
    - FPS calculation and reporting
    
    Architecture:
        Camera → OpenCV VideoCapture → Frame Processing → Renderer → Display
        
    The frame read loop runs asynchronously using Tkinter's .after() scheduler,
    allowing the UI to remain responsive. Frames are processed through an optional
    frame_processor callback (e.g., for object detection/tracking) before rendering.
    
    Note:
        This class requires Tkinter for scheduling. The canvas parameter is used
        to access the Tk root for the event scheduler.
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
            index: Camera device index (default 0, typically the first webcam)
            mirror: Whether to horizontally flip the frame (True = mirror mode)
            on_fps: Optional callback(float) for FPS updates
            renderer: Optional CanvasRenderer instance for frame display
            canvas: Optional Tkinter canvas (creates CanvasRenderer if renderer is None)
            frame_processor: Optional callable(frame_bgr) -> processed_frame_bgr
                           Called after mirroring, before rendering.
                           Used for object detection/tracking processing.
                           Example: tracker.process_frame(frame_bgr)
        """
        # === Configuration ===
        self.index = index
        self.mirror = mirror
        self.on_fps = on_fps
        self.frame_processor = frame_processor

        # === Camera State ===
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._after_id = None  # Tkinter after() callback ID

        # === Renderer Setup ===
        # If canvas is provided but no renderer, create one automatically
        if renderer is None and canvas is not None:
            renderer = CanvasRenderer(canvas)
        self.renderer = renderer

        # === Tkinter Scheduler Access ===
        # We need access to Tk's .after() method for scheduling frame reads
        # Use canvas as the Tk owner (canvas.master is the Tk root)
        self._tk_owner = canvas if canvas is not None else getattr(renderer, "canvas", None)

    def start(self):
        """
        Start camera capture and frame processing loop.
        
        Opens the camera device and begins the asynchronous frame read loop.
        
        Raises:
            RuntimeError: If camera cannot be opened (device not found, in use, etc.)
        """
        if self._running:
            return  # Already running
        
        # === Open Camera ===
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._safe_release()
            raise RuntimeError(
                f"Unable to open camera index {self.index}. "
                "Check that the camera is connected and not in use by another application."
            )
        
        # === Start Frame Loop ===
        self._running = True
        self._schedule_next()

    def stop(self):
        """
        Stop camera capture and clean up resources.
        
        Cancels scheduled frame reads, releases the camera, and clears the display.
        Safe to call multiple times.
        """
        self._running = False
        
        # === Cancel Scheduled Callback ===
        if self._after_id is not None and self._tk_owner is not None:
            try:
                self._tk_owner.after_cancel(self._after_id)
            except Exception:
                # Tk owner may be destroyed, ignore errors
                pass
            self._after_id = None

        # === Release Camera ===
        self._safe_release()

        # === Clear Display ===
        if self.renderer:
            try:
                self.renderer.clear()
            except Exception:
                # Renderer may be destroyed, ignore errors
                pass

    def is_running(self) -> bool:
        """
        Check if camera is currently running.
        
        Returns:
            True if camera is active and capturing frames, False otherwise
        """
        return self._running

    def _safe_release(self):
        """
        Safely release the camera resource.
        
        Handles errors gracefully if camera is already released or in an invalid state.
        """
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                # Camera may already be released, ignore errors
                pass
            self._cap = None

    def _schedule_next(self):
        """
        Schedule the next frame read using Tkinter's event scheduler.
        
        Uses a 15ms delay (~60 FPS target) to balance responsiveness and performance.
        The actual FPS will depend on frame processing time and system performance.
        """
        if not self._running:
            return
        
        delay_ms = 15  # Target ~60 FPS (1000ms / 60 ≈ 16ms, use 15ms for safety)
        
        if self._tk_owner is None:
            # No Tk owner means we can't schedule - stop the camera
            self.stop()
            return
        
        # Schedule next frame read
        self._after_id = self._tk_owner.after(delay_ms, self._read_one)

    def _read_one(self):
        """
        Read one frame from the camera and process it.
        
        This method is called by Tkinter's event scheduler. It:
        1. Reads a frame from the camera
        2. Applies mirroring if enabled
        3. Processes frame through frame_processor (e.g., object detection)
        4. Renders the frame via CanvasRenderer
        5. Calculates and reports FPS
        6. Schedules the next frame read
        
        If frame read fails or processing errors occur, the loop continues
        to avoid crashing the application.
        """
        if not self._running or self._cap is None:
            return

        t0 = time.time()
        ok, frame_bgr = self._cap.read()

        if ok:
            # === Apply Mirroring ===
            # Horizontally flip frame if mirror mode is enabled
            if self.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            
            # === Apply Frame Processing ===
            # This is where object detection/tracking happens
            # The processor can modify the frame (e.g., draw bounding boxes)
            if self.frame_processor is not None:
                try:
                    frame_bgr = self.frame_processor(frame_bgr)
                except Exception as e:
                    # Log error but continue processing to avoid crashing
                    print(f"Frame processor error: {e}")
                    # Optionally re-raise if you want to stop on errors
                    # raise
            
            # === Render Frame ===
            # Draw the (possibly processed) frame on the canvas
            if self.renderer:
                self.renderer.draw(frame_bgr)
        else:
            # Frame read failed (camera disconnected, etc.)
            # Continue loop - camera may reconnect
            pass

        # === Calculate and Report FPS ===
        if self.on_fps:
            dt = max(1e-6, time.time() - t0)  # Avoid division by zero
            fps = 1.0 / dt
            self.on_fps(fps)

        # === Schedule Next Frame ===
        self._schedule_next()
