"""
Scrollable Frame Utility

Provides a scrollable container for widgets that may overflow their available space.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame with a scrollbar that appears when content exceeds available space.
    Optimized to reduce flickering and improve scroll speed.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Debounce variables for scrollregion updates
        self._update_after_id = None
        self._last_height = 0
        
        # Configure scrollable frame with debounced update
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling (Windows uses <MouseWheel>, Linux uses <Button-4> and <Button-5>)
        # Use bind instead of bind_all to avoid conflicts with other widgets
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)
        # Also bind to scrollable frame for better coverage
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<Button-4>", self._on_mousewheel_linux)
        self.scrollable_frame.bind("<Button-5>", self._on_mousewheel_linux)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_frame_configure(self, event):
        """Debounced scrollregion update to reduce flickering."""
        # Only update if height actually changed
        if event.height != self._last_height:
            self._last_height = event.height
            # Cancel any pending update
            if self._update_after_id:
                self.after_cancel(self._update_after_id)
            # Schedule update after a short delay to batch multiple configure events
            self._update_after_id = self.after(10, self._update_scrollregion)
    
    def _update_scrollregion(self):
        """Update the canvas scrollregion."""
        self._update_after_id = None
        try:
            # Update scrollregion only if canvas is ready
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
        except tk.TclError:
            # Canvas might be destroyed, ignore
            pass
    
    def _on_canvas_configure(self, event):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        # Update scrollregion after width change
        self.after_idle(self._update_scrollregion)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling (Windows)."""
        # Faster scrolling: use 3 units per wheel notch instead of 1
        scroll_amount = int(-3 * (event.delta / 120))
        if scroll_amount != 0:
            self.canvas.yview_scroll(scroll_amount, "units")
    
    def _on_mousewheel_linux(self, event):
        """Handle mousewheel scrolling (Linux)."""
        # Faster scrolling: use 3 units per wheel notch
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
    
    def get_content_frame(self):
        """Return the frame where widgets should be placed."""
        return self.scrollable_frame

