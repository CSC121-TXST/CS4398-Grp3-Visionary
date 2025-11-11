"""
Vision Narration Panel

This module provides a UI component for displaying the Vision Narrator's
human-friendly descriptions of what the camera sees.
"""

import tkinter as tk
from tkinter import ttk
from ui.style import ACCENT, ACCENT_2, MUTED, SEC_SURF


class VisionNarrationPanel(ttk.LabelFrame):
    """
    Panel that displays the Vision Narrator's descriptions.
    
    Shows a read-only text box with the current narration, and optionally
    a button to trigger narration on demand.
    """
    
    def __init__(self, parent, narrator=None, on_narrate_request=None):
        """
        Initialize the Vision Narration Panel.
        
        Args:
            parent: Parent widget
            narrator: VisionNarrator instance (optional)
            on_narrate_request: Callback function when user requests narration
        """
        super().__init__(parent, text="Vision Narration", padding=10)
        self.narrator = narrator
        self.on_narrate_request = on_narrate_request
        
        self._build()
        
        # Set up narration callback if narrator is provided
        if self.narrator:
            self.narrator.set_narration_callback(self.update_narration)
    
    def _build(self):
        """Build the UI components."""
        # Title/description
        desc_label = ttk.Label(
            self,
            text="Accessibility description of what the camera sees:",
            foreground=MUTED,
            font=("Segoe UI", 9)
        )
        desc_label.pack(anchor="w", pady=(0, 8))
        
        # Read-only text box for narration
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Text widget with scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.text_widget = tk.Text(
            text_frame,
            height=6,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=SEC_SURF,
            fg="#e6edf3",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#232a33",
            highlightcolor=ACCENT,
            yscrollcommand=scrollbar.set
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        # Default message
        self.update_narration("Waiting for narration...")
        
        # Button to trigger narration on demand
        if self.on_narrate_request:
            button_frame = ttk.Frame(self)
            button_frame.pack(fill="x")
            
            self.narrate_button = ttk.Button(
                button_frame,
                text="Narrate Now",
                command=self._on_narrate_clicked,
                style="Accent.TButton"
            )
            self.narrate_button.pack(side="left", padx=(0, 5))
            
            # Status label
            self.status_label = ttk.Label(
                button_frame,
                text="",
                foreground=MUTED,
                font=("Segoe UI", 8)
            )
            self.status_label.pack(side="left")
    
    def update_narration(self, narration: str):
        """
        Update the narration text display.
        
        Args:
            narration: The narration text to display
        """
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, narration)
        self.text_widget.config(state=tk.DISABLED)
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Updated")
            # Clear status after 2 seconds
            self.after(2000, lambda: self.status_label.config(text="") if hasattr(self, 'status_label') else None)
    
    def _on_narrate_clicked(self):
        """Handle narration button click."""
        if self.on_narrate_request:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Generating...", foreground=ACCENT_2)
            self.on_narrate_request()
    
    def set_narrator(self, narrator):
        """Set the narrator instance and configure callback."""
        self.narrator = narrator
        if self.narrator:
            self.narrator.set_narration_callback(self.update_narration)

