"""
Narration Panel UI Component

Displays LLM-generated descriptions of detected objects.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime


class NarrationPanel(ttk.LabelFrame):
    """
    Panel for displaying vision narration descriptions.
    """
    
    def __init__(self, parent):
        """
        Initialize narration panel.
        
        Args:
            parent: Parent Tkinter widget
        """
        super().__init__(parent, text="Vision Narration")
        
        # Create main text area for narration
        self.text_area = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            height=8,
            state=tk.DISABLED,
            font=("Arial", 10)
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="Ready",
            font=("Arial", 9)
        )
        self.status_label.pack(pady=(0, 5))
    
    def update_narration(self, description: str, timestamp: bool = True):
        """
        Update the narration display with a new description.
        
        Args:
            description: The narration text to display
            timestamp: Whether to include a timestamp
        """
        self.text_area.config(state=tk.NORMAL)
        
        if timestamp:
            time_str = datetime.now().strftime("%H:%M:%S")
            self.text_area.insert(tk.END, f"[{time_str}] ", "timestamp")
        
        self.text_area.insert(tk.END, f"{description}\n\n")
        self.text_area.see(tk.END)  # Auto-scroll to bottom
        self.text_area.config(state=tk.DISABLED)
    
    def update_status(self, status_text: str):
        """
        Update the status label.
        
        Args:
            status_text: Status message to display
        """
        self.status_label.config(text=status_text)
    
    def clear(self):
        """Clear all narration text."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")

