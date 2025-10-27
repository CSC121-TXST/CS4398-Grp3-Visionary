# ui/interface_layout/title_bar.py
from tkinter import ttk
from ui.style import ACCENT

def build_title(root):
    """Places the centered 'Visionary' label."""
    title = ttk.Label(
        root,
        text="Visionary",
        font=("Consolas", 18, "bold"),
        foreground=ACCENT
    )
    title.pack(side="top", pady=(6, 6))
    return title