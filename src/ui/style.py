# Central CSS adjacent theme + HUD utilities (this is pure Tkinter/ttk)

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

# Palette Constants
SEC_BG   = "#464646"
SEC_SURF = "#2b2b2b"
SEC_LINE = "#3a3a3a"
ACCENT   = "#00c2c7"
ACCENT_2 = "#15ffa8"
TEXT     = "#e6edf3"
MUTED    = "#9aa4b1"

# Theme State
_is_dark_mode = True

def font(primary=True, size=10, weight="normal"):
    # Font Selection
    family = "Segoe UI" if primary else "Calibri" 
    return (family, size, weight)

def apply_theme(root: tk.Tk):
    # Initial Theme Setup
    style = ThemedStyle(root)
    try:
        style.set_theme("equilux")
    except Exception:
        style.theme_use("clam")
    
    _configure_custom_styles(style, root)

def toggle_theme(root: tk.Tk):
    # Theme Toggle Logic
    global _is_dark_mode
    style = ThemedStyle(root)
    
    if _is_dark_mode:
        style.set_theme("arc") 
        _is_dark_mode = False
    else:
        style.set_theme("equilux") 
        _is_dark_mode = True
        
    _configure_custom_styles(style, root)
    return _is_dark_mode

def _configure_custom_styles(style, root):
    # Global Defaults
    style.configure(".", font=font(True, 10))
    
    # Button Style
    style.configure("TButton", padding=6, font=font(True, 10, "bold"))

    # Labelframe Style
    style.configure("TLabelframe", font=font(True, 10, "bold"))
    style.configure("TLabelframe.Label", font=font(True, 10, "bold"))

    # Header Style
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    
    # Status Bar Style
    style.configure("Status.TLabel", font=("Consolas", 9), background=style.lookup("TFrame", "background"))

    # Window Background
    bg_color = style.lookup("TFrame", "background")
    root.configure(bg=bg_color)

def style_menu(menu: tk.Menu):
    # Menu Styling
    menu.configure(
        tearoff=0,
        bg="#464646", fg="#e6edf3",
        activebackground="#5e5e5e", activeforeground="#ffffff",
        bd=0, relief="flat"
    )

def add_video_grid(parent: ttk.LabelFrame):
    # Grid Canvas Setup
    canvas = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    line_color = "#3a3a3a"

    def redraw_grid(_evt=None):
        # Grid Drawing Logic
        canvas.delete("grid")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        step = 40
        for x in range(0, w, step):
            canvas.create_line(x, 0, x, h, fill=line_color, tags="grid")
        for y in range(0, h, step):
            canvas.create_line(0, y, w, y, fill=line_color, tags="grid")
        cx, cy = w // 2, h // 2
        canvas.create_line(cx-25, cy, cx+25, cy, fill=ACCENT, width=2, tags="grid")
        canvas.create_line(cx, cy-25, cx, cy+25, fill=ACCENT, width=2, tags="grid")
        canvas.create_oval(cx-10, cy-10, cx+10, cy+10, outline=ACCENT, width=2, tags="grid")

    canvas.bind("<Configure>", redraw_grid)
    return canvas