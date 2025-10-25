# Central CSS adjacent theme + HUD utilities (this is pure Tkinter/ttk)

import tkinter as tk
from tkinter import ttk

# ---- Palette
SEC_BG   = "#0f1115"   # app background
SEC_SURF = "#161a20"   # panels
SEC_LINE = "#232a33"   # borders
ACCENT   = "#00c2c7"   # primary accent
ACCENT_2 = "#15ffa8"   # active accent
TEXT     = "#e6edf3"   # main text
MUTED    = "#9aa4b1"   # secondary text

def font(primary=True, size=10, weight="normal"):
    family = "Consolas" if primary else "Segoe UI"
    return (family, size, weight)

def apply_theme(root: tk.Tk):
    """Apply dark security-console theme to ttk + window."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Global defaults
    style.configure(".",
        background=SEC_BG,
        foreground=TEXT,
        fieldbackground=SEC_SURF,
        highlightthickness=0
    )
    # Containers
    style.configure("TFrame", background=SEC_BG)
    style.configure("TLabelframe", background=SEC_BG, bordercolor=SEC_LINE)
    style.configure("TLabelframe.Label", background=SEC_BG, foreground=MUTED,
                    font=font(False, 10, "bold"))
    # Text
    style.configure("TLabel", background=SEC_BG, foreground=TEXT)
    # Buttons (accented)
    style.configure("Accent.TButton",
        background=SEC_SURF, foreground=TEXT, bordercolor=ACCENT,
        focusthickness=0, padding=8)
    style.map("Accent.TButton",
        background=[("active", "#1b212a")],
        bordercolor=[("active", ACCENT_2)])

    # Inputs
    style.configure("TCombobox", fieldbackground=SEC_SURF,
                    background=SEC_SURF, foreground=TEXT)
    style.configure("TScale", background=SEC_BG, troughcolor=SEC_SURF)

    # Menubuttons (top header menu bar)
    style.configure("TMenubutton",
        background=SEC_SURF,
        foreground=TEXT,
        padding=(10, 4),
        borderwidth=0
    )
    style.map("TMenubutton",
        background=[
            ("active", "#0f141a"),
            ("pressed", "#0d1116")
        ],
        foreground=[
            ("active", "#00e5ff"),   # bright cyan-blue text
            ("pressed", "#00c2c7")
        ]
    )

    # Window bg
    root.configure(bg=SEC_BG)

def style_menu(menu: tk.Menu):
    """Apply dark style to a tk.Menu instance."""
    menu.configure(
        tearoff=0,
        bg=SEC_SURF, fg=TEXT,
        activebackground="#1b212a", activeforeground=TEXT,
        bd=0, relief="flat"
    )

def add_video_grid(parent: ttk.LabelFrame):
    """
    This creates a canvas with a subtle grid + crosshair for the video area.
    Returns the canvas; We'll later just get rid of this once the Camera function works.
    """
    canvas = tk.Canvas(parent, bg=SEC_SURF, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    line_color = "#1f2630"

    def redraw_grid(_evt=None):
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
