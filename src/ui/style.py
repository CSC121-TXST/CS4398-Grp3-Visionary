# Equilix theme

import tkinter as tk
from tkinter import ttk

# Colors
SEC_BG      = "#0b0f14"
SEC_SURF    = "#1e2630"
BUTTON_SURF = "#2a3441"
SEC_LINE    = "#2f3944"
ACCENT      = "#22d3ee"
ACCENT_2    = "#45ffc6"
TEXT        = "#e6edf3"
MUTED       = "#9aa4b1"

# Font
def font(primary=True, size=10, weight="normal"):
    families = ("Inter", "SF Pro Text", "Segoe UI", "Helvetica", "Arial")
    return (families[0], size, weight)

# Theme
DEFAULT_THEME = "equilux"

def apply_theme(root, theme_name: str | None = None):
    style = ttk.Style(root)
    theme = theme_name or DEFAULT_THEME

    if hasattr(root, "set_theme"):
        try:
            root.set_theme(theme)
        except:
            _use_builtin_theme(style)
    else:
        try:
            style.theme_use(theme)
        except:
            _use_builtin_theme(style)

    style.configure(".", background=SEC_BG, foreground=TEXT, fieldbackground=SEC_SURF)

    style.configure("TFrame", background=SEC_BG)
    style.configure("Title.TLabel",   font=font(True, 20, "bold"))
    style.configure("Section.TLabel", font=font(True, 12, "bold"))
    style.configure("TLabel",         background=SEC_BG, foreground=TEXT)
    style.configure("Muted.TLabel",   background=SEC_BG, foreground=MUTED)

    style.configure(
        "TButton",
        background=BUTTON_SURF,
        foreground=TEXT,
        relief="flat",
        padding=(10, 8),
        font=font(True, 11, "bold"),
    )
    style.map(
        "TButton",
        background=[("active", "#33404e"), ("pressed", "#1f2730")],
        foreground=[("disabled", MUTED)]
    )

    style.configure(
        "Accent.TButton",
        background=BUTTON_SURF,
        foreground=TEXT,
        relief="flat",
        padding=(10, 8),
        font=font(True, 11, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#34414f"), ("pressed", "#1d242b")],
        bordercolor=[("active", ACCENT_2)]
    )

    root.configure(bg=SEC_BG)

def _use_builtin_theme(style):
    for t in ("clam", "default", "vista", "xpnative"):
        try:
            style.theme_use(t)
            return
        except:
            pass

# Menu Style
def style_menu(menu: tk.Menu):
    menu.configure(tearoff=0, bg=SEC_SURF, fg=TEXT, activebackground="#2c3642", activeforeground=TEXT)
