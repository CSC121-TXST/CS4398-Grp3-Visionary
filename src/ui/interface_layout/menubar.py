# ui/interface_layout/menubar.py
import tkinter as tk
from tkinter import ttk
from ui.style import style_menu

def build_menubar(root, on_exit, on_about):
    """Adds File/Settings/Help to the top of the window."""
    header = ttk.Frame(root, padding=(10, 6))
    header.pack(side=tk.TOP, fill="x")

    def make_menu_button(parent, text):
        mb = ttk.Menubutton(parent, text=text)
        menu = tk.Menu(mb, tearoff=0)
        style_menu(menu)
        mb["menu"] = menu
        mb.pack(side="left", padx=(0, 12))
        return menu

    file_menu = make_menu_button(header, "File")
    file_menu.add_command(label="Exit", command=on_exit)

    settings_menu = make_menu_button(header, "Settings")
    # add settings_menu.add_checkbutton(...) later, couldn't think of anything right now

    help_menu = make_menu_button(header, "Help")
    help_menu.add_command(label="About", command=on_about)

    return header
