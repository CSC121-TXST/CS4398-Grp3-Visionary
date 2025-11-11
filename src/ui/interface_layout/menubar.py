# ui/interface_layout/menubar.py
import tkinter as tk
from tkinter import ttk
from ui.style import style_menu

def build_menubar(root, on_exit, on_about, on_toggle_debug=None, on_change_detection_classes=None):
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
    if on_toggle_debug is not None:
        # Attach a toggle for showing debugging
        if not hasattr(root, "_var_debug"):
            root._var_debug = tk.BooleanVar(value=False)
        settings_menu.add_checkbutton(
            label="Show Debugging",
            onvalue=True,
            offvalue=False,
            variable=root._var_debug,
            command=lambda: on_toggle_debug(root._var_debug.get())
        )

    # Detection Settings submenu (optional)
    if on_change_detection_classes is not None:
        det_menu = tk.Menu(settings_menu, tearoff=0)
        style_menu(det_menu)
        # Create vars on the root so they persist
        if not hasattr(root, "_var_cls_person"):
            root._var_cls_person = tk.BooleanVar(value=True)
        if not hasattr(root, "_var_cls_cell_phone"):
            root._var_cls_cell_phone = tk.BooleanVar(value=True)

        def _emit_detection_classes():
            classes = []
            if root._var_cls_person.get():
                classes.append("person")
            if root._var_cls_cell_phone.get():
                classes.append("cell phone")
            on_change_detection_classes(classes)

        det_menu.add_checkbutton(
            label="Person",
            onvalue=True,
            offvalue=False,
            variable=root._var_cls_person,
            command=_emit_detection_classes
        )
        det_menu.add_checkbutton(
            label="Cell Phone",
            onvalue=True,
            offvalue=False,
            variable=root._var_cls_cell_phone,
            command=_emit_detection_classes
        )
        settings_menu.add_cascade(label="Detection Settings", menu=det_menu)

    help_menu = make_menu_button(header, "Help")
    help_menu.add_command(label="About", command=on_about)

    return header
