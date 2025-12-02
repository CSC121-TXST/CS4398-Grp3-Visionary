"""
Integration tests for UI components.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
@pytest.mark.ui
class TestUIIntegration:
    """Test UI component integration."""
    
    def test_status_bar_initialization(self):
        """Test StatusBar widget initialization."""
        root = tk.Tk()
        root.withdraw()  # Hide window during test
        
        from ui.interface_layout.status_bar import StatusBar
        status = StatusBar(root)
        
        assert status.var_laser is not None
        assert status.var_servo is not None
        assert status.var_status is not None
        assert status.var_fps is not None
        assert status.var_tracking is not None
        
        root.destroy()
    
    def test_narration_panel_initialization(self):
        """Test NarrationPanel widget initialization."""
        root = tk.Tk()
        root.withdraw()
        
        from ui.interface_layout.narration_panel import NarrationPanel
        panel = NarrationPanel(root)
        
        assert panel.text_area is not None
        assert panel.status_label is not None
        
        root.destroy()
    
    def test_narration_panel_update(self):
        """Test updating narration panel with text."""
        root = tk.Tk()
        root.withdraw()
        
        from ui.interface_layout.narration_panel import NarrationPanel
        panel = NarrationPanel(root)
        panel.update_narration("Test narration")
        
        # Check that text was added
        content = panel.text_area.get(1.0, tk.END)
        assert "Test narration" in content
        
        root.destroy()
    
    def test_menubar_build(self):
        """Test menubar building."""
        root = tk.Tk()
        root.withdraw()
        
        from ui.interface_layout.menubar import build_menubar
        
        def mock_exit():
            pass
        
        def mock_about():
            pass
        
        menubar = build_menubar(
            root,
            on_exit=mock_exit,
            on_about=mock_about
        )
        
        assert menubar is not None
        
        root.destroy()

