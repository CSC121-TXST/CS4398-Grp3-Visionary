"""
Pytest configuration and shared fixtures for Visionary tests.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import cv2

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame (640x480, 3 channels)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_with_objects():
    """Create a sample frame with colored rectangles (simulating objects)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some rectangles to simulate objects
    cv2.rectangle(frame, (100, 100), (200, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(frame, (400, 150), (500, 250), (255, 0, 0), -1)  # Blue rectangle
    return frame


@pytest.fixture
def mock_tracked_objects():
    """Sample tracked objects list as returned by ObjectTracker."""
    return [
        {"id": 1, "cls": "person", "conf": 0.85, "bbox": (100, 150, 200, 300)},
        {"id": 2, "cls": "person", "conf": 0.92, "bbox": (400, 200, 500, 350)},
        {"id": 3, "cls": "cell phone", "conf": 0.78, "bbox": (300, 400, 450, 500)},
        {"id": 4, "cls": "book", "conf": 0.65, "bbox": (250, 300, 300, 350)},
    ]


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.names = {0: "person", 67: "cell phone", 73: "book", 16: "dog", 15: "cat", 24: "backpack"}
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    return mock_model


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I see two people and a cell phone in the frame."
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 200
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_env_file(temp_dir):
    """Create a mock .env file for testing."""
    env_file = Path(temp_dir) / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-key-12345\n")
    return str(env_file)


@pytest.fixture
def mock_arduino():
    """Mock Arduino controller for testing."""
    mock = MagicMock()
    mock.is_connected.return_value = True
    mock.send_command.return_value = True
    return mock


@pytest.fixture
def mock_camera():
    """Mock camera for testing."""
    mock = MagicMock()
    mock.is_running.return_value = True
    mock.start.return_value = True
    mock.stop.return_value = True
    return mock

