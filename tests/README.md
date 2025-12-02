# Visionary Test Suite

Comprehensive testing package for all Visionary components.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_tracking.py     # ObjectTracker tests
│   ├── test_llm_integration.py  # LLM client tests
│   ├── test_vision_narrator.py  # Vision Narrator tests
│   ├── test_event_logger.py     # Event Logger tests
│   └── test_tts_engine.py       # TTS Engine tests
├── integration/             # Integration tests
│   ├── test_tracking_narration_integration.py
│   └── test_ui_integration.py
├── reports/                 # Generated test reports (gitignored)
│   ├── report.html          # HTML test report
│   ├── coverage/            # Coverage HTML report
│   ├── junit.xml            # JUnit XML format
│   └── test_results.json    # JSON results summary
└── run_tests.py             # Test runner script
```

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Vision components
pytest -m vision -v

# AI components
pytest -m ai -v

# UI components
pytest -m ui -v
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run Specific Test File
```bash
pytest tests/unit/test_tracking.py -v
```

### Run Specific Test
```bash
pytest tests/unit/test_tracking.py::TestObjectTracker::test_initialization -v
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.vision` - Vision/tracking tests
- `@pytest.mark.ai` - AI/LLM tests
- `@pytest.mark.ui` - UI component tests
- `@pytest.mark.hardware` - Hardware tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_api` - Tests requiring OpenAI API
- `@pytest.mark.requires_camera` - Tests requiring camera
- `@pytest.mark.requires_arduino` - Tests requiring Arduino

## Test Reports

After running tests, reports are generated in `tests/reports/`:

- **report.html** - Interactive HTML test report
- **coverage/index.html** - Code coverage report
- **junit.xml** - JUnit XML format (for CI/CD)
- **test_results.json** - JSON summary

## Test Coverage Goals

- **Unit Tests**: >80% coverage for core components
- **Integration Tests**: Cover main workflows
- **Critical Paths**: 100% coverage (tracking, narration, event logging)

## Writing New Tests

1. **Unit Tests**: Test individual functions/methods in isolation
2. **Integration Tests**: Test component interactions
3. **Use Fixtures**: Leverage `conftest.py` fixtures for common setup
4. **Mock External Dependencies**: Mock API calls, hardware, etc.
5. **Mark Tests**: Use appropriate markers for categorization

## Example Test

```python
@pytest.mark.unit
@pytest.mark.vision
class TestObjectTracker:
    def test_initialization(self):
        tracker = ObjectTracker(conf=0.25)
        assert tracker.conf == 0.25
```

## Continuous Integration

Tests can be run in CI/CD pipelines:

```bash
pytest tests/ --junitxml=junit.xml --cov=src --cov-report=xml
```

## Notes

- Some tests may be skipped if dependencies are missing (YOLO model, API keys)
- Hardware tests require actual hardware
- API tests require valid API keys (use mocks in CI)

