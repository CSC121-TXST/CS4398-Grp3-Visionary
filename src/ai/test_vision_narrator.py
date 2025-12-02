"""
Test script for Vision Narrator

This script tests the Vision Narrator without requiring a real API key.
Run this to verify the implementation works.
"""

from ai.vision_narrator import VisionNarrator

def test_format_detections():
    """Test detection formatting."""
    narrator = VisionNarrator(use_mock=True)
    
    # Sample detection data (similar to ObjectTracker output)
    tracked_objects = [
        {"id": 1, "cls": "person", "conf": 0.85, "bbox": (100, 150, 200, 300)},
        {"id": 2, "cls": "person", "conf": 0.92, "bbox": (400, 200, 500, 350)},
        {"id": 3, "cls": "laptop", "conf": 0.78, "bbox": (300, 400, 450, 500)},
        {"id": 4, "cls": "cell phone", "conf": 0.65, "bbox": (250, 300, 300, 350)},
    ]
    
    formatted = narrator.format_detections(tracked_objects)
    print(f"Formatted detections: {formatted}")
    assert "2 persons" in formatted or "2 person" in formatted
    assert "1 laptop" in formatted
    print("✓ Format test passed")
    
    return tracked_objects

def test_generate_description():
    """Test description generation (using mock)."""
    narrator = VisionNarrator(use_mock=True)
    
    tracked_objects = [
        {"id": 1, "cls": "person", "conf": 0.85, "bbox": (100, 150, 200, 300)},
        {"id": 2, "cls": "dog", "conf": 0.90, "bbox": (400, 200, 500, 350)},
    ]
    
    description = narrator.generate_description(tracked_objects, force=True)
    print(f"Generated description: {description}")
    assert description is not None
    assert len(description) > 0
    print("✓ Description generation test passed")

def test_empty_detections():
    """Test handling of empty detection list."""
    narrator = VisionNarrator(use_mock=True)
    
    description = narrator.generate_description([], force=True)
    print(f"Empty detection description: {description}")
    assert description is not None
    print("✓ Empty detection test passed")

if __name__ == "__main__":
    print("Testing Vision Narrator...")
    print("-" * 50)
    
    try:
        test_format_detections()
        test_generate_description()
        test_empty_detections()
        
        print("-" * 50)
        print("All tests passed! ✓")
        print("\nNext steps:")
        print("1. Set up OpenAI API key in .env file")
        print("2. Test with real API: narrator = VisionNarrator(use_mock=False)")
        print("3. Integrate into main_interface.py")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

