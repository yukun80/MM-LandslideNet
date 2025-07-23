"""
Helper module for importing InternImage to avoid namespace conflicts
"""

import sys
import os
import importlib.util


def import_intern_image():
    """Import InternImage from the InternImage repository"""

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to InternImage classification
    intern_image_path = os.path.join(os.path.dirname(current_dir), "InternImage", "classification")

    # Path to intern_image.py
    intern_image_file = os.path.join(intern_image_path, "models", "intern_image.py")

    # Check if file exists
    if not os.path.exists(intern_image_file):
        raise FileNotFoundError(f"InternImage file not found: {intern_image_file}")

    # Add to path
    if intern_image_path not in sys.path:
        sys.path.insert(0, intern_image_path)

    # Import using importlib
    spec = importlib.util.spec_from_file_location("intern_image_module", intern_image_file)
    intern_image_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intern_image_module)

    return intern_image_module.InternImage


# Lazy import - only import when requested
_InternImage = None


def get_intern_image():
    """Get InternImage class with lazy loading"""
    global _InternImage
    if _InternImage is None:
        try:
            _InternImage = import_intern_image()
            print("✅ InternImage imported successfully")
        except Exception as e:
            print(f"❌ Failed to import InternImage: {e}")
            raise
    return _InternImage


# For compatibility, export InternImage as a property
class InternImageProxy:
    """Proxy class that loads InternImage on first access"""

    def __new__(cls, *args, **kwargs):
        InternImageClass = get_intern_image()
        return InternImageClass(*args, **kwargs)


# Export the proxy as InternImage
InternImage = InternImageProxy
