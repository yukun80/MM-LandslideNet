#!/usr/bin/env python3

import sys
import os

# Add InternImage path to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
intern_image_path = os.path.join(os.path.dirname(current_dir), "InternImage", "classification")
sys.path.append(intern_image_path)
print(f"Added InternImage path: {intern_image_path}")
print(f"Path exists: {os.path.exists(intern_image_path)}")

try:
    print("Testing config import...")
    from config import config

    print("✅ Config import successful")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    print("Testing InternImage import...")
    from InternImage.classification.models.intern_image import InternImage

    print("✅ InternImage import successful")
except Exception as e:
    print(f"❌ InternImage import failed: {e}")

try:
    print("Testing models import...")
    from models import create_model

    print("✅ Models import successful")
except Exception as e:
    print(f"❌ Models import failed: {e}")

try:
    print("Testing dataset import...")
    from dataset import create_datasets

    print("✅ Dataset import successful")
except Exception as e:
    print(f"❌ Dataset import failed: {e}")

try:
    print("Testing utils import...")
    from utils import CombinedLoss

    print("✅ Utils import successful")
except Exception as e:
    print(f"❌ Utils import failed: {e}")

print("\nAll imports tested!")
