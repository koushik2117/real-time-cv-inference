import sys
print("Python path:", sys.path)

try:
    from backend.api.models.inference import ModelInference
    print("✅ inference.py imported successfully")
except Exception as e:
    print("❌ Failed to import inference.py:", e)

try:
    from backend.api.core.preprocessor import ImagePreprocessor
    print("✅ preprocessor.py imported successfully")
except Exception as e:
    print("❌ Failed to import preprocessor.py:", e)

print("\nTest complete!")