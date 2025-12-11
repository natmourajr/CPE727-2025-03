# Test the installation
try:
    import ThreeWToolkit
    print(f"✅ 3WToolkit successfully imported!")
    print(f"📦 Version: {ThreeWToolkit.__version__}")
except ImportError as e:
    print(f"❌ Failed to import 3WToolkit: {e}")
    print("Please check your installation.")

# Test key imports
try:
    from ThreeWToolkit.dataset import ParquetDataset
    from ThreeWToolkit.preprocessing import ImputeMissing, Normalize
    from ThreeWToolkit.feature_extraction import ExtractStatisticalFeatures
    from ThreeWToolkit.models import SklearnModels
    
    print("✅ All key modules imported successfully!")
    print("📚 Available modules:")
    print("   - ParquetDataset (data loading)")
    print("   - ImputeMissing, Normalize (preprocessing)")
    print("   - ExtractStatisticalFeatures (feature extraction)")
    print("   - SklearnModels (machine learning)")
    
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Please check your installation.")

# Check Python version
import sys
print(f"🐍 Python version: {sys.version}")

if sys.version_info >= (3, 10):
    print("✅ Python version is compatible (3.10+)")
else:
    print("⚠️  Warning: Python 3.10+ is recommended")

# Check key dependencies
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import torch

print("📦 Key dependencies:")
print(f"   - NumPy: {np.__version__}")
print(f"   - Pandas: {pd.__version__}")
print(f"   - Scikit-learn: {sklearn.__version__}")
print(f"   - Matplotlib: {matplotlib.__version__}")
print(f"   - PyTorch: {torch.__version__}")

print("\n✅ All dependencies are available!")
