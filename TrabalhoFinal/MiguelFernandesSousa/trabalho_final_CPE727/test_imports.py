"""
Quick test script to verify model imports work correctly
Run from: TrabalhoFinal/MiguelFernandesSousa/trabalho_final_CPE727/
"""

import sys
import os

print("="*60)
print("IARA Model Import Test")
print("="*60)

# Test 1: Check if timm is available (RECOMMENDED for modern models)
print("\n1. Testing timm library (for EfficientNet, ConvNeXt, ResNet)...")
try:
    import timm
    print(f"   ✅ timm version: {timm.__version__}")
    print(f"   Available models: {len(timm.list_models())} models")
    
    # Test creating a model
    import torch
    model = timm.create_model('resnet18', pretrained=False, num_classes=4, in_chans=1)
    x = torch.randn(2, 1, 224, 224)
    output = model(x)
    print(f"   ✅ ResNet18 test successful: input {x.shape} → output {output.shape}")
except ImportError as e:
    print(f"   ❌ timm not installed. Install with: pip install timm")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Check PyTorch
print("\n2. Testing PyTorch...")
try:
    import torch
    import torchvision
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ TorchVision version: {torchvision.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"   ❌ PyTorch not installed. Install with: pip install torch torchvision")

# Test 3: Check repository custom models
print("\n3. Testing repository custom models...")
repo_root = os.path.abspath('../../..')
seminars_path = os.path.join(repo_root, 'Seminarios/2 - Regularization/notebooks/curvas_aprendizado')
sys.path.insert(0, seminars_path)

try:
    from models.resnet import ResNet18CIFAR
    model = ResNet18CIFAR(num_classes=4, input_channels=1)
    print(f"   ✅ ResNet18CIFAR imported successfully")
    print(f"   Location: {seminars_path}/models/resnet.py")
except ImportError as e:
    print(f"   ⚠️  ResNet18CIFAR not found: {e}")
    print(f"   Check path: {seminars_path}/models/resnet.py")

try:
    from models.cnn import SimpleCNN
    model = SimpleCNN(num_classes=4, input_channels=1, input_size=256)
    print(f"   ✅ SimpleCNN imported successfully")
except ImportError as e:
    print(f"   ⚠️  SimpleCNN not found: {e}")

try:
    from models.mlp import MLP
    model = MLP(hidden_sizes=[512, 256], num_classes=4, input_channels=1, input_size=256)
    print(f"   ✅ Enhanced MLP imported successfully")
except ImportError as e:
    print(f"   ⚠️  Enhanced MLP not found: {e}")

# Test 4: Check IARA models
print("\n4. Testing IARA baseline models...")
iara_path = os.path.join(os.getcwd(), 'IARA/src')
sys.path.insert(0, iara_path)

try:
    import iara.ml.models.mlp as iara_mlp
    model = iara_mlp.MLP(input_shape=[1, 256, 256], hidden_channels=[256, 128], n_targets=4)
    print(f"   ✅ IARA MLP imported successfully")
    print(f"   Location: {iara_path}/iara/ml/models/mlp.py")
except ImportError as e:
    print(f"   ⚠️  IARA MLP not found: {e}")
    print(f"   Make sure IARA submodule is initialized:")
    print(f"   git submodule update --init --recursive")

try:
    import iara.ml.models.cnn as iara_cnn
    model = iara_cnn.CNN(input_shape=[1, 256, 256], conv_n_neurons=[128, 64], n_targets=4)
    print(f"   ✅ IARA CNN imported successfully")
except ImportError as e:
    print(f"   ⚠️  IARA CNN not found: {e}")

# Test 5: Check scientific libraries
print("\n5. Testing scientific libraries...")
required_libs = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('librosa', 'librosa'),
]

for lib_name, pip_name in required_libs:
    try:
        lib = __import__(lib_name)
        version = getattr(lib, '__version__', 'unknown')
        print(f"   ✅ {lib_name}: {version}")
    except ImportError:
        print(f"   ❌ {lib_name} not installed. Install with: pip install {pip_name}")

# Summary
print("\n" + "="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)
print("\n✅ RECOMMENDED APPROACH:")
print("   Use 'timm' library for modern models (ResNet, EfficientNet, ConvNeXt)")
print("   Install: pip install timm")
print("   Usage: model = timm.create_model('resnet18', pretrained=True, num_classes=4, in_chans=1)")
print("\n📖 For complete import guide, see:")
print("   MODEL_IMPORT_GUIDE.md")
print("\n🚀 To install all requirements:")
print("   pip install torch torchvision timm librosa scikit-learn pandas numpy matplotlib seaborn")
print("\n" + "="*60)
