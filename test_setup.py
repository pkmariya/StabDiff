"""
Test script to verify Stable Diffusion application setup
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__} installed")
    except ImportError as e:
        print(f"✗ Diffusers not installed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} installed")
    except ImportError as e:
        print(f"✗ Transformers not installed: {e}")
        return False
    
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__} installed")
    except ImportError as e:
        print(f"✗ Gradio not installed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow installed")
    except ImportError as e:
        print(f"✗ Pillow not installed: {e}")
        return False
    
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__} installed")
    except ImportError as e:
        print(f"✗ Accelerate not installed: {e}")
        return False
    
    return True

def test_model_access():
    """Test if model can be accessed"""
    print("\nTesting model access...")
    try:
        from diffusers import StableDiffusionPipeline
        print("✓ StableDiffusionPipeline can be imported")
        return True
    except Exception as e:
        print(f"✗ Error accessing StableDiffusionPipeline: {e}")
        return False

def test_app_import():
    """Test if the app can be imported"""
    print("\nTesting app module...")
    try:
        import app
        print("✓ App module can be imported")
        print("✓ StableDiffusionApp class available")
        return True
    except Exception as e:
        print(f"✗ Error importing app: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Stable Diffusion Application - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test model access
    results.append(("Model Access", test_model_access()))
    
    # Test app import
    results.append(("App Module", test_app_import()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the application.")
        print("\nRun the application with: python app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please install missing packages:")
        print("\n  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
