#!/usr/bin/env python3
"""
Quick Test and Setup Script for Modular Audio Processing Application
Kiểm tra và thiết lập nhanh ứng dụng xử lý âm thanh modular
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_socketio', 'librosa', 'soundfile', 
        'sklearn', 'tensorflow', 'scipy', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'flask_socketio':
                import flask_socketio
            else:
                importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static/results', 'static/js', 'models', 'data']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory created/exists: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")

def install_missing_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n🔧 Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', 'requirements_modular.txt'
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_audio_functionality():
    """Test basic audio functionality"""
    try:
        import librosa
        import numpy as np
        
        # Create test audio signal
        sr = 22050
        duration = 1  # 1 second
        t = np.linspace(0, duration, sr * duration)
        test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test basic librosa functions
        mfcc = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=test_audio, sr=sr)
        
        print("✅ Audio processing functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def test_ml_functionality():
    """Test machine learning functionality"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Test basic ML functionality
        X = np.random.random((100, 10))
        y = np.random.randint(0, 5, 100)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(X_scaled, y)
        
        print("✅ Machine Learning functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Machine Learning test failed: {e}")
        return False

def test_web_framework():
    """Test Flask functionality"""
    try:
        from flask import Flask
        from flask_socketio import SocketIO
        
        app = Flask(__name__)
        socketio = SocketIO(app)
        
        print("✅ Web framework functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Web framework test failed: {e}")
        return False

def check_modules_import():
    """Check if our custom modules can be imported"""
    try:
        sys.path.append('.')
        
        from modules.equalizer_engine import EqualizerEngine
        from modules.noise_reduction_engine import NoiseReductionEngine
        from modules.genre_classification_engine import GenreClassificationEngine
        from modules.realtime_processing_engine import RealTimeProcessingEngine
        
        print("✅ All custom modules can be imported")
        return True
        
    except Exception as e:
        print(f"❌ Module import test failed: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    try:
        from modules.equalizer_engine import EqualizerEngine
        import numpy as np
        
        # Test equalizer
        eq = EqualizerEngine()
        test_audio = np.random.random(1024)
        gains = {'bass': 2.0, 'mid': -1.0, 'treble': 3.0}
        
        processed = eq.apply_equalizer_fft(test_audio, gains)
        
        print("✅ Quick functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quick functionality test failed: {e}")
        return False

def main():
    """Main setup and test function"""
    print("🚀 Advanced Audio Processing - Setup and Test")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check required packages
    print("\n📦 Checking required packages...")
    missing_packages = check_required_packages()
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected: {missing_packages}")
        
        install = input("Install missing packages? (y/N): ").lower().strip()
        if install == 'y':
            if not install_missing_packages(missing_packages):
                print("\n❌ Setup failed: Package installation failed")
                return False
        else:
            print("\n⚠️  Some packages are missing. Application may not work properly.")
    
    # Test functionality
    print("\n🧪 Testing functionality...")
    
    tests = [
        ("Audio Processing", test_audio_functionality),
        ("Machine Learning", test_ml_functionality), 
        ("Web Framework", test_web_framework),
        ("Custom Modules", check_modules_import),
        ("Quick Test", run_quick_test)
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if not test_func():
            failed_tests.append(test_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    if not failed_tests:
        print("🎉 All tests passed! Application is ready to run.")
        print("\n🚀 To start the application:")
        print("   python main_modular.py")
        print("\n🌐 Then open browser to:")
        print("   http://localhost:5000")
        return True
    else:
        print(f"⚠️  {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        print("\n🔧 Try installing missing dependencies:")
        print("   pip install -r requirements_modular.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
