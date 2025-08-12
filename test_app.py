#!/usr/bin/env python3
"""
Test script for Audio Processing Application
Kiểm tra toàn bộ functionality của ứng dụng
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def test_imports():
    """Test import các modules"""
    print("🔍 Testing imports...")
    
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        print("✅ AdvancedAudioProcessor imported")
    except Exception as e:
        print(f"❌ Failed to import AdvancedAudioProcessor: {e}")
        return False
    
    try:
        from web_app import app
        print("✅ Web app imported")
    except Exception as e:
        print(f"❌ Failed to import web app: {e}")
        return False
    
    try:
        from spotify_integration import SpotifyIntegration
        print("✅ Spotify integration imported")
    except Exception as e:
        print(f"❌ Failed to import Spotify integration: {e}")
        return False
    
    return True

def test_audio_processor():
    """Test audio processor"""
    print("\n🎵 Testing audio processor...")
    
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        
        processor = AdvancedAudioProcessor()
        print("✅ Audio processor initialized")
        
        # Test với file test nếu có
        test_file = "test_audio.wav"
        if os.path.exists(test_file):
            print(f"🎵 Testing with {test_file}...")
            
            # Test basic processing
            results = processor.process_audio_file_advanced(
                test_file,
                equalizer_params={
                    'bass_gain': 1.2,
                    'mid_gain': 0.9,
                    'treble_gain': 1.1,
                    'sub_bass_gain': 1.0,
                    'presence_gain': 1.0,
                    'air_gain': 1.0
                },
                denoise_method='spectral_subtraction',
                analyze=True
            )
            
            print(f"✅ Processing completed")
            print(f"   Genre: {results['genre']}")
            print(f"   Confidence: {results['confidence']:.1%}")
            print(f"   Tempo: {results['analysis'].get('tempo', 'N/A')} BPM")
            
            # Test visualization
            viz_path = "static/results/test_analysis.png"
            processor.create_visualization(results, viz_path)
            print(f"✅ Visualization created: {viz_path}")
            
        else:
            print("⚠️ No test audio file found, skipping audio processing test")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False

def test_web_server():
    """Test web server"""
    print("\n🌐 Testing web server...")
    
    try:
        # Test server status
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Web server is running")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Models loaded: {data.get('models_loaded', {})}")
            return True
        else:
            print(f"❌ Web server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Web server not running on port 5000")
        print("💡 Start server with: python web_app.py")
        return False
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n📁 Testing file upload...")
    
    test_file = "test_audio.wav"
    if not os.path.exists(test_file):
        print("⚠️ No test file found, skipping upload test")
        return True
    
    try:
        # Test upload
        with open(test_file, 'rb') as f:
            files = {'file': f}
            data = {
                'bass_gain': '1.2',
                'mid_gain': '0.9',
                'treble_gain': '1.1',
                'sub_bass_gain': '1.0',
                'presence_gain': '1.0',
                'air_gain': '1.0',
                'denoise_method': 'spectral_subtraction'
            }
            
            response = requests.post(
                "http://localhost:5000/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload successful")
            print(f"   Genre: {result.get('genre', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', '0%')}")
            print(f"   Processed file: {result.get('processed_file', 'None')}")
            return True
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False

def test_directories():
    """Test required directories"""
    print("\n📂 Testing directories...")
    
    required_dirs = [
        'uploads',
        'static/results',
        'data',
        'models',
        'templates'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            all_good = False
    
    return all_good

def main():
    """Main test function"""
    print("🧪 AUDIO PROCESSING APP TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("Audio Processor Test", test_audio_processor),
        ("Web Server Test", test_web_server),
        ("File Upload Test", test_file_upload)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! App is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Open http://localhost:5000 in your browser")
        print("   2. Upload an audio file to test processing")
        print("   3. Try real-time processing")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all dependencies are installed")
        print("   2. Check if web server is running")
        print("   3. Verify file permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


