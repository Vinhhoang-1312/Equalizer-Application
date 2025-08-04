#!/usr/bin/env python3
"""
Test Web App Directly
Kiểm tra web app trực tiếp không qua browser
"""

import requests
import json
import os

def test_web_app_direct():
    """Test web app endpoints directly"""
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Web App Directly")
    print("=" * 40)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Server is running (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        return False
    
    # Test 2: Check status endpoint
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Status endpoint working: {status_data}")
        else:
            print(f"⚠️ Status endpoint returned {response.status_code}")
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
    
    # Test 3: Test with a simple audio file
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"📁 Found test file: {test_file}")
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file, f, 'audio/wav')}
                data = {
                    'bass_gain': '1.0',
                    'mid_gain': '1.0', 
                    'treble_gain': '1.0',
                    'sub_bass_gain': '1.0',
                    'presence_gain': '1.0',
                    'air_gain': '1.0',
                    'denoise_method': 'wiener'
                }
                
                response = requests.post(f"{base_url}/upload", files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Upload successful: {result.get('genre', 'Unknown')}")
                    print(f"   Confidence: {result.get('confidence', 0):.2%}")
                else:
                    print(f"❌ Upload failed: {response.status_code}")
                    print(f"   Response: {response.text[:200]}")
                    
        except Exception as e:
            print(f"❌ Upload test error: {e}")
    else:
        print(f"⚠️ Test file not found: {test_file}")
    
    print("\n🎉 Web App Test Complete!")
    return True

if __name__ == "__main__":
    test_web_app_direct() 