#!/usr/bin/env python3
"""
Test API Status
Kiểm tra API status của web app
"""

import requests
import json

def test_api_status():
    """Test API status endpoint"""
    try:
        print("🔍 Testing API status...")
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Status Response:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server on port 5000")
        print("💡 Make sure server is running: python web_app.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_api_status()
