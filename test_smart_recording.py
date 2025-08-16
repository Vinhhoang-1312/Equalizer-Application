#!/usr/bin/env python3
"""
Test Script: Demo Smart Recording Feature
"""

import requests
import time
import json

def test_smart_recording():
    base_url = "http://127.0.0.1:5000"
    
    print("🎙️ Testing Smart Recording Feature...")
    print("=" * 50)
    
    # Test 1: Check if real-time is initially off
    print("1. Checking initial real-time status...")
    try:
        response = requests.get(f"{base_url}/api/realtime/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   Real-time active: {data.get('is_processing', False)}")
        else:
            print("   Real-time not active (expected)")
    except:
        print("   Real-time not active (expected)")
    
    # Test 2: Start recording (should auto-start real-time)
    print("\n2. Starting recording (should auto-start real-time)...")
    try:
        response = requests.post(f"{base_url}/api/realtime/start_recording", 
                               json={"filename": "test_auto_recording.wav", "duration": 5})
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Recording started: {data.get('filename')}")
            print(f"   Duration: {data.get('duration')} seconds")
        else:
            print(f"   ❌ Failed to start recording: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Check real-time status after recording start
    print("\n3. Checking real-time status after recording...")
    try:
        response = requests.get(f"{base_url}/api/realtime/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   Real-time active: {data.get('is_processing', False)}")
            print(f"   Sample rate: {data.get('sample_rate')} Hz")
            print(f"   Chunk size: {data.get('chunk_size')}")
        else:
            print("   Could not get real-time status")
    except Exception as e:
        print(f"   Error getting status: {e}")
    
    # Wait for recording to finish (5 seconds)
    print("\n4. Waiting for 5-second recording to complete...")
    time.sleep(6)
    
    # Test 4: Stop recording
    print("\n5. Stopping recording...")
    try:
        response = requests.post(f"{base_url}/api/realtime/stop_recording")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Recording stopped: {data.get('filename')}")
            print(f"   Final duration: {data.get('duration', 0):.2f} seconds")
        else:
            print(f"   Recording may have auto-stopped: {response.text}")
    except Exception as e:
        print(f"   Note: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Smart Recording Test Complete!")
    print("\nNow you can:")
    print("1. Open web app at http://127.0.0.1:5000")
    print("2. Go to Upload tab")
    print("3. Click 'Start Recording' - Real-time akan auto-start!")
    print("4. No need manual setup di Real-time tab lagi!")

if __name__ == "__main__":
    test_smart_recording()
