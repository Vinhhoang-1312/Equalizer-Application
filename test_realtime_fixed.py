#!/usr/bin/env python3
"""
Test script cho Real-time Processing đã sửa
"""

import time
import requests
import json

def test_realtime_functionality():
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Real-time Processing (Fixed Version)")
    print("=" * 50)
    
    # 1. Test audio devices
    print("\n1. Testing audio devices...")
    try:
        response = requests.get(f"{base_url}/api/audio_devices")
        if response.status_code == 200:
            devices = response.json()
            print(f"✅ Found {len(devices.get('input_devices', []))} input devices")
            print(f"✅ Found {len(devices.get('output_devices', []))} output devices")
        else:
            print(f"❌ Failed to get devices: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting devices: {e}")
    
    # 2. Test real-time start
    print("\n2. Testing real-time start...")
    realtime_config = {
        "input_device": None,
        "output_device": None,
        "equalizer_params": {
            "sub_bass": 0, "bass": 0, "low_mid": 0, "mid": 0, "high_mid": 0,
            "presence": 0, "brilliance": 0, "air": 0, "ultra_high": 0, "extreme": 0
        },
        "noise_method": "noisereduce",
        "noise_reduction_level": 0.5,
        "enabled_modules": {
            "equalizer": True,
            "noise_reduction": True,
            "genre_classification": False
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/realtime/start",
            headers={"Content-Type": "application/json"},
            data=json.dumps(realtime_config)
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Real-time processing started successfully!")
            else:
                print(f"❌ Real-time start failed: {result.get('message')}")
        else:
            print(f"❌ Real-time start HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error starting real-time: {e}")
    
    # 3. Test stats
    print("\n3. Testing real-time stats...")
    try:
        time.sleep(2)  # Let it run for a bit
        response = requests.get(f"{base_url}/api/realtime/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats: Latency={stats.get('avg_latency', 0):.1f}ms, Chunks={stats.get('chunks_processed', 0)}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # 4. Test recording
    print("\n4. Testing recording...")
    try:
        record_config = {
            "filename": f"test_realtime_record_{int(time.time())}.wav",
            "duration": 3  # 3 seconds
        }
        
        response = requests.post(
            f"{base_url}/api/realtime/start_recording",
            headers={"Content-Type": "application/json"},
            data=json.dumps(record_config)
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Recording started: {result.get('filename')}")
                
                # Wait for recording to finish
                time.sleep(4)
                
                # Stop recording
                stop_response = requests.post(f"{base_url}/api/realtime/stop_recording")
                if stop_response.status_code == 200:
                    stop_result = stop_response.json()
                    if stop_result.get('success'):
                        print(f"✅ Recording completed: {stop_result.get('filename')}")
                    else:
                        print(f"❌ Recording stop failed: {stop_result.get('message')}")
                else:
                    print(f"❌ Recording stop HTTP error: {stop_response.status_code}")
            else:
                print(f"❌ Recording start failed: {result.get('message')}")
        else:
            print(f"❌ Recording start HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing recording: {e}")
    
    # 5. Test stop
    print("\n5. Testing real-time stop...")
    try:
        response = requests.post(f"{base_url}/api/realtime/stop")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Real-time processing stopped successfully!")
            else:
                print(f"❌ Real-time stop failed: {result.get('message')}")
        else:
            print(f"❌ Real-time stop HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error stopping real-time: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Real-time test completed!")
    print("\n💡 Tab Real-time bây giờ nên hoạt động độc lập:")
    print("   - Start/Stop Real-time Processing")
    print("   - Recording trong tab Real-time")
    print("   - Không ảnh hưởng đến các tab khác")

if __name__ == "__main__":
    test_realtime_functionality()
