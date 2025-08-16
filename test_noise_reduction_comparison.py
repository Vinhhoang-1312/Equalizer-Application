#!/usr/bin/env python3
"""
Test script for new Noise Reduction comparison features
Kiểm tra tính năng so sánh noise reduction mới
"""

import numpy as np
import librosa
import soundfile as sf
import os
import sys
sys.path.append('.')

from modules.noise_reduction_engine import NoiseReductionEngine

def create_test_audio_with_noise():
    """Tạo file audio test với nhiễu để kiểm tra"""
    # Tạo clean audio signal
    duration = 3  # seconds
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal: combination of sine waves
    clean_signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.1 * np.sin(2 * np.pi * 1320 * t)   # E6 note
    )
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Normalize
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.8
    
    # Save test files
    test_dir = 'uploads'
    os.makedirs(test_dir, exist_ok=True)
    
    clean_path = os.path.join(test_dir, 'test_clean.wav')
    noisy_path = os.path.join(test_dir, 'test_noisy.wav')
    
    sf.write(clean_path, clean_signal, sample_rate)
    sf.write(noisy_path, noisy_signal, sample_rate)
    
    return clean_path, noisy_path, noisy_signal, sample_rate

def test_noise_reduction_comparison():
    """Test noise reduction comparison functionality"""
    print("🧪 Testing Noise Reduction Comparison Features...")
    
    # Create test audio
    clean_path, noisy_path, noisy_audio, sample_rate = create_test_audio_with_noise()
    print(f"✓ Created test audio files: {clean_path}, {noisy_path}")
    
    # Initialize noise reduction engine
    nr_engine = NoiseReductionEngine(sample_rate=sample_rate)
    print("✓ Noise Reduction Engine initialized")
    
    # Test different methods
    methods_to_test = ['noisereduce', 'spectral', 'wiener', 'adaptive']
    
    for method in methods_to_test:
        print(f"\n🔧 Testing method: {method}")
        
        try:
            # Apply noise reduction
            processed_audio = nr_engine.reduce_noise(
                noisy_audio, method=method, reduction_level=0.7
            )
            print(f"  ✓ Noise reduction applied with {method}")
            
            # Create comparison analysis
            comparison_analysis = nr_engine.create_comparison_analysis(
                noisy_audio, processed_audio, method, 0.7
            )
            print(f"  ✓ Comparison analysis created")
            
            # Print key metrics
            if 'comparison_metrics' in comparison_analysis:
                metrics = comparison_analysis['comparison_metrics']
                print(f"  📊 SNR Improvement: {metrics.get('snr_improvement_db', 0):.2f} dB")
                print(f"  📊 RMS Reduction: {metrics.get('rms_reduction_percent', 0):.1f}%")
            
            # Check technical explanation
            if 'technical_explanation' in comparison_analysis:
                tech_exp = comparison_analysis['technical_explanation']
                method_desc = tech_exp.get('method_description', {})
                print(f"  🔬 Method: {method_desc.get('name', 'Unknown')}")
                
                results = tech_exp.get('results_interpretation', {})
                quality = results.get('quality_assessment', 'Unknown')
                print(f"  📈 Quality Assessment: {quality}")
            
            # Check if chart was created
            if comparison_analysis.get('comparison_chart_path'):
                print(f"  📊 Comparison chart created: {comparison_analysis['comparison_chart_path']}")
            
            print(f"  ✅ {method} method test completed successfully")
            
        except Exception as e:
            print(f"  ❌ Error testing {method}: {e}")
    
    print("\n🎉 Noise Reduction Comparison Test Completed!")
    
    # Test autoencoder method if available
    print(f"\n🤖 Testing Autoencoder method...")
    try:
        processed_audio = nr_engine.reduce_noise(
            noisy_audio, method='autoencoder', reduction_level=0.7
        )
        
        comparison_analysis = nr_engine.create_comparison_analysis(
            noisy_audio, processed_audio, 'autoencoder', 0.7
        )
        
        print("✓ Autoencoder method works")
        
        if 'comparison_metrics' in comparison_analysis:
            metrics = comparison_analysis['comparison_metrics']
            print(f"📊 Autoencoder SNR Improvement: {metrics.get('snr_improvement_db', 0):.2f} dB")
        
    except Exception as e:
        print(f"⚠️ Autoencoder method issue (expected if no model): {e}")

def test_analysis_functions():
    """Test individual analysis functions"""
    print("\n🔬 Testing Individual Analysis Functions...")
    
    # Create simple test audio
    sample_rate = 22050
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 1, len(t))
    
    nr_engine = NoiseReductionEngine(sample_rate=sample_rate)
    
    # Test noise characteristics analysis
    try:
        noise_chars = nr_engine.analyze_noise_characteristics(test_audio)
        print("✓ analyze_noise_characteristics() works")
        print(f"  SNR: {noise_chars.get('snr_estimate', 'N/A'):.1f} dB")
        print(f"  RMS: {noise_chars.get('rms_level', 'N/A'):.4f}")
        print(f"  Recommended method: {noise_chars.get('recommended_method', 'N/A')}")
    except Exception as e:
        print(f"❌ analyze_noise_characteristics() error: {e}")
    
    # Test method recommendation
    try:
        methods = nr_engine.get_available_methods()
        print(f"✓ Available methods: {methods}")
    except Exception as e:
        print(f"❌ get_available_methods() error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Noise Reduction Comparison Tests...\n")
    
    # Test individual functions
    test_analysis_functions()
    
    # Test full comparison workflow
    test_noise_reduction_comparison()
    
    print("\n🎯 All tests completed! Check the results above.")
    print("\n💡 Next steps:")
    print("1. Run the web app: python main_modular.py")
    print("2. Upload an audio file")
    print("3. Go to Noise Reduction tab")
    print("4. Process audio and see detailed comparison")
    print("5. Listen to both audio files and check the detailed analysis")
