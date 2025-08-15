#!/usr/bin/env python3
"""
Test script để debug rock classification logic
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from modules.advanced_genre_classifier import AdvancedGenreClassifier
import librosa
import numpy as np

def test_audio_file(file_path, expected_genre="rock"):
    """Test classification for a single audio file"""
    print(f"\n🎵 Testing: {file_path}")
    print(f"🎯 Expected: {expected_genre}")
    print("-" * 50)
    
    # Load audio and extract features manually for debugging
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Extract same features as classifier
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroids)
        
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_mean = np.mean(rolloff)
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8)
        
        print(f"📊 Features:")
        print(f"   Spectral Centroid: {centroid_mean:.0f} Hz")
        print(f"   Spectral Rolloff:  {rolloff_mean:.0f} Hz")
        print(f"   Zero Crossing Rate: {zcr_mean:.3f}")
        print(f"   Tempo: {tempo:.0f} BPM")
        print(f"   Harmonic Ratio: {harmonic_ratio:.2f}")
        
        # Test with classifier
        classifier = AdvancedGenreClassifier()
        result = classifier.option1_musicnn_classify(file_path)
        
        print(f"\n🤖 Classification Result:")
        print(f"   Method: {result['method']}")
        print(f"   Predicted: {result['predicted_genre']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Status: {result['status']}")
        
        # Check if correct
        correct = result['predicted_genre'] == expected_genre
        print(f"   ✅ Correct: {correct}")
        
        return correct
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test multiple rock files"""
    rock_files = [
        "uploads/rock.00003.wav",
        "uploads/rock.00006.wav", 
        "uploads/rock.00008.wav"
    ]
    
    metal_files = [
        "uploads/metal.00003.wav",
        "uploads/metal.00006.wav"
    ]
    
    hiphop_files = [
        "uploads/hiphop.00008.wav",
        "uploads/hiphop.00009.wav"
    ]
    
    print("🎸 ROCK CLASSIFICATION TEST")
    print("=" * 60)
    
    correct_rock = 0
    for file_path in rock_files:
        if os.path.exists(file_path):
            if test_audio_file(file_path, "rock"):
                correct_rock += 1
    
    print(f"\n🎸 Rock Accuracy: {correct_rock}/{len(rock_files)} = {correct_rock/len(rock_files)*100:.1f}%")
    
    print("\n\n🔥 METAL CLASSIFICATION TEST")
    print("=" * 60)
    
    correct_metal = 0
    for file_path in metal_files:
        if os.path.exists(file_path):
            if test_audio_file(file_path, "metal"):
                correct_metal += 1
                
    print(f"\n🔥 Metal Accuracy: {correct_metal}/{len(metal_files)} = {correct_metal/len(metal_files)*100:.1f}%")
    
    print("\n\n🎤 HIPHOP CLASSIFICATION TEST") 
    print("=" * 60)
    
    correct_hiphop = 0
    for file_path in hiphop_files:
        if os.path.exists(file_path):
            if test_audio_file(file_path, "hiphop"):
                correct_hiphop += 1
                
    print(f"\n🎤 HipHop Accuracy: {correct_hiphop}/{len(hiphop_files)} = {correct_hiphop/len(hiphop_files)*100:.1f}%")

if __name__ == "__main__":
    main()
