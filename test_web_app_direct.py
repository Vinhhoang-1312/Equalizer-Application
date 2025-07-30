#!/usr/bin/env python3
"""
Direct test of web app processing
"""

import os
import sys
sys.path.append('.')

from web_app import audio_processor
from werkzeug.utils import secure_filename

def test_web_app_direct():
    """Test the web app's audio processor directly"""
    print("Testing web app audio processor directly...")
    
    # Check if test file exists
    if not os.path.exists('test_audio.wav'):
        print("✗ Test audio file not found. Run create_test_audio.py first.")
        return
    
    # Simulate web app processing
    filename = secure_filename('test_audio.wav')
    filepath = os.path.join('uploads', filename)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Copy test file to uploads directory
    import shutil
    shutil.copy('test_audio.wav', filepath)
    
    print(f"Processing file: {filepath}")
    
    try:
        # Process audio using web app's audio processor
        results = audio_processor.process_audio_file_advanced(
            filepath, 
            equalizer_params={
                'bass_gain': 1.0,
                'mid_gain': 1.0,
                'treble_gain': 1.0,
                'sub_bass_gain': 1.0,
                'presence_gain': 1.0,
                'air_gain': 1.0
            },
            denoise_method='autoencoder',
            analyze=True
        )
        
        print(f"✓ Processing completed")
        print(f"Genre: {results['genre']}")
        print(f"Confidence: {results['confidence']}")
        print(f"Confidence type: {type(results['confidence'])}")
        
        # Test genre classification directly
        print("\nTesting genre classification directly...")
        import librosa
        audio, sr = librosa.load(filepath, sr=audio_processor.sample_rate)
        genre, confidence, info = audio_processor.advanced_genre_classification(audio)
        print(f"Direct genre: {genre}")
        print(f"Direct confidence: {confidence}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_web_app_direct() 