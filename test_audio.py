#!/usr/bin/env python3
"""
Test script for Audio Processing Application
"""

import numpy as np
import librosa
import soundfile as sf
import os
import time
from audio_processor import AudioProcessor

def create_test_audio(duration=10, sample_rate=22050):
    """Create test audio with different frequencies"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create complex audio with multiple frequencies
    audio = (np.sin(2 * np.pi * 100 * t) +  # Bass
             0.5 * np.sin(2 * np.pi * 1000 * t) +  # Mid
             0.3 * np.sin(2 * np.pi * 5000 * t))  # Treble
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def test_equalizer():
    """Test equalizer functionality"""
    print("Testing Equalizer...")
    
    # Create test audio
    audio = create_test_audio()
    
    # Save original
    sf.write('test_original.wav', audio, 22050)
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Test different equalizer settings
    settings = [
        (1.0, 1.0, 1.0, "Normal"),
        (2.0, 1.0, 1.0, "Bass Boost"),
        (1.0, 2.0, 1.0, "Mid Boost"),
        (1.0, 1.0, 2.0, "Treble Boost"),
        (0.5, 0.5, 0.5, "All Reduced")
    ]
    
    for bass, mid, treble, name in settings:
        print(f"  Testing {name} (Bass: {bass}, Mid: {mid}, Treble: {treble})")
        
        # Apply equalizer
        processed = processor.equalizer(audio, bass, mid, treble)
        
        # Save result
        sf.write(f'test_{name.lower().replace(" ", "_")}.wav', processed, 22050)
        
        # Check that audio is not corrupted
        assert not np.any(np.isnan(processed)), f"NaN values in {name}"
        assert not np.any(np.isinf(processed)), f"Inf values in {name}"
        
        print(f"    ✓ {name} test passed")
    
    print("✓ Equalizer tests completed\n")

def test_noise_reduction():
    """Test noise reduction functionality"""
    print("Testing Noise Reduction...")
    
    # Create clean audio
    clean_audio = create_test_audio()
    
    # Add different types of noise
    noise_types = {
        'white': np.random.normal(0, 0.2, len(clean_audio)),
        'pink': np.convolve(np.random.normal(0, 1, len(clean_audio)), 
                           np.ones(100)/100, mode='same'),
        'brown': np.cumsum(np.random.normal(0, 0.01, len(clean_audio)))
    }
    
    processor = AudioProcessor()
    
    for noise_type, noise in noise_types.items():
        print(f"  Testing {noise_type} noise reduction")
        
        # Create noisy audio
        noisy_audio = clean_audio + noise * 0.3
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
        
        # Save noisy audio
        sf.write(f'test_noisy_{noise_type}.wav', noisy_audio, 22050)
        
        # Apply noise reduction
        denoised = processor.reduce_noise_ml(noisy_audio)
        
        # Save denoised audio
        sf.write(f'test_denoised_{noise_type}.wav', denoised, 22050)
        
        # Calculate SNR improvement
        original_snr = 10 * np.log10(np.var(clean_audio) / np.var(noise))
        denoised_snr = 10 * np.log10(np.var(denoised) / np.var(noise))
        improvement = denoised_snr - original_snr
        
        print(f"    SNR improvement: {improvement:.2f} dB")
        
        # Check that audio is not corrupted
        assert not np.any(np.isnan(denoised)), f"NaN values in {noise_type} denoising"
        assert not np.any(np.isinf(denoised)), f"Inf values in {noise_type} denoising"
        
        print(f"    ✓ {noise_type} noise reduction test passed")
    
    print("✓ Noise reduction tests completed\n")

def test_genre_classification():
    """Test genre classification functionality"""
    print("Testing Genre Classification...")
    
    processor = AudioProcessor()
    
    # Create audio samples for different genres
    genres = {
        'classical': lambda t: np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t),
        'jazz': lambda t: np.sin(2 * np.pi * 220 * t) + 0.7 * np.sin(2 * np.pi * 330 * t),
        'rock': lambda t: np.clip(np.sin(2 * np.pi * 110 * t) * 1.5, -1, 1),
        'pop': lambda t: np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
    }
    
    duration = 5
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    for genre_name, audio_func in genres.items():
        print(f"  Testing {genre_name} classification")
        
        # Generate audio
        audio = audio_func(t)
        audio = audio / np.max(np.abs(audio))
        
        # Save test audio
        sf.write(f'test_{genre_name}.wav', audio, sample_rate)
        
        # Classify
        predicted_genre, confidence = processor.classify_genre(audio)
        
        print(f"    Predicted: {predicted_genre} (Confidence: {confidence:.1%})")
        
        # Check that classification works
        assert predicted_genre in ['blues', 'classical', 'country', 'disco', 'hiphop', 
                                 'jazz', 'metal', 'pop', 'reggae', 'rock'], \
               f"Invalid genre: {predicted_genre}"
        assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
        
        print(f"    ✓ {genre_name} classification test passed")
    
    print("✓ Genre classification tests completed\n")

def test_real_time_processing():
    """Test real-time processing functionality"""
    print("Testing Real-time Processing...")
    
    processor = AudioProcessor()
    
    # Create a short audio chunk
    chunk_duration = 0.1  # 100ms
    sample_rate = 22050
    t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
    audio_chunk = np.sin(2 * np.pi * 440 * t)
    
    # Test processing time
    start_time = time.time()
    
    # Apply equalizer
    processed_chunk = processor.equalizer(audio_chunk)
    
    # Apply noise reduction
    denoised_chunk = processor.reduce_noise_ml(processed_chunk)
    
    # Classify
    genre, confidence = processor.classify_genre(denoised_chunk)
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"  Processing time: {processing_time:.2f} ms")
    
    # Check that processing time is within limits
    assert processing_time < 500, f"Processing too slow: {processing_time} ms"
    
    print(f"  Predicted genre: {genre} (Confidence: {confidence:.1%})")
    print("  ✓ Real-time processing test passed")
    
    print("✓ Real-time processing tests completed\n")

def test_full_pipeline():
    """Test the complete audio processing pipeline"""
    print("Testing Full Pipeline...")
    
    # Create test audio
    audio = create_test_audio()
    sf.write('test_pipeline_input.wav', audio, 22050)
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Process through full pipeline
    processed_audio, genre, confidence = processor.process_audio_file(
        'test_pipeline_input.wav',
        bass_gain=1.5,
        mid_gain=0.8,
        treble_gain=1.2,
        denoise=True
    )
    
    # Save result
    processor.save_audio(processed_audio, 'test_pipeline_output.wav')
    
    # Check results
    assert len(processed_audio) > 0, "No output audio"
    assert not np.any(np.isnan(processed_audio)), "NaN values in output"
    assert not np.any(np.isinf(processed_audio)), "Inf values in output"
    assert genre in ['blues', 'classical', 'country', 'disco', 'hiphop', 
                    'jazz', 'metal', 'pop', 'reggae', 'rock'], f"Invalid genre: {genre}"
    assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
    
    print(f"  Genre: {genre} (Confidence: {confidence:.1%})")
    print("  ✓ Full pipeline test passed")
    
    print("✓ Full pipeline tests completed\n")

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        'test_original.wav',
        'test_normal.wav',
        'test_bass_boost.wav',
        'test_mid_boost.wav',
        'test_treble_boost.wav',
        'test_all_reduced.wav',
        'test_noisy_white.wav',
        'test_noisy_pink.wav',
        'test_noisy_brown.wav',
        'test_denoised_white.wav',
        'test_denoised_pink.wav',
        'test_denoised_brown.wav',
        'test_classical.wav',
        'test_jazz.wav',
        'test_rock.wav',
        'test_pop.wav',
        'test_pipeline_input.wav',
        'test_pipeline_output.wav'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

def main():
    """Run all tests"""
    print("Starting Audio Processing Application Tests\n")
    
    try:
        # Run tests
        test_equalizer()
        test_noise_reduction()
        test_genre_classification()
        test_real_time_processing()
        test_full_pipeline()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    finally:
        # Clean up
        print("\nCleaning up test files...")
        cleanup_test_files()
        print("✓ Cleanup completed")

if __name__ == "__main__":
    main() 