#!/usr/bin/env python3
"""
Demo script for Audio Processing Application
Demonstrates all features with interactive examples
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import time
import os
from audio_processor import AudioProcessor
import matplotlib.pyplot as plt

class AudioDemo:
    def __init__(self):
        self.processor = AudioProcessor()
        self.sample_rate = 22050
        
    def create_demo_audio(self, genre='classical', duration=10):
        """Create demo audio for different genres"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if genre == 'classical':
            # Harmonic frequencies typical of classical music
            audio = (np.sin(2 * np.pi * 440 * t) +  # A4
                     0.5 * np.sin(2 * np.pi * 880 * t) +  # A5
                     0.3 * np.sin(2 * np.pi * 660 * t))  # E5
        elif genre == 'jazz':
            # Complex harmonies typical of jazz
            audio = (np.sin(2 * np.pi * 220 * t) +  # A3
                     0.7 * np.sin(2 * np.pi * 330 * t) +  # E4
                     0.4 * np.sin(2 * np.pi * 440 * t))  # A4
        elif genre == 'rock':
            # Distorted sound typical of rock
            audio = np.sin(2 * np.pi * 110 * t)  # A2
            audio = np.clip(audio * 1.5, -1, 1)  # Add distortion
        elif genre == 'pop':
            # Clean frequencies typical of pop
            audio = (np.sin(2 * np.pi * 330 * t) +  # E4
                     0.3 * np.sin(2 * np.pi * 660 * t) +  # E5
                     0.2 * np.sin(2 * np.pi * 990 * t))  # B5
        else:
            # Default: mixed frequencies
            audio = (np.sin(2 * np.pi * 100 * t) +  # Bass
                     0.5 * np.sin(2 * np.pi * 1000 * t) +  # Mid
                     0.3 * np.sin(2 * np.pi * 5000 * t))  # Treble
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, len(audio))
        audio = audio + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def demo_equalizer(self):
        """Demonstrate equalizer functionality"""
        print("\n🎵 DEMO: Equalizer")
        print("=" * 50)
        
        # Create demo audio
        audio = self.create_demo_audio('mixed')
        sf.write('demo_original.wav', audio, self.sample_rate)
        
        print("Original audio created. Playing...")
        sd.play(audio, self.sample_rate)
        sd.wait()
        
        # Test different equalizer settings
        settings = [
            (2.0, 1.0, 1.0, "Bass Boost"),
            (1.0, 2.0, 1.0, "Mid Boost"),
            (1.0, 1.0, 2.0, "Treble Boost"),
            (0.5, 0.5, 0.5, "All Reduced")
        ]
        
        for bass, mid, treble, name in settings:
            print(f"\nApplying {name} (Bass: {bass}, Mid: {mid}, Treble: {treble})")
            
            # Apply equalizer
            processed = self.processor.equalizer(audio, bass, mid, treble)
            
            # Save and play
            sf.write(f'demo_{name.lower().replace(" ", "_")}.wav', processed, self.sample_rate)
            
            print(f"Playing {name}...")
            sd.play(processed, self.sample_rate)
            sd.wait()
            
            time.sleep(1)  # Pause between examples
        
        print("\n✓ Equalizer demo completed")
    
    def demo_noise_reduction(self):
        """Demonstrate noise reduction functionality"""
        print("\n🔇 DEMO: Noise Reduction")
        print("=" * 50)
        
        # Create clean audio
        clean_audio = self.create_demo_audio('classical')
        
        # Add different types of noise
        noise_types = {
            'White Noise': np.random.normal(0, 0.2, len(clean_audio)),
            'Pink Noise': np.convolve(np.random.normal(0, 1, len(clean_audio)), 
                                    np.ones(100)/100, mode='same'),
            'Brown Noise': np.cumsum(np.random.normal(0, 0.01, len(clean_audio)))
        }
        
        print("Playing clean audio...")
        sd.play(clean_audio, self.sample_rate)
        sd.wait()
        
        for noise_name, noise in noise_types.items():
            print(f"\nAdding {noise_name}...")
            
            # Create noisy audio
            noisy_audio = clean_audio + noise * 0.3
            noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
            
            # Save noisy audio
            sf.write(f'demo_noisy_{noise_name.lower().replace(" ", "_")}.wav', 
                    noisy_audio, self.sample_rate)
            
            print(f"Playing audio with {noise_name}...")
            sd.play(noisy_audio, self.sample_rate)
            sd.wait()
            
            print(f"Applying noise reduction...")
            # Apply noise reduction
            denoised = self.processor.reduce_noise_ml(noisy_audio)
            
            # Save denoised audio
            sf.write(f'demo_denoised_{noise_name.lower().replace(" ", "_")}.wav', 
                    denoised, self.sample_rate)
            
            print(f"Playing denoised audio...")
            sd.play(denoised, self.sample_rate)
            sd.wait()
            
            # Calculate SNR improvement
            original_snr = 10 * np.log10(np.var(clean_audio) / np.var(noise))
            denoised_snr = 10 * np.log10(np.var(denoised) / np.var(noise))
            improvement = denoised_snr - original_snr
            
            print(f"SNR improvement: {improvement:.2f} dB")
            time.sleep(1)
        
        print("\n✓ Noise reduction demo completed")
    
    def demo_genre_classification(self):
        """Demonstrate genre classification functionality"""
        print("\n🎼 DEMO: Genre Classification")
        print("=" * 50)
        
        # Test different genres
        genres = ['classical', 'jazz', 'rock', 'pop']
        
        for genre in genres:
            print(f"\nCreating {genre} audio...")
            
            # Generate audio
            audio = self.create_demo_audio(genre)
            
            # Save audio
            sf.write(f'demo_{genre}.wav', audio, self.sample_rate)
            
            print(f"Playing {genre} audio...")
            sd.play(audio, self.sample_rate)
            sd.wait()
            
            # Classify
            predicted_genre, confidence = self.processor.classify_genre(audio)
            
            print(f"Predicted genre: {predicted_genre.title()}")
            print(f"Confidence: {confidence:.1%}")
            
            # Check if prediction is correct
            if predicted_genre == genre:
                print("✓ Correct classification!")
            else:
                print(f"✗ Incorrect classification (expected: {genre})")
            
            time.sleep(1)
        
        print("\n✓ Genre classification demo completed")
    
    def demo_real_time_processing(self):
        """Demonstrate real-time processing"""
        print("\n⏱️ DEMO: Real-time Processing")
        print("=" * 50)
        
        print("This demo will record audio from your microphone and process it in real-time.")
        print("Press Enter to start recording (5 seconds)...")
        input()
        
        # Record audio
        print("Recording... (speak or play music)")
        recording = sd.rec(int(5 * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        
        # Play back original
        print("Playing original recording...")
        sd.play(recording, self.sample_rate)
        sd.wait()
        
        # Process in real-time simulation
        print("Processing in real-time...")
        start_time = time.time()
        
        # Apply equalizer
        processed = self.processor.equalizer(recording.flatten())
        
        # Apply noise reduction
        denoised = self.processor.reduce_noise_ml(processed)
        
        # Classify
        genre, confidence = self.processor.classify_genre(denoised)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        print(f"Processing time: {processing_time:.2f} ms")
        print(f"Predicted genre: {genre.title()} (Confidence: {confidence:.1%})")
        
        # Play processed audio
        print("Playing processed audio...")
        sd.play(denoised, self.sample_rate)
        sd.wait()
        
        # Save results
        sf.write('demo_realtime_original.wav', recording, self.sample_rate)
        sf.write('demo_realtime_processed.wav', denoised, self.sample_rate)
        
        print("\n✓ Real-time processing demo completed")
    
    def demo_visualization(self):
        """Demonstrate audio visualization"""
        print("\n📊 DEMO: Audio Visualization")
        print("=" * 50)
        
        # Create demo audio
        original = self.create_demo_audio('classical')
        processed = self.processor.equalizer(original, 1.5, 0.8, 1.2)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time domain
        time_axis = np.linspace(0, len(original)/self.sample_rate, len(original))
        ax1.plot(time_axis[:22050], original[:22050], label='Original')
        ax1.set_title('Original Audio (Time Domain)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(time_axis[:22050], processed[:22050], label='Processed', color='orange')
        ax2.set_title('Processed Audio (Time Domain)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)
        
        # Frequency domain
        fft_original = np.abs(np.fft.fft(original))
        fft_processed = np.abs(np.fft.fft(processed))
        freq_axis = np.fft.fftfreq(len(original), 1/self.sample_rate)
        
        # Only show positive frequencies
        positive_freqs = freq_axis > 0
        ax3.plot(freq_axis[positive_freqs], fft_original[positive_freqs], label='Original')
        ax3.set_title('Original Audio (Frequency Domain)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xlim(0, 5000)  # Limit to 5kHz for better visualization
        
        ax4.plot(freq_axis[positive_freqs], fft_processed[positive_freqs], 
                label='Processed', color='orange')
        ax4.set_title('Processed Audio (Frequency Domain)')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(0, 5000)
        
        plt.tight_layout()
        plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'demo_visualization.png'")
        print("\n✓ Visualization demo completed")
    
    def demo_full_pipeline(self):
        """Demonstrate the complete processing pipeline"""
        print("\n🚀 DEMO: Complete Processing Pipeline")
        print("=" * 50)
        
        # Create demo audio
        print("Creating demo audio...")
        audio = self.create_demo_audio('jazz')
        sf.write('demo_pipeline_input.wav', audio, self.sample_rate)
        
        print("Playing original audio...")
        sd.play(audio, self.sample_rate)
        sd.wait()
        
        # Process through complete pipeline
        print("Processing through complete pipeline...")
        start_time = time.time()
        
        processed_audio, genre, confidence = self.processor.process_audio_file(
            'demo_pipeline_input.wav',
            bass_gain=1.5,
            mid_gain=0.8,
            treble_gain=1.2,
            denoise=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Predicted genre: {genre.title()}")
        print(f"Confidence: {confidence:.1%}")
        
        # Save and play processed audio
        self.processor.save_audio(processed_audio, 'demo_pipeline_output.wav')
        
        print("Playing processed audio...")
        sd.play(processed_audio, self.sample_rate)
        sd.wait()
        
        print("\n✓ Complete pipeline demo completed")
    
    def cleanup_demo_files(self):
        """Clean up demo files"""
        demo_files = [
            'demo_original.wav',
            'demo_bass_boost.wav',
            'demo_mid_boost.wav',
            'demo_treble_boost.wav',
            'demo_all_reduced.wav',
            'demo_noisy_white_noise.wav',
            'demo_noisy_pink_noise.wav',
            'demo_noisy_brown_noise.wav',
            'demo_denoised_white_noise.wav',
            'demo_denoised_pink_noise.wav',
            'demo_denoised_brown_noise.wav',
            'demo_classical.wav',
            'demo_jazz.wav',
            'demo_rock.wav',
            'demo_pop.wav',
            'demo_realtime_original.wav',
            'demo_realtime_processed.wav',
            'demo_pipeline_input.wav',
            'demo_pipeline_output.wav'
        ]
        
        for file in demo_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
    
    def run_all_demos(self):
        """Run all demos"""
        print("🎵 AUDIO PROCESSING APPLICATION DEMO")
        print("=" * 60)
        print("This demo will showcase all features of the application.")
        print("Make sure your speakers/headphones are connected and volume is up.")
        print("\nPress Enter to start...")
        input()
        
        try:
            # Run demos
            self.demo_equalizer()
            self.demo_noise_reduction()
            self.demo_genre_classification()
            self.demo_real_time_processing()
            self.demo_visualization()
            self.demo_full_pipeline()
            
            print("\n🎉 All demos completed successfully!")
            print("Check the generated files to see the results.")
            
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
        finally:
            # Ask if user wants to clean up
            print("\nClean up demo files? (y/n): ", end="")
            if input().lower() == 'y':
                self.cleanup_demo_files()
                print("✓ Cleanup completed")

def main():
    """Main demo function"""
    demo = AudioDemo()
    demo.run_all_demos()

if __name__ == "__main__":
    main() 