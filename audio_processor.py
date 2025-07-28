import librosa
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
from typing import Tuple, List, Optional
import sounddevice as sd
import threading
import queue

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.noise_reducer = None
        self.genre_classifier = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models for noise reduction and genre classification"""
        try:
            if os.path.exists('models/noise_reducer.h5'):
                self.noise_reducer = tf.keras.models.load_model('models/noise_reducer.h5')
            if os.path.exists('models/genre_classifier.pkl'):
                self.genre_classifier = joblib.load('models/genre_classifier.pkl')
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
        except:
            print("Models not found. Will train new models.")
    
    def equalizer(self, audio: np.ndarray, bass_gain: float = 1.0, 
                  mid_gain: float = 1.0, treble_gain: float = 1.0) -> np.ndarray:
        """
        Apply equalizer to audio with bass, mid, and treble controls
        
        Args:
            audio: Input audio array
            bass_gain: Gain for low frequencies (20-250 Hz)
            mid_gain: Gain for mid frequencies (250-4000 Hz)
            treble_gain: Gain for high frequencies (4000-20000 Hz)
        
        Returns:
            Processed audio array
        """
        # Convert to frequency domain
        fft_audio = fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Create frequency response
        freq_response = np.ones_like(freqs)
        
        # Apply bass filter (20-250 Hz)
        bass_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 250)
        freq_response[bass_mask] *= bass_gain
        
        # Apply mid filter (250-4000 Hz)
        mid_mask = (np.abs(freqs) >= 250) & (np.abs(freqs) <= 4000)
        freq_response[mid_mask] *= mid_gain
        
        # Apply treble filter (4000-20000 Hz)
        treble_mask = (np.abs(freqs) >= 4000) & (np.abs(freqs) <= 20000)
        freq_response[treble_mask] *= treble_gain
        
        # Apply frequency response
        processed_fft = fft_audio * freq_response
        
        # Convert back to time domain
        processed_audio = np.real(ifft(processed_fft))
        
        return processed_audio
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract audio features for genre classification
        
        Args:
            audio: Input audio array
        
        Returns:
            Feature vector
        """
        features = []
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend([np.mean(mfccs[i]) for i in range(13)])
        features.extend([np.std(mfccs[i]) for i in range(13)])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend([np.mean(chroma[i]) for i in range(12)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # Root mean square energy
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        return np.array(features)
    
    def reduce_noise_ml(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using machine learning approach
        
        Args:
            audio: Input audio with noise
        
        Returns:
            Denoised audio
        """
        if self.noise_reducer is None:
            # Simple Wiener filter if no ML model
            return self.wiener_filter(audio)
        
        # Prepare input for ML model
        # Convert to spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Normalize magnitude
        magnitude_norm = magnitude / np.max(magnitude)
        
        # Reshape for model input
        magnitude_reshaped = magnitude_norm.reshape(1, magnitude_norm.shape[0], 
                                                   magnitude_norm.shape[1], 1)
        
        # Predict clean magnitude
        clean_magnitude = self.noise_reducer.predict(magnitude_reshaped, verbose=0)
        clean_magnitude = clean_magnitude.reshape(magnitude_norm.shape)
        
        # Restore original scale
        clean_magnitude = clean_magnitude * np.max(magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple Wiener filter for noise reduction
        
        Args:
            audio: Input audio with noise
        
        Returns:
            Denoised audio
        """
        # Estimate noise from first 0.1 seconds
        noise_samples = int(0.1 * self.sample_rate)
        noise_estimate = np.mean(audio[:noise_samples]**2)
        
        # Apply Wiener filter
        signal_power = np.convolve(audio**2, np.ones(1000)/1000, mode='same')
        wiener_gain = signal_power / (signal_power + noise_estimate)
        
        # Apply gain
        denoised = audio * wiener_gain
        
        return denoised
    
    def classify_genre(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Classify music genre
        
        Args:
            audio: Input audio array
        
        Returns:
            Tuple of (genre, confidence)
        """
        if self.genre_classifier is None:
            return "Unknown", 0.0
        
        # Extract features
        features = self.extract_features(audio)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.genre_classifier.predict_proba(features.reshape(1, -1))[0]
        genre_idx = np.argmax(prediction)
        confidence = prediction[genre_idx]
        
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        return genres[genre_idx], confidence
    
    def process_audio_file(self, file_path: str, bass_gain: float = 1.0,
                          mid_gain: float = 1.0, treble_gain: float = 1.0,
                          denoise: bool = True) -> Tuple[np.ndarray, str, float]:
        """
        Process audio file with all features
        
        Args:
            file_path: Path to audio file
            bass_gain: Bass equalizer gain
            mid_gain: Mid equalizer gain
            treble_gain: Treble equalizer gain
            denoise: Whether to apply noise reduction
        
        Returns:
            Tuple of (processed_audio, genre, confidence)
        """
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Apply equalizer
        audio = self.equalizer(audio, bass_gain, mid_gain, treble_gain)
        
        # Apply noise reduction
        if denoise:
            audio = self.reduce_noise_ml(audio)
        
        # Classify genre
        genre, confidence = self.classify_genre(audio)
        
        return audio, genre, confidence
    
    def start_real_time_processing(self, callback):
        """
        Start real-time audio processing
        
        Args:
            callback: Function to call with processed audio data
        """
        self.is_recording = True
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                # Process audio chunk
                audio_chunk = indata[:, 0]  # Take first channel
                
                # Apply equalizer (default settings)
                processed_chunk = self.equalizer(audio_chunk)
                
                # Apply noise reduction
                processed_chunk = self.reduce_noise_ml(processed_chunk)
                
                # Send to callback
                callback(processed_chunk)
        
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=1024):
            while self.is_recording:
                time.sleep(0.1)
    
    def stop_real_time_processing(self):
        """Stop real-time audio processing"""
        self.is_recording = False
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio array to save
            file_path: Output file path
        """
        import soundfile as sf
        sf.write(file_path, audio, self.sample_rate) 