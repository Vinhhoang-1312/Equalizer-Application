#!/usr/bin/env python3
"""
Noise Reduction Engine Module
Giảm nhiễu bằng Machine Learning và Deep Learning với nhiều phương pháp
"""

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import UpSampling1D, UpSampling2D, Input, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import scipy.signal
import noisereduce as nr
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class NoiseReductionEngine:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Noise Reduction Engine
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.autoencoder_model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        # Parameters for different methods
        self.wiener_params = {
            'noise_estimate_seconds': 0.5,
            'filter_order': 5
        }
        
        self.spectral_params = {
            'alpha': 2.0,
            'beta': 0.01,
            'window_size': 2048,
            'hop_length': 512
        }
        
        # Try to load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained noise reduction models"""
        try:
            # Try to load autoencoder model
            model_path = 'models/advanced_noise_reducer.h5'
            if tf.io.gfile.exists(model_path):
                self.autoencoder_model = tf.keras.models.load_model(model_path)
                self.is_model_trained = True
                print("✓ Noise reduction model loaded successfully")
            else:
                print("! Noise reduction model not found, will create new one")
                self._build_autoencoder_model()
        except Exception as e:
            print(f"⚠️ Error loading noise reduction model: {e}")
            self._build_autoencoder_model()
    
    def _build_autoencoder_model(self):
        """Build and compile autoencoder model for noise reduction"""
        try:
            # Simple 1D CNN Autoencoder for real-time processing
            model = Sequential([
                # Encoder
                Conv1D(32, 3, activation='relu', padding='same', input_shape=(None, 1)),
                MaxPooling1D(2, padding='same'),
                Conv1D(16, 3, activation='relu', padding='same'),
                MaxPooling1D(2, padding='same'),
                Conv1D(8, 3, activation='relu', padding='same'),
                
                # Decoder
                Conv1D(8, 3, activation='relu', padding='same'),
                UpSampling1D(2),
                Conv1D(16, 3, activation='relu', padding='same'),
                UpSampling1D(2),
                Conv1D(32, 3, activation='relu', padding='same'),
                Conv1D(1, 3, activation='tanh', padding='same')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mse', 
                         metrics=['mae'])
            
            self.autoencoder_model = model
            print("✓ Autoencoder model built successfully")
            
        except Exception as e:
            print(f"⚠️ Error building autoencoder model: {e}")
            self.autoencoder_model = None
    
    def _build_spectrogram_autoencoder(self, input_shape):
        """Build 2D CNN autoencoder for spectrogram processing"""
        try:
            input_layer = Input(shape=input_shape)
            
            # Encoder
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)
            
            # Decoder
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            
            model = Model(input_layer, decoded)
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mse',
                         metrics=['mae'])
            
            return model
            
        except Exception as e:
            print(f"⚠️ Error building spectrogram autoencoder: {e}")
            return None
    
    def noisereduce_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        """
        Noise reduction using noisereduce library
        
        Args:
            audio: Input audio signal
            reduction_level: Strength of noise reduction (0.0 to 1.0)
            
        Returns:
            Denoised audio signal
        """
        try:
            # Basic noise reduction with noisereduce
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                prop_decrease=reduction_level,
                stationary=False,
                n_std_thresh_stationary=1.5,
                n_thresh_nonstationary=0.5
            )
            return reduced_noise.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Noisereduce method failed: {e}")
            return audio
    
    def wiener_filter_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        """
        Wiener filter based noise reduction
        
        Args:
            audio: Input audio signal  
            reduction_level: Filter strength (0.0 to 1.0)
            
        Returns:
            Filtered audio signal
        """
        try:
            # Estimate noise from first portion of audio
            noise_samples = int(self.wiener_params['noise_estimate_seconds'] * self.sample_rate)
            noise_segment = audio[:min(noise_samples, len(audio)//4)]
            
            # Compute power spectral densities
            f, psd_signal = scipy.signal.welch(audio, self.sample_rate, nperseg=1024)
            f, psd_noise = scipy.signal.welch(noise_segment, self.sample_rate, nperseg=1024)
            
            # Wiener filter transfer function
            H = psd_signal / (psd_signal + psd_noise * (1/reduction_level - 1))
            
            # Apply filter in frequency domain
            audio_fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            
            # Interpolate filter response to match FFT bins
            H_interp = np.interp(np.abs(freqs), f, H)
            
            # Apply filter
            filtered_fft = audio_fft * H_interp
            filtered_audio = np.real(np.fft.ifft(filtered_fft))
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Wiener filter method failed: {e}")
            return audio
    
    def spectral_subtraction_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        """
        Spectral subtraction noise reduction
        
        Args:
            audio: Input audio signal
            reduction_level: Subtraction strength (0.0 to 1.0)
            
        Returns:
            Denoised audio signal
        """
        try:
            window_size = self.spectral_params['window_size']
            hop_length = self.spectral_params['hop_length']
            alpha = self.spectral_params['alpha'] * reduction_level
            beta = self.spectral_params['beta']
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum from first few frames
            noise_frames = magnitude[:, :5]  # First 5 frames
            noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Apply spectral floor (beta)
            enhanced_magnitude = np.maximum(enhanced_magnitude, 
                                          beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Spectral subtraction method failed: {e}")
            return audio
    
    def autoencoder_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        """
        Autoencoder-based noise reduction
        
        Args:
            audio: Input audio signal
            reduction_level: Processing strength (0.0 to 1.0)
            
        Returns:
            Denoised audio signal
        """
        try:
            if self.autoencoder_model is None:
                print("⚠️ Autoencoder model not available, falling back to spectral subtraction")
                return self.spectral_subtraction_method(audio, reduction_level)
            
            # Prepare audio for model (reshape for Conv1D)
            original_length = len(audio)
            
            # Pad to make divisible by model requirements
            pad_length = 0
            if len(audio) % 4 != 0:
                pad_length = 4 - (len(audio) % 4)
                audio_padded = np.pad(audio, (0, pad_length), mode='constant')
            else:
                audio_padded = audio
            
            # Reshape for Conv1D input
            audio_input = audio_padded.reshape(-1, len(audio_padded), 1)
            
            # Predict with autoencoder
            denoised = self.autoencoder_model.predict(audio_input, verbose=0)
            denoised_audio = denoised.reshape(-1)
            
            # Remove padding
            if pad_length > 0:
                denoised_audio = denoised_audio[:original_length]
            
            # Blend with original based on reduction level
            blended_audio = (1 - reduction_level) * audio + reduction_level * denoised_audio[:len(audio)]
            
            return blended_audio.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Autoencoder method failed: {e}")
            return self.spectral_subtraction_method(audio, reduction_level)
    
    def adaptive_filter_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        """
        Adaptive filtering noise reduction using LMS algorithm
        
        Args:
            audio: Input audio signal
            reduction_level: Adaptation strength (0.0 to 1.0)
            
        Returns:
            Filtered audio signal
        """
        try:
            # Simple LMS adaptive filter
            filter_length = 32
            mu = 0.01 * reduction_level  # Step size
            
            # Initialize filter weights
            w = np.zeros(filter_length)
            y = np.zeros_like(audio)
            
            # Apply adaptive filter
            for n in range(filter_length, len(audio)):
                x = audio[n-filter_length:n]
                y[n] = np.dot(w, x[::-1])  # Filter output
                e = audio[n] - y[n]  # Error signal
                w += mu * e * x[::-1]  # Update weights
            
            # The error signal is our denoised output
            denoised = audio - y
            
            return denoised.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Adaptive filter method failed: {e}")
            return audio
    
    def reduce_noise(self, audio: np.ndarray, method: str = 'autoencoder', 
                    reduction_level: float = 0.7) -> np.ndarray:
        """
        Apply noise reduction using specified method
        
        Args:
            audio: Input audio signal
            method: Noise reduction method
            reduction_level: Strength of noise reduction (0.0 to 1.0)
            
        Returns:
            Denoised audio signal
        """
        # Clamp reduction level
        reduction_level = np.clip(reduction_level, 0.0, 1.0)
        
        if method == 'noisereduce':
            return self.noisereduce_method(audio, reduction_level)
        elif method == 'wiener':
            return self.wiener_filter_method(audio, reduction_level)
        elif method == 'spectral':
            return self.spectral_subtraction_method(audio, reduction_level)
        elif method == 'autoencoder':
            return self.autoencoder_method(audio, reduction_level)
        elif method == 'adaptive':
            return self.adaptive_filter_method(audio, reduction_level)
        else:
            print(f"⚠️ Unknown method '{method}', using autoencoder")
            return self.autoencoder_method(audio, reduction_level)
    
    def process_audio_file(self, input_path: str, output_path: str, 
                          method: str = 'autoencoder',
                          reduction_level: float = 0.7) -> Dict:
        """
        Process audio file for noise reduction
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            method: Noise reduction method
            reduction_level: Strength of noise reduction
            
        Returns:
            Dictionary with processing results
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Calculate original noise metrics
        original_rms = np.sqrt(np.mean(audio ** 2))
        original_snr = self._estimate_snr(audio)
        
        # Apply noise reduction
        denoised_audio = self.reduce_noise(audio, method, reduction_level)
        
        # Calculate processed noise metrics
        processed_rms = np.sqrt(np.mean(denoised_audio ** 2))
        processed_snr = self._estimate_snr(denoised_audio)
        
        # Save processed audio
        sf.write(output_path, denoised_audio, self.sample_rate)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'method': method,
            'reduction_level': reduction_level,
            'original_rms': float(original_rms),
            'processed_rms': float(processed_rms),
            'original_snr': float(original_snr),
            'processed_snr': float(processed_snr),
            'snr_improvement': float(processed_snr - original_snr),
            'sample_rate': self.sample_rate,
            'duration': len(denoised_audio) / self.sample_rate,
            'noise_reduction_db': float(20 * np.log10(processed_rms / max(original_rms, 1e-10)))
        }
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio of audio
        
        Args:
            audio: Audio signal
            
        Returns:
            Estimated SNR in dB
        """
        try:
            # Simple SNR estimation using signal energy vs quiet segments
            
            # Find quiet segments (bottom 10% of energy)
            frame_size = 1024
            frames = librosa.util.frame(audio, frame_length=frame_size, 
                                      hop_length=frame_size//2, axis=0)
            frame_energy = np.sum(frames**2, axis=1)
            
            # Estimate noise and signal power
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frames[frame_energy <= noise_threshold]
            signal_frames = frames[frame_energy > noise_threshold]
            
            if len(noise_frames) > 0 and len(signal_frames) > 0:
                noise_power = np.mean(noise_frames**2)
                signal_power = np.mean(signal_frames**2)
                snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            else:
                # Fallback: use overall RMS
                snr = 20 * np.log10(np.sqrt(np.mean(audio**2)) / max(np.std(audio), 1e-10))
            
            return np.clip(snr, -20, 60)  # Reasonable range
            
        except Exception:
            return 10.0  # Default reasonable SNR
    
    def train_model_on_data(self, clean_audio_paths: List[str], 
                           noisy_audio_paths: List[str],
                           epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train autoencoder model on clean/noisy audio pairs
        
        Args:
            clean_audio_paths: List of paths to clean audio files
            noisy_audio_paths: List of paths to noisy audio files  
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history dictionary
        """
        if self.autoencoder_model is None:
            self._build_autoencoder_model()
        
        try:
            # Load and prepare training data
            X_train, y_train = [], []
            
            for clean_path, noisy_path in zip(clean_audio_paths, noisy_audio_paths):
                clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
                noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate)
                
                # Segment audio into chunks for training
                chunk_size = 8192
                for i in range(0, min(len(clean_audio), len(noisy_audio)) - chunk_size, chunk_size//2):
                    clean_chunk = clean_audio[i:i+chunk_size]
                    noisy_chunk = noisy_audio[i:i+chunk_size]
                    
                    X_train.append(noisy_chunk.reshape(chunk_size, 1))
                    y_train.append(clean_chunk.reshape(chunk_size, 1))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train model
            history = self.autoencoder_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model
            model_path = 'models/trained_noise_reducer.h5'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.autoencoder_model.save(model_path)
            self.is_model_trained = True
            
            return {
                'training_loss': history.history['loss'],
                'validation_loss': history.history['val_loss'],
                'epochs': epochs,
                'samples_trained': len(X_train)
            }
            
        except Exception as e:
            print(f"⚠️ Error training model: {e}")
            return {'error': str(e)}
    
    def get_available_methods(self) -> List[str]:
        """Get list of available noise reduction methods"""
        return ['autoencoder', 'noisereduce', 'wiener', 'spectral', 'adaptive']
    
    def analyze_noise_characteristics(self, audio: np.ndarray) -> Dict:
        """
        Analyze noise characteristics in audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with noise analysis
        """
        try:
            # Basic noise metrics
            rms = np.sqrt(np.mean(audio**2))
            snr = self._estimate_snr(audio)
            
            # Spectral analysis
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Noise floor estimation
            noise_floor = np.percentile(magnitude, 10)
            signal_peak = np.percentile(magnitude, 90)
            dynamic_range = 20 * np.log10(signal_peak / max(noise_floor, 1e-10))
            
            # Frequency analysis
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
            
            return {
                'rms_level': float(rms),
                'snr_estimate': float(snr),
                'dynamic_range': float(dynamic_range),
                'noise_floor': float(noise_floor),
                'signal_peak': float(signal_peak),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'spectral_rolloff': float(spectral_rolloff),
                'recommended_method': self._recommend_method(snr, dynamic_range),
                'recommended_reduction': self._recommend_reduction_level(snr)
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing noise: {e}")
            return {'error': str(e)}
    
    def _recommend_method(self, snr: float, dynamic_range: float) -> str:
        """Recommend best noise reduction method based on audio characteristics"""
        if snr < -5:  # Very noisy
            return 'autoencoder'
        elif snr < 5:  # Moderately noisy
            return 'spectral'
        elif dynamic_range < 20:  # Low dynamic range
            return 'wiener'
        else:  # Relatively clean
            return 'noisereduce'
    
    def _recommend_reduction_level(self, snr: float) -> float:
        """Recommend noise reduction level based on SNR"""
        if snr < -10:
            return 0.9
        elif snr < 0:
            return 0.7
        elif snr < 10:
            return 0.5
        else:
            return 0.3
