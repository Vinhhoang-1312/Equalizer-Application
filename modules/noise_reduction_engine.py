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
import os
import time

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
            # Basic noise reduction with noisereduce - using correct parameters
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                prop_decrease=reduction_level,
                stationary=False,
                n_std_thresh_stationary=1.5
            )
            return reduced_noise.astype(np.float32)

        except Exception as e:
            print(f"⚠️ Noisereduce method failed: {e}")
            # Fallback to basic noise reduction
            try:
                reduced_noise = nr.reduce_noise(y=audio, sr=self.sample_rate)
                return reduced_noise.astype(np.float32)
            except:
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

            # For large audio files, use spectral subtraction to avoid memory issues
            if len(audio) > 100000:  # ~4.5 seconds at 22050 Hz
                print("⚠️ Audio too large for autoencoder, using spectral subtraction")
                return self.spectral_subtraction_method(audio, reduction_level)

            # Process in smaller chunks to avoid memory issues
            chunk_size = 8192
            original_length = len(audio)
            processed_chunks = []

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                
                # Pad chunk if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                # Reshape for Conv1D input (batch_size, timesteps, features)
                chunk_input = chunk.reshape(1, len(chunk), 1)

                try:
                    # Predict with autoencoder
                    denoised_chunk = self.autoencoder_model.predict(chunk_input, verbose=0)
                    denoised_chunk = denoised_chunk.reshape(-1)
                    
                    # Only keep the original length of the chunk
                    if i + chunk_size > original_length:
                        denoised_chunk = denoised_chunk[:original_length - i]
                    
                    processed_chunks.append(denoised_chunk)
                    
                except Exception as chunk_error:
                    print(f"⚠️ Chunk processing failed: {chunk_error}")
                    # Fallback to original chunk
                    if i + chunk_size > original_length:
                        chunk = chunk[:original_length - i]
                    processed_chunks.append(chunk)

            # Concatenate processed chunks
            denoised_audio = np.concatenate(processed_chunks)[:original_length]

            # Blend with original based on reduction level
            blended_audio = (1 - reduction_level) * audio + reduction_level * denoised_audio

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
            # Return fallback values instead of just error to prevent frontend crashes
            return {
                'rms_level': 0.0,
                'snr_estimate': 0.0,
                'dynamic_range': 0.0,
                'noise_floor': 0.0,
                'signal_peak': 0.0,
                'spectral_centroid': 0.0,
                'spectral_bandwidth': 0.0,
                'spectral_rolloff': 0.0,
                'recommended_method': 'spectral',
                'recommended_reduction': 0.5,
                'error': str(e),
                'analysis_failed': True
            }

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

    def create_comparison_analysis(self, original_audio: np.ndarray, processed_audio: np.ndarray, method_used: str, reduction_level: float) -> Dict:
        """
        Tạo phân tích so sánh chi tiết giữa audio gốc và audio đã xử lý
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend

            analysis_data = {
                'method_used': method_used,
                'reduction_level': reduction_level,
                'processing_details': {},
                'comparison_metrics': {},
                'technical_explanation': {}
            }

            # Phân tích audio gốc
            original_analysis = self.analyze_noise_characteristics(original_audio)
            processed_analysis = self.analyze_noise_characteristics(processed_audio)

            # Check for errors in analysis but use fallback values
            if 'analysis_failed' in original_analysis:
                print(f"⚠️ Original audio analysis had issues: {original_analysis.get('error', 'Unknown error')}")
            if 'analysis_failed' in processed_analysis:
                print(f"⚠️ Processed audio analysis had issues: {processed_analysis.get('error', 'Unknown error')}")

            analysis_data['original_metrics'] = original_analysis
            analysis_data['processed_metrics'] = processed_analysis

            # Tính toán metrics so sánh với fallback values
            snr_improvement = processed_analysis.get('snr_estimate', 0.0) - original_analysis.get('snr_estimate', 0.0)
            
            original_rms = original_analysis.get('rms_level', 0.001)  # Prevent division by zero
            processed_rms = processed_analysis.get('rms_level', 0.001)
            rms_reduction = (original_rms - processed_rms) / original_rms * 100 if original_rms > 0 else 0.0

            analysis_data['comparison_metrics'] = {
                'snr_improvement_db': float(snr_improvement),
                'rms_reduction_percent': float(rms_reduction),
                'noise_floor_reduction': float(original_analysis.get('noise_floor', 0.0) - processed_analysis.get('noise_floor', 0.0)),
                'dynamic_range_change': float(processed_analysis.get('dynamic_range', 0.0) - original_analysis.get('dynamic_range', 0.0))
            }

            # Giải thích kỹ thuật
            analysis_data['technical_explanation'] = self._generate_technical_explanation(
                method_used, reduction_level, original_analysis, processed_analysis, analysis_data['comparison_metrics']
            )

            # Tạo biểu đồ so sánh
            comparison_chart_path = self._create_comparison_charts(
                original_audio, processed_audio, method_used, analysis_data
            )
            analysis_data['comparison_chart_path'] = comparison_chart_path

            return analysis_data

        except Exception as e:
            print(f"⚠️ Error creating comparison analysis: {e}")
            return {'error': str(e)}

    def _generate_technical_explanation(self, method: str, reduction_level: float, original_metrics: Dict, processed_metrics: Dict, comparison_metrics: Dict) -> Dict:
        """
        Tạo giải thích kỹ thuật chi tiết về quá trình xử lý
        """
        explanations = {
            'method_description': {},
            'processing_steps': [],
            'parameter_explanation': {},
            'results_interpretation': {}
        }

        # Giải thích phương pháp
        method_descriptions = {
            'autoencoder': {
                'name': 'Deep Learning Autoencoder (1D CNN)',
                'description': 'Sử dụng mạng neural tích chập 1D để học cách loại bỏ nhiễu từ audio signal. Model được train để tái tạo lại audio sạch từ audio có nhiễu.',
                'advantages': 'Hiệu quả cao với nhiễu phức tạp, adaptive learning, xử lý được nhiều loại nhiễu',
                'suitable_for': 'Audio có nhiễu không đều, nhiễu background phức tạp'
            },
            'noisereduce': {
                'name': 'Spectral Gating Algorithm',
                'description': 'Sử dụng thuật toán spectral gating để phân tích và loại bỏ nhiễu dựa trên đặc tính tần số.',
                'advantages': 'Nhanh, hiệu quả với nhiễu stationary, không cần training',
                'suitable_for': 'Nhiễu ổn định, background noise đều'
            },
            'wiener': {
                'name': 'Wiener Filter',
                'description': 'Bộ lọc optimal dựa trên thống kê để ước lượng tín hiệu sạch từ tín hiệu nhiễu.',
                'advantages': 'Optimal cho nhiễu Gaussian, bảo toàn tín hiệu gốc tốt',
                'suitable_for': 'Nhiễu Gaussian, white noise'
            },
            'spectral': {
                'name': 'Spectral Subtraction',
                'description': 'Loại bỏ nhiễu bằng cách trừ phổ nhiễu ước lượng từ phổ tín hiệu gốc.',
                'advantages': 'Đơn giản, hiệu quả với nhiễu additive',
                'suitable_for': 'Nhiễu additive, white noise'
            },
            'adaptive': {
                'name': 'Adaptive LMS Filter',
                'description': 'Bộ lọc thích nghi sử dụng thuật toán LMS để loại bỏ nhiễu.',
                'advantages': 'Thích nghi với môi trường thay đổi',
                'suitable_for': 'Nhiễu thay đổi theo thời gian'
            }
        }

        explanations['method_description'] = method_descriptions.get(method, {
            'name': f'Unknown Method: {method}',
            'description': 'Phương pháp không xác định',
            'advantages': 'N/A',
            'suitable_for': 'N/A'
        })

        # Các bước xử lý với safe access to metrics
        original_snr = original_metrics.get('snr_estimate', 0.0)
        processed_snr = processed_metrics.get('snr_estimate', 0.0) 
        snr_improvement = comparison_metrics.get('snr_improvement_db', 0.0)
        
        processing_steps = [
            f"1. Phân tích đặc tính nhiễu của audio gốc (SNR: {original_snr:.1f} dB)",
            f"2. Áp dụng phương pháp {explanations['method_description']['name']}",
            f"3. Điều chỉnh mức độ giảm nhiễu: {reduction_level:.1%}",
            f"4. Tái tạo audio với SNR mới: {processed_snr:.1f} dB",
            f"5. Cải thiện SNR tổng thể: {snr_improvement:.1f} dB"
        ]

        explanations['processing_steps'] = processing_steps

        # Giải thích tham số với safe access
        explanations['parameter_explanation'] = {
            'reduction_level': f"Mức độ giảm nhiễu {reduction_level:.1%} - {'Mạnh' if reduction_level > 0.7 else 'Vừa phải' if reduction_level > 0.4 else 'Nhẹ'}",
            'snr_improvement': f"Cải thiện SNR {snr_improvement:.1f} dB - {'Tốt' if snr_improvement > 3 else 'Khá' if snr_improvement > 1 else 'Thấp'}",
            'dynamic_range': f"Thay đổi dynamic range: {comparison_metrics.get('dynamic_range_change', 0.0):.1f} dB"
        }

        # Giải thích kết quả với safe access
        if snr_improvement > 5:
            result_quality = "Xuất sắc - Chất lượng audio được cải thiện đáng kể"
        elif snr_improvement > 2:
            result_quality = "Tốt - Nhiễu được giảm rõ rệt"
        elif snr_improvement > 0:
            result_quality = "Khá - Có cải thiện nhẹ về chất lượng"
        else:
            result_quality = "Kém - Cần điều chỉnh tham số hoặc thử phương pháp khác"

        explanations['results_interpretation'] = {
            'overall_quality': result_quality,
            'snr_analysis': f"SNR tăng từ {original_snr:.1f} dB lên {processed_snr:.1f} dB",
            'recommendation': self._get_improvement_recommendation(comparison_metrics, method)
        }

        return explanations

    def _get_improvement_recommendation(self, metrics: Dict, current_method: str) -> str:
        """Đưa ra khuyến nghị cải thiện với safe access"""
        snr_improvement = metrics.get('snr_improvement_db', 0.0)
        
        if snr_improvement < 1:
            if current_method != 'autoencoder':
                return "Thử sử dụng phương pháp Autoencoder để có kết quả tốt hơn"
            else:
                return "Tăng reduction_level hoặc thử phương pháp Spectral Subtraction"
        elif snr_improvement > 5:
            return "Kết quả tốt! Có thể giảm reduction_level để bảo toàn chất lượng âm thanh gốc"
        else:
            return "Kết quả khá tốt, có thể tinh chỉnh tham số để tối ưu hơn"

    def _create_comparison_charts(self, original_audio: np.ndarray, processed_audio: np.ndarray, method: str, analysis_data: Dict) -> str:
        """Tạo biểu đồ so sánh chi tiết TRƯỚC vs SAU xử lý"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            try:
                import seaborn as sns
                plt.style.use('default')
                sns.set_palette("husl")
            except ImportError:
                plt.style.use('default')
                pass
            
            # Limit audio length
            max_samples = 44100
            if len(original_audio) > max_samples:
                original_audio = original_audio[:max_samples]
            if len(processed_audio) > max_samples:
                processed_audio = processed_audio[:max_samples]
            
            # Create BEFORE vs AFTER comparison layout: 4 rows x 2 columns
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            fig.suptitle(f'🎵 NOISE REDUCTION: TRƯỚC vs SAU - {method.upper()}\n👈 BEFORE (Có Nhiễu) vs AFTER (Đã Lọc) 👉', 
                        fontsize=16, fontweight='bold', y=0.98)

            # ROW 1: WAVEFORM COMPARISON - BEFORE vs AFTER
            # Left: Original (BEFORE)
            ax_orig_wave = axes[0, 0]
            downsample_factor = max(1, len(original_audio) // 2000)
            orig_downsampled = original_audio[::downsample_factor]
            time_orig = np.linspace(0, len(orig_downsampled)/self.sample_rate*downsample_factor, len(orig_downsampled))
            
            ax_orig_wave.plot(time_orig, orig_downsampled, color='#ff4757', linewidth=1.2, alpha=0.8)
            ax_orig_wave.set_title('🔴 TRƯỚC: Audio Gốc (Có Nhiễu)', fontweight='bold', fontsize=12, color='#ff4757')
            ax_orig_wave.set_xlabel('Time (s)')
            ax_orig_wave.set_ylabel('Amplitude')
            ax_orig_wave.grid(True, alpha=0.3)
            ax_orig_wave.set_facecolor('#fff5f5')
            
            # Right: Processed (AFTER)
            ax_proc_wave = axes[0, 1]
            proc_downsampled = processed_audio[::downsample_factor]
            time_proc = np.linspace(0, len(proc_downsampled)/self.sample_rate*downsample_factor, len(proc_downsampled))
            
            ax_proc_wave.plot(time_proc, proc_downsampled, color='#2ed573', linewidth=1.2, alpha=0.8)
            ax_proc_wave.set_title('� SAU: Audio Đã Lọc (Giảm Nhiễu)', fontweight='bold', fontsize=12, color='#2ed573')
            ax_proc_wave.set_xlabel('Time (s)')
            ax_proc_wave.set_ylabel('Amplitude')
            ax_proc_wave.grid(True, alpha=0.3)
            ax_proc_wave.set_facecolor('#f5fff5')

            # ROW 2: FREQUENCY SPECTRUM COMPARISON
            fft_size = min(2048, len(original_audio))
            orig_fft = np.abs(np.fft.fft(original_audio[:fft_size]))[:fft_size//2]
            proc_fft = np.abs(np.fft.fft(processed_audio[:fft_size]))[:fft_size//2]
            freqs = np.fft.fftfreq(fft_size, 1/self.sample_rate)[:fft_size//2]
            
            # Left: Original Spectrum
            ax_orig_freq = axes[1, 0]
            ax_orig_freq.semilogy(freqs, orig_fft, color='#ff4757', linewidth=1.5, alpha=0.8)
            ax_orig_freq.set_title('🔴 TRƯỚC: Phổ Tần Số (Nhiều Noise)', fontweight='bold', fontsize=12, color='#ff4757')
            ax_orig_freq.set_xlabel('Frequency (Hz)')
            ax_orig_freq.set_ylabel('Magnitude (Log)')
            ax_orig_freq.grid(True, alpha=0.3)
            ax_orig_freq.set_facecolor('#fff5f5')
            
            # Right: Processed Spectrum
            ax_proc_freq = axes[1, 1]
            ax_proc_freq.semilogy(freqs, proc_fft, color='#2ed573', linewidth=1.5, alpha=0.8)
            ax_proc_freq.set_title('🟢 SAU: Phổ Tần Số (Noise Đã Giảm)', fontweight='bold', fontsize=12, color='#2ed573')
            ax_proc_freq.set_xlabel('Frequency (Hz)')
            ax_proc_freq.set_ylabel('Magnitude (Log)')
            ax_proc_freq.grid(True, alpha=0.3)
            ax_proc_freq.set_facecolor('#f5fff5')

            # ROW 3: SPECTROGRAM COMPARISON
            hop_length = 512
            n_fft = 1024
            
            # Left: Original Spectrogram
            ax_orig_spec = axes[2, 0]
            stft_orig = np.abs(self._compute_stft(original_audio, n_fft=n_fft, hop_length=hop_length))
            stft_orig_db = 20 * np.log10(stft_orig + 1e-10)
            
            im1 = ax_orig_spec.imshow(stft_orig_db, aspect='auto', origin='lower', cmap='Reds', vmin=-60, vmax=0)
            ax_orig_spec.set_title('🔴 TRƯỚC: Spectrogram (Noise Rõ Ràng)', fontweight='bold', fontsize=12, color='#ff4757')
            ax_orig_spec.set_xlabel('Time Frames')
            ax_orig_spec.set_ylabel('Frequency Bins')
            plt.colorbar(im1, ax=ax_orig_spec, label='Magnitude (dB)')
            
            # Right: Processed Spectrogram
            ax_proc_spec = axes[2, 1]
            stft_proc = np.abs(self._compute_stft(processed_audio, n_fft=n_fft, hop_length=hop_length))
            stft_proc_db = 20 * np.log10(stft_proc + 1e-10)
            
            im2 = ax_proc_spec.imshow(stft_proc_db, aspect='auto', origin='lower', cmap='Greens', vmin=-60, vmax=0)
            ax_proc_spec.set_title('🟢 SAU: Spectrogram (Noise Đã Sạch)', fontweight='bold', fontsize=12, color='#2ed573')
            ax_proc_spec.set_xlabel('Time Frames')
            ax_proc_spec.set_ylabel('Frequency Bins')
            plt.colorbar(im2, ax=ax_proc_spec, label='Magnitude (dB)')

            # ROW 4: METRICS & SUMMARY
            # Left: Bar Chart Comparison
            ax_metrics = axes[3, 0]
            original_metrics = analysis_data.get('original_metrics', {})
            processed_metrics = analysis_data.get('processed_metrics', {})
            
            metrics_names = ['SNR (dB)', 'RMS Level×1000', 'Dynamic Range']
            orig_values = [
                original_metrics.get('snr_estimate', 0.0),
                original_metrics.get('rms_level', 0.0) * 1000,
                original_metrics.get('dynamic_range', 0.0)
            ]
            proc_values = [
                processed_metrics.get('snr_estimate', 0.0),
                processed_metrics.get('rms_level', 0.0) * 1000,
                processed_metrics.get('dynamic_range', 0.0)
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax_metrics.bar(x - width/2, orig_values, width, label='🔴 TRƯỚC', color='#ff4757', alpha=0.8)
            bars2 = ax_metrics.bar(x + width/2, proc_values, width, label='🟢 SAU', color='#2ed573', alpha=0.8)
            
            ax_metrics.set_title('� So Sánh Metrics: TRƯỚC vs SAU', fontweight='bold', fontsize=12)
            ax_metrics.set_xlabel('Audio Metrics')
            ax_metrics.set_ylabel('Values')
            ax_metrics.set_xticks(x)
            ax_metrics.set_xticklabels(metrics_names, rotation=15)
            ax_metrics.legend(fontsize=10)
            ax_metrics.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9, color='#ff4757')
            for bar in bars2:
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9, color='#2ed573')

            # Right: Improvement Summary
            ax_summary = axes[3, 1]
            ax_summary.axis('off')
            
            comparison_metrics = analysis_data.get('comparison_metrics', {})
            snr_improvement = comparison_metrics.get('snr_improvement_db', 0.0)
            rms_reduction = comparison_metrics.get('rms_reduction_percent', 0.0)
            
            # Quality assessment
            if snr_improvement > 5:
                quality = "XUẤT SẮC ✨"
                quality_color = "#2ed573"
                quality_emoji = "🌟"
            elif snr_improvement > 2:
                quality = "TỐT ✅"
                quality_color = "#3c40c6"
                quality_emoji = "👍"
            elif snr_improvement > 0:
                quality = "KHẤP KHIỂN ⚠️"
                quality_color = "#ffa502"
                quality_emoji = "⚡"
            else:
                quality = "CẦN CẢI THIỆN ❌"
                quality_color = "#ff4757"
                quality_emoji = "🔧"
            
            # Calculate percentage improvement
            orig_snr = original_metrics.get('snr_estimate', 0.0)
            improvement_percent = (snr_improvement / max(orig_snr, 0.1)) * 100 if orig_snr > 0 else 0
            
            summary_text = f"""{quality_emoji} KẾT QUẢ SO SÁNH
            
🎵 Phương Pháp: {method.upper()}
{quality_emoji} Chất Lượng: {quality}

� HIỆU QUẢ TRƯỚC vs SAU:
• SNR Cải Thiện: +{snr_improvement:.2f} dB
• RMS Giảm: {rms_reduction:.1f}%
• Tỷ Lệ Cải Thiện: {improvement_percent:.1f}%

🔧 CHI TIẾT KỸ THUẬT:
• Sample Rate: {self.sample_rate} Hz
• Độ Dài: {len(original_audio)/self.sample_rate:.2f}s
• Thuật Toán: {method.title()}

💡 ĐÁNH GIÁ:
{quality_emoji} {quality}
{"Noise đã được loại bỏ hiệu quả!" if snr_improvement > 2 else "Có thể cần điều chỉnh tham số để tối ưu hơn."}
"""
            
            ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=quality_color, alpha=0.2))

            plt.tight_layout()
            
            # Add watermark
            fig.text(0.99, 0.01, 'Advanced Audio Processing - Before vs After Comparison', 
                    fontsize=8, alpha=0.6, ha='right', va='bottom')
            
            # Save chart
            chart_path = f'static/results/comparison_before_after_{method}_{int(time.time())}.png'
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return chart_path

        except Exception as e:
            print(f"⚠️ Error creating comparison charts: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_stft(self, audio: np.ndarray, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
        """Compute Short-Time Fourier Transform"""
        try:
            # Simple STFT implementation
            stft_matrix = []
            for i in range(0, len(audio) - n_fft, hop_length):
                window = audio[i:i + n_fft]
                # Apply Hanning window
                windowed = window * np.hanning(len(window))
                fft_result = np.fft.fft(windowed)[:n_fft//2]
                stft_matrix.append(fft_result)
            
            return np.array(stft_matrix).T
        except Exception:
            # Fallback: return a simple matrix
            return np.random.random((n_fft//2, 10))

    def _estimate_snr_simple(self, audio: np.ndarray) -> float:
        """Simple SNR estimation for a window"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Calculate signal power
            signal_power = np.mean(audio**2)
            
            # Estimate noise power (using quieter portions)
            sorted_power = np.sort(audio**2)
            noise_power = np.mean(sorted_power[:len(sorted_power)//4])  # Bottom 25%
            
            if noise_power == 0:
                return 20.0  # High SNR if no noise detected
            
            snr = 10 * np.log10(signal_power / noise_power)
            return max(0, min(snr, 40))  # Clamp between 0 and 40 dB
        except Exception:
            return 10.0  # Default SNR

    def _create_3d_spectrogram(self, original_audio: np.ndarray, processed_audio: np.ndarray, method: str):
        """Create 3D spectrogram visualization"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create 3D figure
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Compute spectrograms
            hop_length = 512
            n_fft = 1024
            
            stft_orig = np.abs(self._compute_stft(original_audio[:22050], n_fft=n_fft, hop_length=hop_length))
            stft_orig_db = 20 * np.log10(stft_orig + 1e-10)
            
            # Create meshgrid for 3D plot
            time_frames = np.arange(stft_orig_db.shape[1])
            freq_bins = np.arange(stft_orig_db.shape[0])
            T, F = np.meshgrid(time_frames, freq_bins)
            
            # Plot 3D surface
            surf = ax.plot_surface(T, F, stft_orig_db, cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Frequency Bins')
            ax.set_zlabel('Magnitude (dB)')
            ax.set_title(f'3D Spectrogram - {method.upper()}')
            
            # Add colorbar
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
            # Save 3D visualization
            chart_3d_path = f'static/results/3d_spectrogram_{method}_{int(time.time())}.png'
            plt.savefig(chart_3d_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            return chart_3d_path
            
        except Exception as e:
            print(f"⚠️ Error creating 3D spectrogram: {e}")
            return None

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