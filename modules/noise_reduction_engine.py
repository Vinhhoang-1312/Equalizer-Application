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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
import scipy.signal
import noisereduce as nr
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

class NoiseReductionEngine:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.autoencoder_model = None
        self.is_model_trained = False
        self.n_fft = 2048
        self.hop_length = 512
        self.wiener_params = {'noise_estimate_seconds': 0.5, 'filter_order': 5}
        self.spectral_params = {'alpha': 2.0, 'beta': 0.01, 'window_size': self.n_fft, 'hop_length': self.hop_length}
        self._load_models()
    def _load_models(self):
        try:
            model_path = 'models/advanced_noise_reducer.h5'
            if tf.io.gfile.exists(model_path):
                self.autoencoder_model = tf.keras.models.load_model(model_path)
                self.is_model_trained = True
                print("✓ 2D CNN Noise reduction model loaded successfully")
            else:
                print("! 2D CNN model not found, will create new one")
                self._build_autoencoder_model()
        except Exception as e:
            print(f"⚠️ Error loading 2D CNN model: {e}. Building a new one.")
            self._build_autoencoder_model()

    def _build_autoencoder_model(self):
        input_shape = (1024, 1296, 1)
        try:
            input_layer = Input(shape=input_shape)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            model = Model(input_layer, decoded)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            self.autoencoder_model = model
            self.is_model_trained = False
            print("✓ 2D CNN Autoencoder model built successfully")
        except Exception as e:
            print(f"⚠️ Error building 2D CNN autoencoder model: {e}")
            self.autoencoder_model = None

    def autoencoder_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        try:
            if self.autoencoder_model is None:
                print("⚠️ Autoencoder model not available, falling back to spectral subtraction")
                return self.spectral_subtraction_method(audio, reduction_level)
            model_input_shape = self.autoencoder_model.input_shape[1:3]
            stft_original = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude, phase = np.abs(stft_original), np.angle(stft_original)
            original_shape = magnitude.shape
            magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-8)
            magnitude_reshaped = magnitude_normalized[:, :, np.newaxis]
            magnitude_resized = tf.image.resize(magnitude_reshaped, model_input_shape)
            model_input = magnitude_resized[np.newaxis, :, :, :]
            predicted_spec_normalized = self.autoencoder_model.predict(model_input, verbose=0).squeeze()
            predicted_spec_resized = tf.image.resize(predicted_spec_normalized[:, :, np.newaxis], original_shape)
            denoised_magnitude_normalized = predicted_spec_resized.numpy().squeeze()
            denoised_magnitude = denoised_magnitude_normalized * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)
            denoised_stft = denoised_magnitude * np.exp(1j * phase)
            denoised_audio = librosa.istft(denoised_stft, hop_length=self.hop_length, length=len(audio))
            blended_audio = (1 - reduction_level) * audio + reduction_level * denoised_audio
            return blended_audio.astype(np.float32)
        except Exception as e:
            print(f"⚠️ Autoencoder method failed: {e}")
            return self.spectral_subtraction_method(audio, reduction_level)

    def noisereduce_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        try:
            # Removed unsupported parameter 'n_thresh_nonstationary' for compatibility
            return nr.reduce_noise(y=audio, sr=self.sample_rate, prop_decrease=reduction_level, stationary=False, n_std_thresh_stationary=1.5).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Noisereduce method failed: {e}")
            return audio

    def wiener_filter_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        try:
            noise_samples = int(self.wiener_params['noise_estimate_seconds'] * self.sample_rate)
            noise_segment = audio[:min(noise_samples, len(audio)//4)]
            f, psd_signal = scipy.signal.welch(audio, self.sample_rate, nperseg=1024)
            f, psd_noise = scipy.signal.welch(noise_segment, self.sample_rate, nperseg=1024)
            H = psd_signal / (psd_signal + psd_noise * (1/reduction_level - 1))
            audio_fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            H_interp = np.interp(np.abs(freqs), f, H)
            filtered_fft = audio_fft * H_interp
            return np.real(np.fft.ifft(filtered_fft)).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Wiener filter method failed: {e}")
            return audio

    def spectral_subtraction_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        try:
            stft = librosa.stft(audio, n_fft=self.spectral_params['window_size'], hop_length=self.spectral_params['hop_length'])
            magnitude, phase = np.abs(stft), np.angle(stft)
            noise_spectrum = np.mean(magnitude[:, :5], axis=1, keepdims=True)
            enhanced_magnitude = np.maximum(magnitude - self.spectral_params['alpha'] * reduction_level * noise_spectrum, self.spectral_params['beta'] * magnitude)
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            return librosa.istft(enhanced_stft, hop_length=self.spectral_params['hop_length']).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Spectral subtraction method failed: {e}")
            return audio

    def adaptive_filter_method(self, audio: np.ndarray, reduction_level: float = 0.7) -> np.ndarray:
        try:
            filter_length = 32
            mu = 0.01 * reduction_level
            w = np.zeros(filter_length)
            y = np.zeros_like(audio)
            for n in range(filter_length, len(audio)):
                x = audio[n-filter_length:n]
                y[n] = np.dot(w, x[::-1])
                e = audio[n] - y[n]
                w += mu * e * x[::-1]
            return (audio - y).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Adaptive filter method failed: {e}")
            return audio

    def reduce_noise(self, audio: np.ndarray, method: str = 'autoencoder', reduction_level: float = 0.7) -> np.ndarray:
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

    def train_model_on_data(self, clean_audio_paths: List[str], noisy_audio_paths: List[str], epochs: int = 10, batch_size: int = 8) -> Dict:
        if self.autoencoder_model is None:
            self._build_autoencoder_model()
        try:
            model_input_shape = self.autoencoder_model.input_shape[1:3]
            X_train, y_train = [], []
            print(f"Preparing data for training with target shape {model_input_shape}...")
            for clean_path, noisy_path in zip(clean_audio_paths, noisy_audio_paths):
                clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
                noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate)
                clean_mag = np.abs(librosa.stft(clean_audio, n_fft=self.n_fft, hop_length=self.hop_length))
                noisy_mag = np.abs(librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length))
                clean_mag = (clean_mag - np.min(clean_mag)) / (np.max(clean_mag) - np.min(clean_mag) + 1e-8)
                noisy_mag = (noisy_mag - np.min(noisy_mag)) / (np.max(noisy_mag) - np.min(noisy_mag) + 1e-8)
                clean_resized = tf.image.resize(clean_mag[:, :, np.newaxis], model_input_shape)
                noisy_resized = tf.image.resize(noisy_mag[:, :, np.newaxis], model_input_shape)
                X_train.append(noisy_resized)
                y_train.append(clean_resized)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print(f"Data prepared. Starting training on {len(X_train)} samples...")
            history = self.autoencoder_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            model_path = 'models/trained_noise_reducer_2d.h5'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.autoencoder_model.save(model_path)
            self.is_model_trained = True
            return {'training_loss': history.history['loss'], 'validation_loss': history.history['val_loss'], 'epochs': epochs, 'samples_trained': len(X_train)}
        except Exception as e:
            print(f"⚠️ Error training model: {e}")
            return {'error': str(e)}

    def get_available_methods(self) -> List[str]:
        return ['autoencoder', 'noisereduce', 'wiener', 'spectral', 'adaptive']

    def _estimate_snr(self, audio: np.ndarray) -> float:
        try:
            frame_size = 1024
            frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=frame_size//2, axis=0)
            frame_energy = np.sum(frames**2, axis=1)
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frames[frame_energy <= noise_threshold]
            signal_frames = frames[frame_energy > noise_threshold]
            if len(noise_frames) > 0 and len(signal_frames) > 0:
                noise_power = np.mean(noise_frames**2)
                signal_power = np.mean(signal_frames**2)
                snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            else:
                snr = 20 * np.log10(np.sqrt(np.mean(audio**2)) / max(np.std(audio), 1e-10))
            return np.clip(snr, -20, 60)
        except Exception:
            return 10.0

    def analyze_noise_characteristics(self, audio: np.ndarray) -> Dict:
        try:
            rms = np.sqrt(np.mean(audio**2))
            snr = self._estimate_snr(audio)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            noise_floor = np.percentile(magnitude, 10)
            signal_peak = np.percentile(magnitude, 90)
            dynamic_range = 20 * np.log10(signal_peak / max(noise_floor, 1e-10))
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
        if snr < -5: return 'autoencoder'
        elif snr < 5: return 'spectral'
        elif dynamic_range < 20: return 'wiener'
        else: return 'noisereduce'

    def _recommend_reduction_level(self, snr: float) -> float:
        if snr < -10: return 0.9
        elif snr < 0: return 0.7
        elif snr < 10: return 0.5
        else: return 0.3

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
            
            analysis_data['original_metrics'] = original_analysis
            analysis_data['processed_metrics'] = processed_analysis
            
            # Tính toán metrics so sánh
            snr_improvement = processed_analysis['snr_estimate'] - original_analysis['snr_estimate']
            rms_reduction = (original_analysis['rms_level'] - processed_analysis['rms_level']) / original_analysis['rms_level'] * 100
            
            analysis_data['comparison_metrics'] = {
                'snr_improvement_db': float(snr_improvement),
                'rms_reduction_percent': float(rms_reduction),
                'noise_floor_reduction': float(original_analysis['noise_floor'] - processed_analysis['noise_floor']),
                'dynamic_range_change': float(processed_analysis['dynamic_range'] - original_analysis['dynamic_range'])
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
                'name': 'Deep Learning Autoencoder (2D CNN)',
                'description': 'Sử dụng mạng neural tích chập 2D để học cách loại bỏ nhiễu từ spectrogram. Model được train để tái tạo lại audio sạch từ audio có nhiễu.',
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
                'description': 'Trừ ước lượng phổ nhiễu từ phổ tín hiệu gốc trong miền tần số.',
                'advantages': 'Đơn giản, hiệu quả với nhiễu additive',
                'suitable_for': 'Nhiễu cộng (additive noise)'
            },
            'adaptive': {
                'name': 'Adaptive Filter (LMS)',
                'description': 'Bộ lọc thích ứng sử dụng thuật toán Least Mean Squares để học pattern nhiễu.',
                'advantages': 'Thích ứng real-time, học được pattern nhiễu',
                'suitable_for': 'Nhiễu có pattern lặp lại'
            }
        }
        
        explanations['method_description'] = method_descriptions.get(method, {
            'name': 'Unknown Method',
            'description': 'Phương pháp không xác định'
        })
        
        # Các bước xử lý
        common_steps = [
            f"1. Phân tích audio gốc: {len(original_metrics)} features được trích xuất",
            f"2. Áp dụng phương pháp {method} với mức giảm nhiễu {reduction_level:.1%}",
            f"3. Ước lượng SNR gốc: {original_metrics['snr_estimate']:.1f} dB",
            f"4. Xử lý trong miền {'tần số' if method in ['spectral', 'wiener'] else 'time-frequency'}",
            f"5. Tái tạo audio với SNR mới: {processed_metrics['snr_estimate']:.1f} dB"
        ]
        
        if method == 'autoencoder':
            common_steps.extend([
                "6. Convert audio → Spectrogram (STFT)",
                "7. Normalize magnitude spectrogram",
                f"8. Resize to model input shape: {self.autoencoder_model.input_shape[1:3] if self.autoencoder_model else 'N/A'}",
                "9. Deep learning inference",
                "10. Denormalize và convert back to audio"
            ])
        
        explanations['processing_steps'] = common_steps
        
        # Giải thích parameters
        explanations['parameter_explanation'] = {
            'reduction_level': f"Mức độ giảm nhiễu: {reduction_level:.1%} - {'Mạnh' if reduction_level > 0.7 else 'Vừa phải' if reduction_level > 0.4 else 'Nhẹ'}",
            'sample_rate': f"Tần số lấy mẫu: {self.sample_rate} Hz",
            'n_fft': f"FFT window size: {self.n_fft} samples",
            'hop_length': f"Hop length: {self.hop_length} samples"
        }
        
        # Giải thích kết quả
        snr_change = comparison_metrics['snr_improvement_db']
        rms_change = comparison_metrics['rms_reduction_percent']
        
        result_quality = "Excellent" if snr_change > 5 else "Good" if snr_change > 2 else "Moderate" if snr_change > 0 else "Poor"
        
        explanations['results_interpretation'] = {
            'quality_assessment': result_quality,
            'snr_explanation': f"SNR cải thiện {snr_change:.1f} dB - {'Rất tốt' if snr_change > 5 else 'Tốt' if snr_change > 2 else 'Khá' if snr_change > 0 else 'Cần cải thiện'}",
            'rms_explanation': f"RMS giảm {rms_change:.1f}% - mức độ giảm âm lượng nhiễu",
            'recommendation': self._get_quality_recommendation(snr_change, rms_change, method)
        }
        
        return explanations
    
    def _get_quality_recommendation(self, snr_change: float, rms_change: float, method: str) -> str:
        """Đưa ra khuyến nghị dựa trên kết quả"""
        if snr_change > 5:
            return "Kết quả rất tốt! Nhiễu đã được loại bỏ hiệu quả."
        elif snr_change > 2:
            return "Kết quả tốt. Audio đã được cải thiện đáng kể."
        elif snr_change > 0:
            return f"Có cải thiện nhẹ. Có thể thử phương pháp khác hoặc tăng reduction_level."
        else:
            available_methods = [m for m in self.get_available_methods() if m != method]
            return f"Kết quả chưa tối ưu. Khuyến nghị thử phương pháp: {available_methods[0] if available_methods else 'khác'}"
    
    def _create_comparison_charts(self, original_audio: np.ndarray, processed_audio: np.ndarray, method: str, analysis_data: Dict) -> str:
        """
        Tạo biểu đồ so sánh chi tiết giữa audio gốc và đã xử lý
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Setup figure với multiple subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Noise Reduction Analysis - {method.upper()} Method', fontsize=16, fontweight='bold')
            
            # Tạo time axis
            time_original = np.linspace(0, len(original_audio)/self.sample_rate, len(original_audio))
            time_processed = np.linspace(0, len(processed_audio)/self.sample_rate, len(processed_audio))
            
            # 1. Waveform comparison
            axes[0,0].plot(time_original, original_audio, alpha=0.7, label='Original', color='red')
            axes[0,0].set_title('Original Audio Waveform')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Amplitude')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            
            axes[0,1].plot(time_processed, processed_audio, alpha=0.7, label='Processed', color='blue')
            axes[0,1].set_title('Processed Audio Waveform')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Amplitude')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # 2. Spectrogram comparison
            stft_orig = librosa.stft(original_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            stft_proc = librosa.stft(processed_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Convert to dB
            db_orig = librosa.amplitude_to_db(np.abs(stft_orig), ref=np.max)
            db_proc = librosa.amplitude_to_db(np.abs(stft_proc), ref=np.max)
            
            im1 = axes[1,0].imshow(db_orig, aspect='auto', origin='lower', cmap='viridis')
            axes[1,0].set_title('Original Spectrogram (dB)')
            axes[1,0].set_xlabel('Time Frames')
            axes[1,0].set_ylabel('Frequency Bins')
            plt.colorbar(im1, ax=axes[1,0])
            
            im2 = axes[1,1].imshow(db_proc, aspect='auto', origin='lower', cmap='viridis')
            axes[1,1].set_title('Processed Spectrogram (dB)')
            axes[1,1].set_xlabel('Time Frames')
            axes[1,1].set_ylabel('Frequency Bins')
            plt.colorbar(im2, ax=axes[1,1])
            
            # 3. Metrics comparison
            metrics_names = ['SNR (dB)', 'RMS Level', 'Dynamic Range', 'Spectral Centroid']
            original_values = [
                analysis_data['original_metrics']['snr_estimate'],
                analysis_data['original_metrics']['rms_level'],
                analysis_data['original_metrics']['dynamic_range'],
                analysis_data['original_metrics']['spectral_centroid']/1000  # Convert to kHz
            ]
            processed_values = [
                analysis_data['processed_metrics']['snr_estimate'],
                analysis_data['processed_metrics']['rms_level'],
                analysis_data['processed_metrics']['dynamic_range'],
                analysis_data['processed_metrics']['spectral_centroid']/1000  # Convert to kHz
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            axes[2,0].bar(x - width/2, original_values, width, label='Original', color='red', alpha=0.7)
            axes[2,0].bar(x + width/2, processed_values, width, label='Processed', color='blue', alpha=0.7)
            axes[2,0].set_title('Metrics Comparison')
            axes[2,0].set_xlabel('Metrics')
            axes[2,0].set_ylabel('Values')
            axes[2,0].set_xticks(x)
            axes[2,0].set_xticklabels(metrics_names, rotation=45, ha='right')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
            
            # 4. Processing details text
            axes[2,1].axis('off')
            details_text = f"""
Processing Details:
─────────────────────
Method: {analysis_data['technical_explanation']['method_description']['name']}
Reduction Level: {analysis_data['reduction_level']:.1%}
Sample Rate: {self.sample_rate} Hz
Processing Time: {datetime.now().strftime('%H:%M:%S')}

Results:
─────────────────────
SNR Improvement: {analysis_data['comparison_metrics']['snr_improvement_db']:.1f} dB
RMS Reduction: {analysis_data['comparison_metrics']['rms_reduction_percent']:.1f}%
Quality: {analysis_data['technical_explanation']['results_interpretation']['quality_assessment']}

Sample Information:
─────────────────────
Original Duration: {len(original_audio)/self.sample_rate:.2f}s
Original Samples: {len(original_audio):,}
Frequency Range: 0 - {self.sample_rate//2:,} Hz
Processing Window: {self.n_fft} samples
Hop Length: {self.hop_length} samples

Algorithm Details:
─────────────────────
{analysis_data['technical_explanation']['method_description']['description'][:200]}...
            """
            axes[2,1].text(0.05, 0.95, details_text, transform=axes[2,1].transAxes, 
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f'noise_reduction_comparison_{method}_{timestamp}.png'
            chart_path = os.path.join('static', 'results', chart_filename)
            
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_filename
            
        except Exception as e:
            print(f"⚠️ Error creating comparison charts: {e}")
            return None
            
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
