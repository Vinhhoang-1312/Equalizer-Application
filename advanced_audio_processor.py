import librosa
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
import time
from typing import Tuple, List, Optional, Dict
import sounddevice as sd
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Models
        self.noise_reducer = None
        self.genre_classifier = None
        self.scaler = None
        self.feature_scaler = None
        
        # Advanced features
        self.spectral_analyzer = None
        self.beat_detector = None
        self.harmony_analyzer = None
        
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists('models/advanced_noise_reducer.h5'):
                self.noise_reducer = tf.keras.models.load_model('models/advanced_noise_reducer.h5')
            if os.path.exists('models/advanced_genre_classifier.pkl'):
                self.genre_classifier = joblib.load('models/advanced_genre_classifier.pkl')
            if os.path.exists('models/advanced_scaler.pkl'):
                self.scaler = joblib.load('models/advanced_scaler.pkl')
            if os.path.exists('models/feature_scaler.pkl'):
                self.feature_scaler = joblib.load('models/feature_scaler.pkl')
        except Exception as e:
            print(f"Models not found or error loading: {e}")
    
    def advanced_equalizer(self, audio: np.ndarray, 
                          bass_gain: float = 1.0, 
                          mid_gain: float = 1.0, 
                          treble_gain: float = 1.0,
                          sub_bass_gain: float = 1.0,
                          presence_gain: float = 1.0,
                          air_gain: float = 1.0) -> np.ndarray:
        """
        Advanced equalizer with 6 frequency bands
        
        Args:
            audio: Input audio array
            bass_gain: Bass gain (60-250 Hz)
            mid_gain: Mid gain (250-2000 Hz)
            treble_gain: Treble gain (2000-8000 Hz)
            sub_bass_gain: Sub-bass gain (20-60 Hz)
            presence_gain: Presence gain (8000-12000 Hz)
            air_gain: Air gain (12000-20000 Hz)
        
        Returns:
            Processed audio array
        """
        # Convert to frequency domain
        fft_audio = fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Create frequency response with smooth transitions
        freq_response = np.ones_like(freqs)
        
        # Sub-bass filter (20-60 Hz) with smooth rolloff
        sub_bass_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 60)
        freq_response[sub_bass_mask] *= sub_bass_gain
        
        # Bass filter (60-250 Hz)
        bass_mask = (np.abs(freqs) >= 60) & (np.abs(freqs) <= 250)
        freq_response[bass_mask] *= bass_gain
        
        # Mid filter (250-2000 Hz)
        mid_mask = (np.abs(freqs) >= 250) & (np.abs(freqs) <= 2000)
        freq_response[mid_mask] *= mid_gain
        
        # Treble filter (2000-8000 Hz)
        treble_mask = (np.abs(freqs) >= 2000) & (np.abs(freqs) <= 8000)
        freq_response[treble_mask] *= treble_gain
        
        # Presence filter (8000-12000 Hz)
        presence_mask = (np.abs(freqs) >= 8000) & (np.abs(freqs) <= 12000)
        freq_response[presence_mask] *= presence_gain
        
        # Air filter (12000-20000 Hz)
        air_mask = (np.abs(freqs) >= 12000) & (np.abs(freqs) <= 20000)
        freq_response[air_mask] *= air_gain
        
        # Apply smooth transitions between bands
        freq_response = self._apply_smooth_transitions(freq_response, freqs)
        
        # Apply frequency response
        processed_fft = fft_audio * freq_response
        
        # Convert back to time domain
        processed_audio = np.real(ifft(processed_fft))
        
        return processed_audio
    
    def _apply_smooth_transitions(self, freq_response: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Apply smooth transitions between frequency bands"""
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(freq_response, sigma=2)
    
    def extract_advanced_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive audio features for genre classification
        
        Args:
            audio: Input audio array
        
        Returns:
            Feature vector
        """
        features = []
        
        try:
            # MFCC features (26 features)
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
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # Chroma features (24 features)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features.extend([np.mean(chroma[i]) for i in range(12)])
            features.extend([np.std(chroma[i]) for i in range(12)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # Tempo and rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features.append(tempo)
            features.append(len(beats))
            
            # Harmonic and percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.mean(harmonic**2) / (np.mean(harmonic**2) + np.mean(percussive**2))
            features.append(harmonic_ratio)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            features.extend([np.mean(contrast[i]) for i in range(7)])
            features.extend([np.std(contrast[i]) for i in range(7)])
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sample_rate)
            features.extend([np.mean(tonnetz[i]) for i in range(6)])
            features.extend([np.std(tonnetz[i]) for i in range(6)])
            
            # Poly features
            poly_features = librosa.feature.poly_features(y=audio, sr=self.sample_rate)
            features.extend([np.mean(poly_features[i]) for i in range(2)])
            features.extend([np.std(poly_features[i]) for i in range(2)])
            
        except Exception as e:
            print(f"Error extracting advanced features: {e}")
            # Return default features if extraction fails
            features = [0.0] * 100  # Ensure consistent feature count
        
        return np.array(features)
    
    def extract_basic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract basic audio features (fallback method)
        
        Args:
            audio: Input audio array
        
        Returns:
            Feature vector
        """
        features = []
        
        try:
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
            
        except Exception as e:
            print(f"Error extracting basic features: {e}")
            features = [0.0] * 50  # Ensure consistent feature count
        
        return np.array(features)
    
    def advanced_noise_reduction(self, audio: np.ndarray, method: str = 'autoencoder') -> np.ndarray:
        """
        Advanced noise reduction using multiple methods
        
        Args:
            audio: Input audio with noise
            method: 'autoencoder', 'wiener', 'spectral_subtraction', or 'adaptive'
        
        Returns:
            Denoised audio
        """
        if method == 'autoencoder' and self.noise_reducer is not None:
            return self._autoencoder_denoise(audio)
        elif method == 'wiener':
            return self._advanced_wiener_filter(audio)
        elif method == 'spectral_subtraction':
            return self._spectral_subtraction(audio)
        elif method == 'adaptive':
            return self._adaptive_noise_reduction(audio)
        else:
            return self._advanced_wiener_filter(audio)
    
    def _autoencoder_denoise(self, audio: np.ndarray) -> np.ndarray:
        """Denoise using trained autoencoder"""
        try:
            # Convert to spectrogram
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Normalize magnitude
            magnitude_norm = magnitude / np.max(magnitude)
            
            # Reshape for model input - handle variable shapes
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
        except Exception as e:
            print(f"Autoencoder denoising failed: {e}, using Wiener filter instead")
            return self._advanced_wiener_filter(audio)
    
    def _advanced_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """Advanced Wiener filter with adaptive noise estimation"""
        # Estimate noise from multiple segments
        segment_length = int(0.1 * self.sample_rate)
        noise_estimates = []
        
        for i in range(0, len(audio) - segment_length, segment_length):
            segment = audio[i:i + segment_length]
            noise_estimates.append(np.mean(segment**2))
        
        noise_estimate = np.median(noise_estimates)
        
        # Apply Wiener filter with adaptive parameters
        signal_power = np.convolve(audio**2, np.ones(1000)/1000, mode='same')
        wiener_gain = signal_power / (signal_power + noise_estimate)
        
        # Apply gain with smoothing
        wiener_gain = np.clip(wiener_gain, 0.1, 1.0)  # Prevent complete suppression
        denoised = audio * wiener_gain
        
        return denoised
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Spectral subtraction method"""
        # Convert to frequency domain
        fft_audio = fft(audio)
        
        # Estimate noise spectrum from first 0.1 seconds
        noise_samples = int(0.1 * self.sample_rate)
        noise_spectrum = np.abs(fft_audio[:noise_samples])
        noise_estimate = np.mean(noise_spectrum)
        
        # Apply spectral subtraction
        signal_spectrum = np.abs(fft_audio)
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        # Spectral subtraction formula
        clean_spectrum = signal_spectrum - alpha * noise_estimate
        clean_spectrum = np.maximum(clean_spectrum, beta * signal_spectrum)
        
        # Reconstruct signal
        clean_fft = clean_spectrum * np.exp(1j * np.angle(fft_audio))
        clean_audio = np.real(ifft(clean_fft))
        
        return clean_audio
    
    def _adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive noise reduction using Kalman filter approach"""
        # Simple adaptive filter implementation
        mu = 0.01  # Step size
        filter_length = 64
        filter_coeffs = np.zeros(filter_length)
        
        # Create reference noise (estimated from silence)
        noise_samples = int(0.1 * self.sample_rate)
        reference_noise = audio[:noise_samples]
        
        # Apply adaptive filter
        output = np.zeros_like(audio)
        for i in range(filter_length, len(audio)):
            # Input vector
            x = audio[i-filter_length:i]
            
            # Filter output
            y = np.dot(filter_coeffs, x)
            
            # Error
            error = audio[i] - y
            
            # Update filter coefficients
            filter_coeffs += mu * error * x
            
            output[i] = y
        
        return output
    
    def advanced_genre_classification(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Advanced genre classification using multiple methods
        
        Args:
            audio: Input audio array
        
        Returns:
            Tuple of (genre, confidence, additional_info)
        """
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        # Method 1: Try trained model with advanced features first
        if self.genre_classifier is not None:
            try:
                features = self.extract_advanced_features(audio)
                
                if self.feature_scaler is not None:
                    features = self.feature_scaler.transform(features.reshape(1, -1))
                
                prediction = self.genre_classifier.predict_proba(features.reshape(1, -1))[0]
                genre_idx = np.argmax(prediction)
                confidence = prediction[genre_idx]
                
                # Only return if confidence is reasonable
                if confidence > 0.15:
                    additional_info = {
                        'method': 'advanced_trained_model',
                        'all_probabilities': dict(zip(genres, prediction.tolist())),
                        'second_choice': genres[np.argsort(prediction)[-2]],
                        'second_confidence': float(np.sort(prediction)[-2]),
                        'feature_vector': features.flatten().tolist()
                    }
                    return genres[genre_idx], confidence, additional_info
            except Exception as e:
                print(f"⚠️ Advanced trained model failed: {e}")
        
        # Method 2: Try basic features with trained model
        if self.genre_classifier is not None:
            try:
                features = self.extract_basic_features(audio)
                
                if self.feature_scaler is not None:
                    features = self.feature_scaler.transform(features.reshape(1, -1))
                
                prediction = self.genre_classifier.predict_proba(features.reshape(1, -1))[0]
                genre_idx = np.argmax(prediction)
                confidence = prediction[genre_idx]
                
                if confidence > 0.2:
                    additional_info = {
                        'method': 'basic_trained_model',
                        'all_probabilities': dict(zip(genres, prediction.tolist())),
                        'second_choice': genres[np.argsort(prediction)[-2]],
                        'second_confidence': float(np.sort(prediction)[-2]),
                        'feature_vector': features.flatten().tolist()
                    }
                    return genres[genre_idx], confidence, additional_info
            except Exception as e:
                print(f"⚠️ Basic trained model failed: {e}")
        
        # Method 3: Rule-based classification using audio characteristics
        try:
            analysis = self.analyze_audio_characteristics(audio)
            genre, confidence = self._improved_rule_based_classification(analysis)
            
            if confidence > 0.3:
                additional_info = {
                    'method': 'improved_rule_based',
                    'analysis': analysis,
                    'all_probabilities': {genre: confidence}
                }
                return genre, confidence, additional_info
        except Exception as e:
            print(f"⚠️ Improved rule-based classification failed: {e}")
        
        # Method 4: Simple feature-based classification
        try:
            genre, confidence = self._simple_classification(audio)
            
            additional_info = {
                'method': 'simple_features',
                'all_probabilities': {genre: confidence}
            }
            return genre, confidence, additional_info
        except Exception as e:
            print(f"⚠️ Simple classification failed: {e}")
        
        # Fallback: Return most common genre with low confidence
        return "pop", 0.1, {'method': 'fallback', 'all_probabilities': {'pop': 0.1}}
    
    def _improved_rule_based_classification(self, analysis: Dict) -> Tuple[str, float]:
        """Improved rule-based genre classification using audio characteristics"""
        tempo = analysis.get('tempo', 120)
        energy = analysis.get('energy', 0.5)
        brightness = analysis.get('brightness', 2000)
        harmonic_ratio = analysis.get('harmonic_ratio', 0.5)
        noisiness = analysis.get('noisiness', 0.1)
        dynamics = analysis.get('dynamics', 0.1)
        spectral_rolloff = analysis.get('spectral_rolloff', 4000)
        
        # More sophisticated rules based on audio characteristics
        if tempo > 140 and energy > 0.7 and noisiness > 0.15:
            return "metal", 0.7
        elif tempo > 140 and energy > 0.7 and noisiness < 0.15:
            return "rock", 0.6
        elif tempo < 100 and harmonic_ratio > 0.7 and brightness < 2000:
            return "jazz", 0.6
        elif tempo < 90 and harmonic_ratio > 0.8 and dynamics < 0.1:
            return "classical", 0.6
        elif tempo > 120 and energy > 0.6 and brightness > 3000:
            return "pop", 0.5
        elif tempo > 130 and energy > 0.6 and brightness > 2500:
            return "disco", 0.5
        elif tempo > 110 and energy > 0.7 and noisiness > 0.1:
            return "hiphop", 0.5
        elif tempo > 100 and energy > 0.5 and brightness < 2000:
            return "blues", 0.4
        elif tempo > 100 and energy > 0.5 and harmonic_ratio > 0.6:
            return "country", 0.4
        elif tempo > 100 and energy > 0.5 and spectral_rolloff < 3000:
            return "reggae", 0.4
        else:
            return "pop", 0.3
    
    def _simple_classification(self, audio: np.ndarray) -> Tuple[str, float]:
        """Simple classification based on basic audio features"""
        # Calculate basic features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Simple heuristics
        avg_mfcc = np.mean(mfccs, axis=1)
        avg_centroid = np.mean(spectral_centroid)
        avg_zcr = np.mean(zero_crossing_rate)
        
        # Classification based on feature patterns
        if avg_centroid > 3000 and avg_zcr > 0.1:
            return "rock", 0.4
        elif avg_centroid < 1500 and avg_zcr < 0.05:
            return "jazz", 0.4
        elif avg_centroid > 2500:
            return "pop", 0.4
        elif avg_zcr > 0.15:
            return "metal", 0.4
        else:
            return "pop", 0.3
    
    def analyze_audio_characteristics(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive audio analysis
        
        Args:
            audio: Input audio array
        
        Returns:
            Dictionary with audio characteristics
        """
        analysis = {}
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        analysis['tempo'] = float(tempo)
        analysis['beat_count'] = int(len(beats))
        
        # Key and mode
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        key = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        analysis['key'] = keys[key]
        
        # Energy and dynamics
        rms = librosa.feature.rms(y=audio)[0]
        analysis['energy'] = float(np.mean(rms))
        analysis['dynamics'] = float(np.std(rms))
        
        # Spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        analysis['brightness'] = float(np.mean(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        analysis['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        # Harmonic content
        harmonic, percussive = librosa.effects.hpss(audio)
        analysis['harmonic_ratio'] = float(np.mean(harmonic**2) / (np.mean(harmonic**2) + np.mean(percussive**2)))
        
        # Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        analysis['noisiness'] = float(np.mean(zcr))
        
        return analysis
    
    def process_audio_file_advanced(self, file_path: str, 
                                   equalizer_params: Dict = None,
                                   denoise_method: str = 'autoencoder',
                                   analyze: bool = True) -> Dict:
        """
        Advanced audio processing with comprehensive analysis
        
        Args:
            file_path: Path to audio file
            equalizer_params: Dictionary with equalizer parameters
            denoise_method: Method for noise reduction
            analyze: Whether to perform detailed analysis
        
        Returns:
            Dictionary with all processing results
        """
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Default equalizer parameters
        if equalizer_params is None:
            equalizer_params = {
                'bass_gain': 1.0,
                'mid_gain': 1.0,
                'treble_gain': 1.0,
                'sub_bass_gain': 1.0,
                'presence_gain': 1.0,
                'air_gain': 1.0
            }
        
        # Genre classification (on original audio for better accuracy)
        print(f"Debug - Genre classifier available: {self.genre_classifier is not None}")
        genre, confidence, additional_info = self.advanced_genre_classification(audio)
        print(f"Debug - Genre result: {genre}, Confidence: {confidence}")
        
        # Apply advanced equalizer
        processed_audio = self.advanced_equalizer(audio, **equalizer_params)
        
        # Apply noise reduction
        denoised_audio = self.advanced_noise_reduction(processed_audio, denoise_method)
        
        # Audio analysis
        analysis = {}
        if analyze:
            analysis = self.analyze_audio_characteristics(denoised_audio)
        
        results = {
            'original_audio': audio,
            'processed_audio': denoised_audio,
            'genre': genre,
            'confidence': confidence,
            'additional_info': additional_info,
            'analysis': analysis,
            'equalizer_params': equalizer_params,
            'denoise_method': denoise_method
        }
        
        return results
    
    def start_advanced_real_time_processing(self, callback, 
                                          equalizer_params: Dict = None,
                                          denoise_method: str = 'autoencoder'):
        """
        Start advanced real-time audio processing
        
        Args:
            callback: Function to call with processed audio data
            equalizer_params: Equalizer parameters
            denoise_method: Noise reduction method
        """
        if equalizer_params is None:
            equalizer_params = {
                'bass_gain': 1.0,
                'mid_gain': 1.0,
                'treble_gain': 1.0,
                'sub_bass_gain': 1.0,
                'presence_gain': 1.0,
                'air_gain': 1.0
            }
        
        self.is_recording = True
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                # Process audio chunk
                audio_chunk = indata[:, 0]  # Take first channel
                
                # Apply advanced equalizer
                processed_chunk = self.advanced_equalizer(audio_chunk, **equalizer_params)
                
                # Apply noise reduction
                denoised_chunk = self.advanced_noise_reduction(processed_chunk, denoise_method)
                
                # Genre classification
                genre, confidence, additional_info = self.advanced_genre_classification(denoised_chunk)
                
                # Send to callback
                callback({
                    'audio': denoised_chunk,
                    'genre': genre,
                    'confidence': confidence,
                    'additional_info': additional_info
                })
        
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=2048):  # Larger blocksize for better processing
            while self.is_recording:
                time.sleep(0.1)
    
    def stop_real_time_processing(self):
        """Stop real-time audio processing"""
        self.is_recording = False
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """Save audio to file"""
        import soundfile as sf
        sf.write(file_path, audio, self.sample_rate)
    
    def create_visualization(self, results: Dict, save_path: str = None):
        """
        Create comprehensive visualization of processing results
        
        Args:
            results: Results from process_audio_file_advanced
            save_path: Path to save visualization
        """
        try:
            # Set matplotlib backend to non-interactive
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            # Original vs Processed audio
            time_axis = np.linspace(0, len(results['original_audio'])/self.sample_rate, 
                                   len(results['original_audio']))
            
            # Limit display to first 5 seconds
            max_samples = min(22050 * 5, len(results['original_audio']))
            
            axes[0, 0].plot(time_axis[:max_samples], results['original_audio'][:max_samples], 
                           label='Original', alpha=0.7)
            axes[0, 0].set_title('Original Audio')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(time_axis[:max_samples], results['processed_audio'][:max_samples], 
                           label='Processed', alpha=0.7, color='orange')
            axes[0, 1].set_title('Processed Audio')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Frequency domain comparison
            fft_original = np.abs(np.fft.fft(results['original_audio']))
            fft_processed = np.abs(np.fft.fft(results['processed_audio']))
            freq_axis = np.fft.fftfreq(len(results['original_audio']), 1/self.sample_rate)
            
            positive_freqs = freq_axis > 0
            axes[1, 0].plot(freq_axis[positive_freqs], fft_original[positive_freqs], 
                           label='Original')
            axes[1, 0].set_title('Original Audio (Frequency Domain)')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_xlim(0, 5000)
            
            axes[1, 1].plot(freq_axis[positive_freqs], fft_processed[positive_freqs], 
                           label='Processed', color='orange')
            axes[1, 1].set_title('Processed Audio (Frequency Domain)')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_xlim(0, 5000)
            
            # Genre probabilities
            if 'additional_info' in results and 'all_probabilities' in results['additional_info']:
                genres = list(results['additional_info']['all_probabilities'].keys())
                probabilities = list(results['additional_info']['all_probabilities'].values())
                
                axes[2, 0].bar(genres, probabilities, color='skyblue')
                axes[2, 0].set_title('Genre Classification Probabilities')
                axes[2, 0].set_xlabel('Genre')
                axes[2, 0].set_ylabel('Probability')
                axes[2, 0].tick_params(axis='x', rotation=45)
                axes[2, 0].grid(True)
            
            # Audio characteristics
            if 'analysis' in results:
                analysis = results['analysis']
                characteristics = ['Tempo', 'Energy', 'Brightness', 'Harmonic Ratio']
                values = [analysis.get('tempo', 0), analysis.get('energy', 0), 
                         analysis.get('brightness', 0), analysis.get('harmonic_ratio', 0)]
                
                axes[2, 1].bar(characteristics, values, color='lightgreen')
                axes[2, 1].set_title('Audio Characteristics')
                axes[2, 1].set_ylabel('Value')
                axes[2, 1].grid(True)
        
            plt.tight_layout()
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Force draw the figure before saving
                fig.canvas.draw()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Visualization saved to: {save_path}")
            
            # Close the figure properly
            plt.close(fig)
            plt.close('all')
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            print("Continuing without visualization...")
            # Create a simple fallback visualization
            if save_path:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.text(0.5, 0.5, f'Audio Processing Completed\nGenre: {results.get("genre", "Unknown")}\nConfidence: {results.get("confidence", 0):.1%}', 
                           ha='center', va='center', fontsize=16)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.canvas.draw()
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    plt.close('all')
                    print(f"📊 Fallback visualization saved to: {save_path}")
                except Exception as fallback_error:
                    print(f"❌ Even fallback visualization failed: {fallback_error}")
                    pass

def main():
    """Test advanced audio processor"""
    processor = AdvancedAudioProcessor()
    
    # Test with a sample audio file if available
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        results = processor.process_audio_file_advanced(test_file)
        print(f"Genre: {results['genre']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Analysis: {results['analysis']}")
        
        # Create visualization
        processor.create_visualization(results, "advanced_analysis.png")
    else:
        print("Test audio file not found. Create a test_audio.wav file to test.")

if __name__ == "__main__":
    main() 