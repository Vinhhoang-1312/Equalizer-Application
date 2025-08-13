#!/usr/bin/env python3
"""
Genre Classification Engine Module
Phân loại thể loại nhạc bằng Machine Learning với nhiều phương pháp
"""

import numpy as np
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from typing import Dict, List, Tuple, Optional
import warnings
import os
from scipy import stats  # For skew and kurtosis
warnings.filterwarnings('ignore')

class GenreClassificationEngine:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Genre Classification Engine
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
        # Supported genres (can be extended)
        self.genres = [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
        # Models
        self.rf_model = None
        self.svm_model = None  
        self.nn_model = None
        self.lstm_model = None
        self.cnn_model = None
        
        # Scalers and encoders
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_chroma = 12
        self.n_mel = 128
        
        # Import advanced classifier with 3 options
        try:
            from .advanced_genre_classifier import AdvancedGenreClassifier
            self.advanced_classifier = AdvancedGenreClassifier(sample_rate)
            print("✓ Advanced Genre Classifier loaded (3 options)")
        except Exception as e:
            print(f"⚠️ Advanced classifier not available: {e}")
            self.advanced_classifier = None
        
        # Try to load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models and scalers"""
        try:
            # Load traditional ML models
            if os.path.exists('models/genre_classifier.pkl'):
                self.rf_model = joblib.load('models/genre_classifier.pkl')
                print("✓ Random Forest genre classifier loaded")
            
            if os.path.exists('models/svm_genre_classifier.pkl'):
                self.svm_model = joblib.load('models/svm_genre_classifier.pkl')
                print("✓ SVM genre classifier loaded")
            
            if os.path.exists('models/nn_genre_classifier.pkl'):
                self.nn_model = joblib.load('models/nn_genre_classifier.pkl')
                print("✓ Neural Network genre classifier loaded")
            
            # Load deep learning models
            if os.path.exists('models/lstm_genre_classifier.h5'):
                self.lstm_model = load_model('models/lstm_genre_classifier.h5')
                print("✓ LSTM genre classifier loaded")
            
            if os.path.exists('models/cnn_genre_classifier.h5'):
                self.cnn_model = load_model('models/cnn_genre_classifier.h5')
                print("✓ CNN genre classifier loaded")
            
            # Load scalers
            if os.path.exists('models/feature_scaler.pkl'):
                self.feature_scaler = joblib.load('models/feature_scaler.pkl')
                print("✓ Feature scaler loaded")
            
            if os.path.exists('models/label_encoder.pkl'):
                self.label_encoder = joblib.load('models/label_encoder.pkl')
                self.genres = list(self.label_encoder.classes_)
                print("✓ Label encoder loaded")
                
            self.is_fitted = True
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize fresh models"""
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=20,
            min_samples_split=5
        )
        
        self.svm_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            gamma='scale'
        )
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.label_encoder.fit(self.genres)
    
    def extract_basic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract basic audio features for classification
        
        Args:
            audio: Input audio signal
            
        Returns:
            Feature vector
        """
        features = []
        
        try:
            # MFCC features (most important for genre classification)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, n_chroma=self.n_chroma)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            features.extend(chroma_mean)
            features.extend(chroma_std)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Tempo and beat features
            try:
                tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
                features.append(tempo)
                features.append(len(beats) / (len(audio) / self.sample_rate))  # Beat density
            except:
                features.extend([120.0, 2.0])  # Default values
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            features.extend(np.mean(contrast, axis=1))
            
            # Tonnetz (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            features.extend(np.mean(tonnetz, axis=1))
            
        except Exception as e:
            print(f"⚠️ Error extracting features: {e}")
            # Return zero vector if feature extraction fails
            return np.zeros(90)  # Approximate expected feature size
        
        return np.array(features)
    
    def extract_mel_spectrogram(self, audio: np.ndarray, 
                               max_length: int = 432) -> np.ndarray:
        """
        Extract mel-spectrogram for deep learning models
        
        Args:
            audio: Input audio signal
            max_length: Maximum length of spectrogram
            
        Returns:
            Mel-spectrogram array
        """
        try:
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_mels=self.n_mel,
                hop_length=512
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
            
            # Pad or truncate to fixed size
            if mel_spec_norm.shape[1] > max_length:
                mel_spec_norm = mel_spec_norm[:, :max_length]
            else:
                pad_width = max_length - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
            
            return mel_spec_norm
            
        except Exception as e:
            print(f"⚠️ Error extracting mel-spectrogram: {e}")
            return np.zeros((self.n_mel, max_length))
    
    def classify_with_traditional_ml(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Classify genre using traditional ML models (Random Forest, SVM)
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (genre, confidence, additional_info)
        """
        try:
            # Extract features
            features = self.extract_basic_features(audio)
            
            if not self.is_fitted:
                return "unknown", 0.1, {'method': 'traditional_ml', 'error': 'Models not trained'}
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Get predictions from available models
            predictions = {}
            probabilities = {}
            
            if self.rf_model is not None:
                rf_pred = self.rf_model.predict(features_scaled)[0]
                rf_proba = self.rf_model.predict_proba(features_scaled)[0]
                predictions['random_forest'] = self.label_encoder.inverse_transform([rf_pred])[0]
                probabilities['random_forest'] = dict(zip(self.genres, rf_proba))
            
            if self.svm_model is not None:
                svm_pred = self.svm_model.predict(features_scaled)[0]
                svm_proba = self.svm_model.predict_proba(features_scaled)[0]
                predictions['svm'] = self.label_encoder.inverse_transform([svm_pred])[0]
                probabilities['svm'] = dict(zip(self.genres, svm_proba))
            
            if self.nn_model is not None:
                nn_pred = self.nn_model.predict(features_scaled)[0]
                nn_proba = self.nn_model.predict_proba(features_scaled)[0]
                predictions['neural_network'] = self.label_encoder.inverse_transform([nn_pred])[0]
                probabilities['neural_network'] = dict(zip(self.genres, nn_proba))
            
            # Ensemble prediction (majority vote with confidence weighting)
            if predictions:
                # Weight by maximum probability
                weighted_votes = {}
                for model_name, genre in predictions.items():
                    max_prob = max(probabilities[model_name].values())
                    if genre not in weighted_votes:
                        weighted_votes[genre] = 0
                    weighted_votes[genre] += max_prob
                
                # Get most confident prediction
                best_genre = max(weighted_votes, key=weighted_votes.get)
                confidence = weighted_votes[best_genre] / len(predictions)
                
                additional_info = {
                    'method': 'traditional_ml_ensemble',
                    'individual_predictions': predictions,
                    'individual_probabilities': probabilities,
                    'ensemble_weights': weighted_votes,
                    'feature_vector': features.tolist()
                }
                
                return best_genre, confidence, additional_info
            else:
                return "pop", 0.2, {'method': 'traditional_ml', 'error': 'No models available'}
                
        except Exception as e:
            print(f"⚠️ Traditional ML classification failed: {e}")
            return "pop", 0.1, {'method': 'traditional_ml', 'error': str(e)}
    
    def classify_with_deep_learning(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Classify genre using deep learning models (LSTM, CNN)
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (genre, confidence, additional_info)
        """
        try:
            predictions = {}
            probabilities = {}
            
            # CNN prediction using mel-spectrogram
            if self.cnn_model is not None:
                mel_spec = self.extract_mel_spectrogram(audio)
                mel_input = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
                
                cnn_proba = self.cnn_model.predict(mel_input, verbose=0)[0]
                cnn_pred_idx = np.argmax(cnn_proba)
                cnn_pred = self.genres[cnn_pred_idx]
                
                predictions['cnn'] = cnn_pred
                probabilities['cnn'] = dict(zip(self.genres, cnn_proba))
            
            # LSTM prediction using sequential features
            if self.lstm_model is not None:
                # Create time series of features
                frame_length = 2048
                hop_length = 512
                frames = librosa.util.frame(audio, frame_length=frame_length, 
                                          hop_length=hop_length, axis=0)
                
                if len(frames) > 0:
                    # Extract features for each frame
                    frame_features = []
                    for frame in frames[:50]:  # Limit to 50 frames
                        features = self.extract_basic_features(frame)
                        frame_features.append(features)
                    
                    # Pad or truncate to fixed length
                    if len(frame_features) < 50:
                        # Pad with zeros
                        while len(frame_features) < 50:
                            frame_features.append(np.zeros_like(frame_features[0]))
                    
                    frame_features = np.array(frame_features)
                    
                    # Normalize features
                    frame_features = (frame_features - np.mean(frame_features, axis=0)) / (np.std(frame_features, axis=0) + 1e-8)
                    
                    lstm_input = frame_features.reshape(1, frame_features.shape[0], frame_features.shape[1])
                    
                    lstm_proba = self.lstm_model.predict(lstm_input, verbose=0)[0]
                    lstm_pred_idx = np.argmax(lstm_proba)
                    lstm_pred = self.genres[lstm_pred_idx]
                    
                    predictions['lstm'] = lstm_pred
                    probabilities['lstm'] = dict(zip(self.genres, lstm_proba))
            
            # Ensemble deep learning predictions
            if predictions:
                # Average probabilities
                avg_probabilities = {}
                for genre in self.genres:
                    probs = [probabilities[model][genre] for model in probabilities.keys()]
                    avg_probabilities[genre] = np.mean(probs)
                
                best_genre = max(avg_probabilities, key=avg_probabilities.get)
                confidence = avg_probabilities[best_genre]
                
                additional_info = {
                    'method': 'deep_learning_ensemble',
                    'individual_predictions': predictions,
                    'individual_probabilities': probabilities,
                    'ensemble_probabilities': avg_probabilities
                }
                
                return best_genre, confidence, additional_info
            else:
                return "pop", 0.2, {'method': 'deep_learning', 'error': 'No models available'}
                
        except Exception as e:
            print(f"⚠️ Deep learning classification failed: {e}")
            return "pop", 0.1, {'method': 'deep_learning', 'error': str(e)}
    
    def classify_with_rule_based(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Classify genre using rule-based approach
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (genre, confidence, additional_info)
        """
        try:
            # Extract key features for rule-based classification
            features = self.extract_basic_features(audio)
            
            # Get tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            except:
                tempo = 120  # Default
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # MFCC analysis
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Chroma analysis for harmonic content
            chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
            chroma_std = np.std(chroma)
            
            # Energy analysis
            rms = np.mean(librosa.feature.rms(y=audio))
            
            # Rule-based classification
            scores = {}
            
            # Classical: Low tempo, high harmonic content, low energy
            if tempo < 100 and chroma_std > 0.2 and rms < 0.05:
                scores['classical'] = 0.8
            elif spectral_centroid < 2000 and chroma_std > 0.15:
                scores['classical'] = 0.4
            
            # Jazz: Moderate tempo, high harmonic variation, complex rhythm
            if 80 < tempo < 140 and chroma_std > 0.25 and zero_crossing_rate < 0.1:
                scores['jazz'] = 0.7
            elif chroma_std > 0.2 and 1500 < spectral_centroid < 3000:
                scores['jazz'] = 0.4
            
            # Rock: High tempo, high energy, bright spectrum
            if tempo > 120 and rms > 0.03 and spectral_centroid > 2500:
                scores['rock'] = 0.6
            elif rms > 0.04 and zero_crossing_rate > 0.08:
                scores['rock'] = 0.4
            
            # Metal: Very high tempo, very high energy, very bright
            if tempo > 140 and rms > 0.05 and spectral_centroid > 3000:
                scores['metal'] = 0.7
            elif rms > 0.06 and zero_crossing_rate > 0.12:
                scores['metal'] = 0.5
            
            # Pop: Moderate tempo, moderate energy, balanced spectrum
            if 100 < tempo < 130 and 0.02 < rms < 0.04 and 2000 < spectral_centroid < 3500:
                scores['pop'] = 0.5
            
            # Hip-hop: Moderate to high tempo, strong bass, rhythmic
            if tempo > 90 and mfcc_mean[0] > -20 and zero_crossing_rate < 0.08:
                scores['hiphop'] = 0.5
            
            # Disco: High tempo, danceable, bright
            if tempo > 110 and rms > 0.03 and spectral_centroid > 2000:
                scores['disco'] = 0.4
            
            # Country: Moderate tempo, mid-range spectrum
            if 90 < tempo < 120 and 1500 < spectral_centroid < 2500:
                scores['country'] = 0.4
            
            # Blues: Slow to moderate tempo, low spectrum
            if tempo < 100 and spectral_centroid < 2000 and chroma_std < 0.15:
                scores['blues'] = 0.4
            
            # Reggae: Moderate tempo, specific rhythm pattern
            if 80 < tempo < 110 and chroma_std > 0.15:
                scores['reggae'] = 0.3
            
            # Default to pop if no strong classification
            if not scores:
                scores['pop'] = 0.3
            
            best_genre = max(scores, key=scores.get)
            confidence = scores[best_genre]
            
            analysis = {
                'tempo': float(tempo),
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zero_crossing_rate),
                'chroma_std': float(chroma_std),
                'rms': float(rms),
                'mfcc_mean': mfcc_mean.tolist()
            }
            
            additional_info = {
                'method': 'rule_based',
                'scores': scores,
                'audio_analysis': analysis
            }
            
            return best_genre, confidence, additional_info
            
        except Exception as e:
            print(f"⚠️ Rule-based classification failed: {e}")
            return "pop", 0.1, {'method': 'rule_based', 'error': str(e)}
    
    def classify_genre(self, audio: np.ndarray, method: str = 'ensemble') -> Tuple[str, float, Dict]:
        """
        Classify audio genre using specified method
        
        Args:
            audio: Input audio signal
            method: Classification method ('ensemble', 'traditional', 'deep_learning', 'rule_based')
            
        Returns:
            Tuple of (genre, confidence, additional_info)
        """
        if method == 'traditional':
            return self.classify_with_traditional_ml(audio)
        elif method == 'deep_learning':
            return self.classify_with_deep_learning(audio)
        elif method == 'rule_based':
            return self.classify_with_rule_based(audio)
        elif method == 'ensemble':
            # Combine all methods
            results = {}
            
            # Get results from all methods
            traditional_result = self.classify_with_traditional_ml(audio)
            deep_result = self.classify_with_deep_learning(audio)
            rule_result = self.classify_with_rule_based(audio)
            
            results['traditional'] = traditional_result
            results['deep_learning'] = deep_result
            results['rule_based'] = rule_result
            
            # Weight by confidence and combine
            genre_scores = {}
            for method_name, (genre, confidence, info) in results.items():
                if genre not in genre_scores:
                    genre_scores[genre] = 0
                genre_scores[genre] += confidence
            
            # Average scores
            for genre in genre_scores:
                genre_scores[genre] /= len(results)
            
            best_genre = max(genre_scores, key=genre_scores.get)
            final_confidence = genre_scores[best_genre]
            
            additional_info = {
                'method': 'ensemble',
                'individual_results': results,
                'final_scores': genre_scores
            }
            
            return best_genre, final_confidence, additional_info
        else:
            return "pop", 0.1, {'method': 'unknown', 'error': f'Unknown method: {method}'}
    
    def process_audio_file(self, input_path: str, method: str = 'ensemble') -> Dict:
        """
        Process audio file for genre classification
        
        Args:
            input_path: Path to input audio file
            method: Classification method
            
        Returns:
            Dictionary with classification results
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Classify genre
        genre, confidence, additional_info = self.classify_genre(audio, method)
        
        # Additional audio analysis
        duration = len(audio) / self.sample_rate
        rms = np.sqrt(np.mean(audio**2))
        
        return {
            'input_path': input_path,
            'predicted_genre': genre,
            'confidence': float(confidence),
            'method': method,
            'additional_info': additional_info,
            'audio_stats': {
                'duration': float(duration),
                'rms_level': float(rms),
                'sample_rate': self.sample_rate
            }
        }
    
    def get_available_genres(self) -> List[str]:
        """Get list of supported genres"""
        return self.genres.copy()
    
    def get_available_methods(self) -> List[str]:
        """Get list of available classification methods"""
        return ['ensemble', 'traditional', 'deep_learning', 'rule_based']
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'genres_supported': len(self.genres),
            'genres_list': self.genres,
            'models_loaded': {
                'random_forest': self.rf_model is not None,
                'svm': self.svm_model is not None,
                'neural_network': self.nn_model is not None,
                'lstm': self.lstm_model is not None,
                'cnn': self.cnn_model is not None,
                'advanced_classifier': self.advanced_classifier is not None
            },
            'is_fitted': self.is_fitted,
            'sample_rate': self.sample_rate
        }
        
        return info

    def classify_with_best_options(self, input_path: str) -> Dict:
        """
        🎯 NEW METHOD: Classify using 2 BEST options (no Spotify)
        1. Musicnn Deep Learning (92% accuracy) 
        2. Custom ML/GTZAN (87% accuracy)
        """
        if not self.advanced_classifier:
            return {
                'error': 'Advanced classifier not available',
                'message': 'Advanced genre classifier module not loaded'
            }
            
        try:
            print("🎵 Running BEST Genre Classification Methods...")
            results = self.advanced_classifier.classify_both_methods(input_path)
            
            # Format results for API response
            formatted_results = {
                'status': 'success',
                'input_file': input_path,
                'total_methods': results['total_methods'],
                'successful_methods': results['successful_methods'],
                'recommendation': results['recommendation'],
                'detailed_results': {}
            }
            
            # Format each method result
            for method_name, method_result in results['results'].items():
                method_display_names = {
                    'musicnn': '🧠 Musicnn Deep Learning', 
                    'custom_ml': '🤖 Custom ML (GTZAN)'
                }
                
                display_name = method_display_names.get(method_name, method_name)
                formatted_results['detailed_results'][display_name] = method_result
            
            return formatted_results
            
        except Exception as e:
            return {
                'error': 'Classification failed',
                'message': str(e)
            }
