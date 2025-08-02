"""
Advanced Model Trainer
Huấn luyện các mô hình nâng cao cho phân loại thể loại nhạc và giảm nhiễu
"""

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        # Model paths
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Advanced features configuration
        self.feature_config = {
            'mfcc_coeffs': 13,
            'mfcc_delta': True,
            'mfcc_delta2': True,
            'spectral_features': True,
            'chroma_features': True,
            'tonnetz_features': True,
            'rhythm_features': True,
            'harmonic_features': True
        }
    
    def extract_advanced_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Trích xuất đặc trưng âm thanh nâng cao
        
        Args:
            audio: Audio array
        
        Returns:
            Feature vector
        """
        features = []
        
        # MFCC features (26 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend([np.mean(mfccs[i]) for i in range(13)])
        features.extend([np.std(mfccs[i]) for i in range(13)])
        
        # Delta and Delta-Delta MFCC (26 features)
        if self.feature_config['mfcc_delta']:
            mfcc_delta = librosa.feature.delta(mfccs)
            features.extend([np.mean(mfcc_delta[i]) for i in range(13)])
        
        if self.feature_config['mfcc_delta2']:
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features.extend([np.mean(mfcc_delta2[i]) for i in range(13)])
        
        # Spectral features (12 features)
        if self.feature_config['spectral_features']:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            features.append(np.mean(spectral_contrast))
            features.append(np.std(spectral_contrast))
            
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features.append(np.mean(spectral_flatness))
            features.append(np.std(spectral_flatness))
            
            spectral_poly_features = librosa.feature.poly_features(y=audio, sr=self.sample_rate)
            features.append(np.mean(spectral_poly_features))
            features.append(np.std(spectral_poly_features))
        
        # Chroma features (24 features)
        if self.feature_config['chroma_features']:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features.extend([np.mean(chroma[i]) for i in range(12)])
            features.extend([np.std(chroma[i]) for i in range(12)])
        
        # Tonnetz features (6 features)
        if self.feature_config['tonnetz_features']:
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            features.extend([np.mean(tonnetz[i]) for i in range(6)])
        
        # Rhythm features (8 features)
        if self.feature_config['rhythm_features']:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features.append(tempo)
            
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            features.append(np.mean(onset_env))
            features.append(np.std(onset_env))
            
            # Beat features
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                features.append(np.mean(beat_intervals))
                features.append(np.std(beat_intervals))
            else:
                features.extend([0, 0])
            
            # Tempo features
            tempo_curve = librosa.beat.tempo(y=audio, sr=self.sample_rate, aggregate=None)
            features.append(np.mean(tempo_curve))
            features.append(np.std(tempo_curve))
        
        # Harmonic and percussive components (4 features)
        if self.feature_config['harmonic_features']:
            harmonic, percussive = librosa.effects.hpss(audio)
            features.append(np.mean(harmonic))
            features.append(np.std(harmonic))
            features.append(np.mean(percussive))
            features.append(np.std(percussive))
        
        # Zero crossing rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # Root mean square energy (2 features)
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        return np.array(features)
    
    def create_synthetic_dataset(self, samples_per_genre: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset tổng hợp cho training
        
        Args:
            samples_per_genre: Số lượng mẫu cho mỗi thể loại
        
        Returns:
            Tuple of (features, labels)
        """
        print("🎵 Tạo dataset tổng hợp...")
        
        features_list = []
        labels_list = []
        
        for genre in self.genres:
            print(f"  Tạo {samples_per_genre} mẫu cho {genre}")
            
            for i in range(samples_per_genre):
                # Tạo audio tổng hợp với đặc trưng của từng thể loại
                duration = 30  # 30 giây
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                
                # Tạo audio dựa trên đặc trưng thể loại
                audio = self._generate_genre_audio(genre, t)
                
                # Trích xuất đặc trưng
                features = self.extract_advanced_features(audio)
                features_list.append(features)
                labels_list.append(genre)
        
        return np.array(features_list), np.array(labels_list)
    
    def _generate_genre_audio(self, genre: str, t: np.ndarray) -> np.ndarray:
        """Tạo audio tổng hợp cho từng thể loại"""
        if genre == 'classical':
            # Classical: harmonic frequencies, clean
            audio = (np.sin(2 * np.pi * 440 * t) + 
                    0.5 * np.sin(2 * np.pi * 880 * t) +
                    0.3 * np.sin(2 * np.pi * 660 * t))
        elif genre == 'jazz':
            # Jazz: complex harmonics, swing
            audio = (np.sin(2 * np.pi * 220 * t) + 
                    0.7 * np.sin(2 * np.pi * 330 * t) +
                    0.4 * np.sin(2 * np.pi * 550 * t))
        elif genre == 'rock':
            # Rock: distorted, heavy bass
            audio = np.sin(2 * np.pi * 110 * t) * 0.8
            audio = np.clip(audio * 1.5, -1, 1)
        elif genre == 'pop':
            # Pop: clean, melodic
            audio = (np.sin(2 * np.pi * 330 * t) + 
                    0.3 * np.sin(2 * np.pi * 660 * t) +
                    0.2 * np.sin(2 * np.pi * 990 * t))
        elif genre == 'electronic':
            # Electronic: synthetic sounds
            audio = np.sin(2 * np.pi * 440 * t) * np.exp(-t/10)
        else:
            # Other genres: random frequencies
            freq = np.random.uniform(200, 800)
            audio = np.sin(2 * np.pi * freq * t)
        
        # Thêm nhiễu và hiệu ứng
        noise = np.random.normal(0, 0.05, len(audio))
        audio = audio + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def train_ensemble_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Huấn luyện ensemble classifier
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Dictionary with trained models and results
        """
        print("🎵 Huấn luyện ensemble classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {}
        results = {}
        
        # Random Forest
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_score = rf.score(X_test_scaled, y_test)
        models['random_forest'] = rf
        results['random_forest'] = rf_score
        
        # Gradient Boosting
        print("  Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_score = gb.score(X_test_scaled, y_test)
        models['gradient_boosting'] = gb
        results['gradient_boosting'] = gb_score
        
        # SVM
        print("  Training SVM...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
        svm.fit(X_train_scaled, y_train)
        svm_score = svm.score(X_test_scaled, y_test)
        models['svm'] = svm
        results['svm'] = svm_score
        
        # Neural Network
        print("  Training Neural Network...")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        mlp_score = mlp.score(X_test_scaled, y_test)
        models['mlp'] = mlp
        results['mlp'] = mlp_score
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        best_score = results[best_model_name]
        
        print(f"✅ Best model: {best_model_name} (Accuracy: {best_score:.3f})")
        
        # Save models
        joblib.dump(best_model, os.path.join(self.model_dir, 'advanced_genre_classifier.pkl'))
        joblib.dump(scaler, os.path.join(self.model_dir, 'advanced_scaler.pkl'))
        joblib.dump(scaler, os.path.join(self.model_dir, 'feature_scaler.pkl'))
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test_scaled)
        classification_rep = classification_report(y_test, y_pred, target_names=self.genres)
        
        return {
            'models': models,
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'best_score': best_score,
            'scaler': scaler,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def create_autoencoder(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Tạo autoencoder cho giảm nhiễu
        
        Args:
            input_shape: Shape of input spectrogram
        
        Returns:
            Autoencoder model
        """
        # Encoder
        encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
        ])
        
        # Decoder
        decoder = models.Sequential([
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        
        # Autoencoder
        autoencoder = models.Sequential([encoder, decoder])
        
        # Compile
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def create_noise_reduction_dataset(self, clean_audio_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset cho giảm nhiễu
        
        Args:
            clean_audio_list: List of clean audio samples
        
        Returns:
            Tuple of (noisy_spectrograms, clean_spectrograms)
        """
        print("🎵 Tạo dataset giảm nhiễu...")
        
        noisy_spectrograms = []
        clean_spectrograms = []
        
        for clean_audio in clean_audio_list:
            # Tạo nhiễu
            noise_types = ['white', 'pink', 'brown', 'gaussian']
            
            for noise_type in noise_types:
                # Tạo nhiễu
                if noise_type == 'white':
                    noise = np.random.normal(0, 0.1, len(clean_audio))
                elif noise_type == 'pink':
                    # Pink noise approximation
                    noise = np.random.normal(0, 1, len(clean_audio))
                    noise = np.convolve(noise, np.ones(100)/100, mode='same')
                elif noise_type == 'brown':
                    noise = np.cumsum(np.random.normal(0, 0.01, len(clean_audio)))
                else:  # gaussian
                    noise = np.random.normal(0, 0.05, len(clean_audio))
                
                # Thêm nhiễu
                noisy_audio = clean_audio + noise * 0.3
                
                # Convert to spectrograms
                clean_stft = librosa.stft(clean_audio)
                noisy_stft = librosa.stft(noisy_audio)
                
                clean_magnitude = np.abs(clean_stft)
                noisy_magnitude = np.abs(noisy_stft)
                
                # Normalize
                clean_magnitude_norm = clean_magnitude / np.max(clean_magnitude)
                noisy_magnitude_norm = noisy_magnitude / np.max(noisy_magnitude)
                
                # Resize to fixed size
                target_height, target_width = 1025, 1292
                
                if clean_magnitude_norm.shape[0] >= target_height and clean_magnitude_norm.shape[1] >= target_width:
                    clean_magnitude_norm = clean_magnitude_norm[:target_height, :target_width]
                    noisy_magnitude_norm = noisy_magnitude_norm[:target_height, :target_width]
                    
                    clean_spectrograms.append(clean_magnitude_norm)
                    noisy_spectrograms.append(noisy_magnitude_norm)
        
        return np.array(noisy_spectrograms), np.array(clean_spectrograms)
    
    def train_noise_reducer(self, clean_audio_list: List[np.ndarray]) -> tf.keras.Model:
        """
        Huấn luyện mô hình giảm nhiễu
        
        Args:
            clean_audio_list: List of clean audio samples
        
        Returns:
            Trained noise reduction model
        """
        print("🎵 Huấn luyện mô hình giảm nhiễu...")
        
        # Create dataset
        noisy_spectrograms, clean_spectrograms = self.create_noise_reduction_dataset(clean_audio_list)
        
        if len(noisy_spectrograms) == 0:
            print("❌ Không có dữ liệu cho training")
            return None
        
        # Add channel dimension
        noisy_spectrograms = noisy_spectrograms.reshape(
            noisy_spectrograms.shape[0], 
            noisy_spectrograms.shape[1], 
            noisy_spectrograms.shape[2], 
            1
        )
        clean_spectrograms = clean_spectrograms.reshape(
            clean_spectrograms.shape[0], 
            clean_spectrograms.shape[1], 
            clean_spectrograms.shape[2], 
            1
        )
        
        # Create model
        input_shape = (noisy_spectrograms.shape[1], noisy_spectrograms.shape[2], 1)
        autoencoder = self.create_autoencoder(input_shape)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_noise_reducer.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history = autoencoder.fit(
            noisy_spectrograms, clean_spectrograms,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        autoencoder.save(os.path.join(self.model_dir, 'advanced_noise_reducer.h5'))
        
        print("✅ Mô hình giảm nhiễu đã được huấn luyện")
        return autoencoder
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Đánh giá các mô hình
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Evaluation results
        """
        print("📊 Đánh giá mô hình...")
        
        # Load best model
        best_model = joblib.load(os.path.join(self.model_dir, 'advanced_genre_classifier.pkl'))
        scaler = joblib.load(os.path.join(self.model_dir, 'advanced_scaler.pkl'))
        
        # Cross-validation
        X_scaled = scaler.transform(X)
        cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
        
        # Final evaluation
        y_pred = best_model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=self.genres)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.genres, yticklabels=self.genres)
        plt.title('Confusion Matrix - Genre Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        results = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm
        }
        
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"✅ Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def train_all_models(self):
        """Huấn luyện tất cả mô hình"""
        print("🎵 Bắt đầu huấn luyện tất cả mô hình...")
        
        # Create synthetic dataset
        X, y = self.create_synthetic_dataset(samples_per_genre=200)
        
        # Train ensemble classifier
        classifier_results = self.train_ensemble_classifier(X, y)
        
        # Create clean audio samples for noise reduction
        clean_audio_list = []
        for genre in self.genres[:5]:  # Use first 5 genres
            for i in range(20):
                duration = 10
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                audio = self._generate_genre_audio(genre, t)
                clean_audio_list.append(audio)
        
        # Train noise reducer
        noise_reducer = self.train_noise_reducer(clean_audio_list)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X, y)
        
        print("🎉 Huấn luyện hoàn tất!")
        print(f"📊 Kết quả cuối cùng:")
        print(f"  - Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"  - Cross-validation: {evaluation_results['cv_mean']:.3f}")
        
        return {
            'classifier_results': classifier_results,
            'noise_reducer': noise_reducer,
            'evaluation_results': evaluation_results
        }

def main():
    """Test advanced model trainer"""
    trainer = AdvancedModelTrainer()
    results = trainer.train_all_models()
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main() 