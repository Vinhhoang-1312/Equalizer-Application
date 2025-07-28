import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob
from tqdm import tqdm
import zipfile
import urllib.request

class ModelTrainer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
        
    def download_gtzan_dataset(self):
        """Download GTZAN dataset if not available"""
        if not os.path.exists('data/gtzan'):
            print("Downloading GTZAN dataset...")
            os.makedirs('data', exist_ok=True)
            
            # Download from Kaggle (you need to have kaggle CLI installed)
            # For now, we'll create a synthetic dataset
            self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset for demonstration"""
        print("Creating synthetic dataset...")
        os.makedirs('data/gtzan', exist_ok=True)
        
        for genre in self.genres:
            os.makedirs(f'data/gtzan/{genre}', exist_ok=True)
            
            # Create synthetic audio files for each genre
            for i in range(50):  # 50 files per genre
                # Generate different frequency patterns for each genre
                duration = 30  # 30 seconds
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                
                if genre == 'classical':
                    # Classical: harmonic frequencies
                    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
                elif genre == 'jazz':
                    # Jazz: complex harmonics
                    audio = np.sin(2 * np.pi * 220 * t) + 0.7 * np.sin(2 * np.pi * 330 * t)
                elif genre == 'rock':
                    # Rock: distorted sound
                    audio = np.sin(2 * np.pi * 110 * t) * 0.8
                    audio = np.clip(audio * 1.5, -1, 1)
                elif genre == 'pop':
                    # Pop: clean frequencies
                    audio = np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
                else:
                    # Other genres: random frequencies
                    freq = np.random.uniform(200, 800)
                    audio = np.sin(2 * np.pi * freq * t)
                
                # Add some noise
                noise = np.random.normal(0, 0.1, len(audio))
                audio = audio + noise
                
                # Normalize
                audio = audio / np.max(np.abs(audio))
                
                # Save file
                import soundfile as sf
                sf.write(f'data/gtzan/{genre}/{genre}_{i:03d}.wav', audio, self.sample_rate)
    
    def extract_features_from_dataset(self):
        """Extract features from all audio files"""
        print("Extracting features from dataset...")
        
        features_list = []
        labels_list = []
        
        for genre in self.genres:
            genre_path = f'data/gtzan/{genre}'
            if not os.path.exists(genre_path):
                continue
                
            audio_files = glob.glob(f'{genre_path}/*.wav')
            
            for audio_file in tqdm(audio_files, desc=f"Processing {genre}"):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                    
                    # Extract features
                    features = self.extract_features(audio)
                    
                    features_list.append(features)
                    labels_list.append(genre)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        return np.array(features_list), np.array(labels_list)
    
    def extract_features(self, audio):
        """Extract audio features"""
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
    
    def train_genre_classifier(self):
        """Train genre classification model"""
        print("Training genre classifier...")
        
        # Extract features
        features, labels = self.extract_features_from_dataset()
        
        if len(features) == 0:
            print("No features extracted. Check dataset.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = classifier.score(X_train_scaled, y_train)
        test_score = classifier.score(X_test_scaled, y_test)
        
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(classifier, 'models/genre_classifier.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        return classifier, scaler
    
    def create_noise_reduction_dataset(self):
        """Create dataset for noise reduction training"""
        print("Creating noise reduction dataset...")
        
        clean_audio_list = []
        noisy_audio_list = []
        
        # Load clean audio files
        for genre in self.genres:
            genre_path = f'data/gtzan/{genre}'
            if not os.path.exists(genre_path):
                continue
                
            audio_files = glob.glob(f'{genre_path}/*.wav')
            
            for audio_file in tqdm(audio_files[:10], desc=f"Processing {genre}"):  # Use first 10 files
                try:
                    # Load clean audio
                    clean_audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                    
                    # Add different types of noise
                    for noise_type in ['white', 'pink', 'brown']:
                        # Create noisy version
                        if noise_type == 'white':
                            noise = np.random.normal(0, 0.1, len(clean_audio))
                        elif noise_type == 'pink':
                            # Pink noise approximation
                            noise = np.random.normal(0, 1, len(clean_audio))
                            noise = np.convolve(noise, np.ones(100)/100, mode='same')
                        else:  # brown
                            noise = np.cumsum(np.random.normal(0, 0.01, len(clean_audio)))
                        
                        noisy_audio = clean_audio + noise * 0.3
                        
                        # Normalize
                        clean_audio_norm = clean_audio / np.max(np.abs(clean_audio))
                        noisy_audio_norm = noisy_audio / np.max(np.abs(noisy_audio))
                        
                        clean_audio_list.append(clean_audio_norm)
                        noisy_audio_list.append(noisy_audio_norm)
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        return clean_audio_list, noisy_audio_list
    
    def train_noise_reducer(self):
        """Train noise reduction model"""
        print("Training noise reducer...")
        
        # Create dataset
        clean_audio_list, noisy_audio_list = self.create_noise_reduction_dataset()
        
        if len(clean_audio_list) == 0:
            print("No audio data for noise reduction training.")
            return
        
        # Prepare data for training
        X_train = []
        y_train = []
        
        for clean_audio, noisy_audio in zip(clean_audio_list, noisy_audio_list):
            # Convert to spectrograms
            clean_stft = librosa.stft(clean_audio)
            noisy_stft = librosa.stft(noisy_audio)
            
            clean_magnitude = np.abs(clean_stft)
            noisy_magnitude = np.abs(noisy_stft)
            
            # Normalize
            clean_magnitude_norm = clean_magnitude / np.max(clean_magnitude)
            noisy_magnitude_norm = noisy_magnitude / np.max(noisy_magnitude)
            
            X_train.append(noisy_magnitude_norm)
            y_train.append(clean_magnitude_norm)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Add channel dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
        
        # Create autoencoder model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save('models/noise_reducer.h5')
        
        return model
    
    def train_all_models(self):
        """Train all models"""
        print("Starting model training...")
        
        # Download/create dataset
        self.download_gtzan_dataset()
        
        # Train genre classifier
        self.train_genre_classifier()
        
        # Train noise reducer
        self.train_noise_reducer()
        
        print("Model training completed!")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models() 