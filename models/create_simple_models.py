import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def create_missing_models():
    """Create missing models for AdvancedAudioProcessor"""
    print("Creating missing models for AdvancedAudioProcessor...")
    
    # Check if basic models exist
    if os.path.exists('models/genre_classifier.pkl') and os.path.exists('models/scaler.pkl'):
        print("Found existing models, creating compatible versions...")
        
        # Load existing models
        genre_classifier = joblib.load('models/genre_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Save with advanced names
        joblib.dump(genre_classifier, 'models/advanced_genre_classifier.pkl')
        joblib.dump(scaler, 'models/advanced_scaler.pkl')
        
        # Create feature scaler (copy of existing scaler)
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        
        print("✓ Created advanced_genre_classifier.pkl")
        print("✓ Created advanced_scaler.pkl")
        print("✓ Created feature_scaler.pkl")
    
    # Create simple noise reducer model
    if not os.path.exists('models/advanced_noise_reducer.h5'):
        print("Creating simple noise reducer model...")
        
        # Simple autoencoder for noise reduction
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1025, 1292, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.save('models/advanced_noise_reducer.h5')
        print("✓ Created advanced_noise_reducer.h5")
    
    print("All required models created successfully!")

if __name__ == "__main__":
    create_missing_models() 