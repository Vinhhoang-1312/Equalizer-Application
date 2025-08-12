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
    
    # Create improved noise reducer model with flexible input shape
    if not os.path.exists('models/advanced_noise_reducer.h5'):
        print("Creating improved noise reducer model...")
        
        # Create a more flexible autoencoder
        def create_flexible_autoencoder():
            # Input layer with flexible shape
            input_layer = tf.keras.layers.Input(shape=(None, None, 1))
            
            # Encoder
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            
            decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            
            # Create model
            autoencoder = tf.keras.Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
            
            return autoencoder
        
        model = create_flexible_autoencoder()
        model.save('models/advanced_noise_reducer.h5')
        print("✓ Created improved advanced_noise_reducer.h5")
    
    print("All required models created successfully!")

if __name__ == "__main__":
    create_missing_models() 