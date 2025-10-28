"""
Baby Cry Analyzer - Training Script
Trains a lightweight CNN model on baby cry audio data for Raspberry Pi deployment
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000  # Lower sample rate for efficiency
DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512

class CryAnalyzerTrainer:
    def __init__(self, data_dir, model_save_path='cry_model'):
        self.data_dir = Path(data_dir)
        self.model_save_path = model_save_path
        self.label_encoder = LabelEncoder()
        self.classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        
    def extract_features(self, file_path, method='mfcc'):
        """
        Extract audio features from wav file
        Returns features suitable for CNN input
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Pad if too short
            if len(audio) < SAMPLE_RATE * DURATION:
                audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
            
            if method == 'mfcc':
                # Extract MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, 
                                            hop_length=HOP_LENGTH)
                # Add delta and delta-delta features
                mfcc_delta = librosa.feature.delta(mfccs)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                
                # Stack features
                features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
                
            elif method == 'melspectrogram':
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                                         n_mels=N_MELS,
                                                         hop_length=HOP_LENGTH)
                # Convert to log scale
                features = librosa.power_to_db(mel_spec, ref=np.max)
            
            return features.T  # Transpose to (time, features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def load_data(self, feature_method='mfcc'):
        """
        Load all audio files and extract features
        """
        X = []
        y = []
        
        print("Loading and processing audio files...")
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} not found!")
                continue
            
            wav_files = list(class_dir.glob('*.wav'))
            print(f"Processing {len(wav_files)} files from {class_name}...")
            
            for wav_file in wav_files:
                features = self.extract_features(wav_file, method=feature_method)
                
                if features is not None:
                    X.append(features)
                    y.append(class_name)
        
        if len(X) == 0:
            raise ValueError("No audio files were loaded. Please check your data directory structure.")
        
        # Convert to numpy arrays
        X = np.array(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nLoaded {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Classes: {self.classes}")
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """
        Build lightweight CNN model optimized for Raspberry Pi
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Reshape for CNN
            layers.Reshape((*input_shape, 1)),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Third conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        """
        # Load data
        X, y = self.load_data()
        
        # Add channel dimension and normalize
        X = X[..., np.newaxis]
        X = (X - X.mean()) / (X.std() + 1e-8)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        num_classes = len(self.classes)
        
        model = self.build_model(input_shape, num_classes)
        model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
        
        # Save model
        self.save_model(model)
        
        return model, history
    
    def save_model(self, model):
        """
        Save model in multiple formats for Raspberry Pi deployment
        """
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Save Keras model
        model.save(f'{self.model_save_path}/cry_model.h5')
        print(f"Saved Keras model to {self.model_save_path}/cry_model.h5")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(f'{self.model_save_path}/cry_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {self.model_save_path}/cry_model.tflite")
        
        # Save label encoder
        label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        with open(f'{self.model_save_path}/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Save configuration
        config = {
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION,
            'n_mfcc': N_MFCC,
            'n_mels': N_MELS,
            'hop_length': HOP_LENGTH,
            'classes': self.classes
        }
        with open(f'{self.model_save_path}/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved configuration to {self.model_save_path}/config.json")


if __name__ == '__main__':
    # Set your data directory path
    DATA_DIR = './cry_data'  # Change this to your data folder path
    
    # Create trainer
    trainer = CryAnalyzerTrainer(data_dir=DATA_DIR, model_save_path='./cry_model')
    
    # Train model
    model, history = trainer.train(epochs=50, batch_size=16)
    
    print("\nTraining complete! Model saved and ready for Raspberry Pi deployment.")