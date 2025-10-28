"""
Baby Cry Analyzer - Raspberry Pi Inference Script
Real-time baby cry classification on Raspberry Pi
"""

import numpy as np
import librosa
import json
import tensorflow as tf
import sounddevice as sd
from pathlib import Path
import time

class CryAnalyzer:
    def __init__(self, model_path='cry_model'):
        """
        Initialize the cry analyzer for Raspberry Pi
        """
        self.model_path = Path(model_path)
        
        # Load configuration
        with open(self.model_path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['sample_rate']
        self.duration = self.config['duration']
        self.n_mfcc = self.config['n_mfcc']
        self.hop_length = self.config['hop_length']
        
        # Load label mapping
        with open(self.model_path / 'label_mapping.json', 'r') as f:
            label_map = json.load(f)
            self.labels = [label_map[str(i)] for i in range(len(label_map))]
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path / 'cry_model.tflite')
        )
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Cry Analyzer initialized successfully!")
        print(f"Classes: {self.labels}")
    
    def extract_features(self, audio):
        """
        Extract MFCC features from audio
        """
        # Pad if too short
        if len(audio) < self.sample_rate * self.duration:
            audio = np.pad(audio, (0, self.sample_rate * self.duration - len(audio)))
        else:
            audio = audio[:self.sample_rate * self.duration]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack features
        features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
        
        return features.T  # Transpose to (time, features)
    
    def predict(self, audio):
        """
        Predict cry type from audio
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Reshape for model input
        features = features[np.newaxis, ..., np.newaxis].astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        
        # Get prediction
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_label = self.labels[predicted_idx]
        
        return predicted_label, confidence, probabilities
    
    def predict_from_file(self, audio_file):
        """
        Predict cry type from audio file
        """
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
        
        # Predict
        label, confidence, probs = self.predict(audio)
        
        return label, confidence, probs
    
    def analyze_and_display(self, audio_file):
        """
        Analyze audio file and display results
        """
        label, confidence, probs = self.predict_from_file(audio_file)
        
        print(f"\n{'='*50}")
        print(f"Predicted Cry Type: {label.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"{'='*50}")
        print("\nAll Probabilities:")
        for i, (class_name, prob) in enumerate(zip(self.labels, probs)):
            bar = '█' * int(prob * 50)
            print(f"{class_name:15s} [{prob*100:5.1f}%] {bar}")
        print(f"{'='*50}\n")
        
        return label, confidence


class RealtimeCryDetector:
    def __init__(self, model_path='cry_model', threshold=0.6):
        """
        Real-time cry detection using microphone
        """
        self.analyzer = CryAnalyzer(model_path)
        self.threshold = threshold
        self.sample_rate = self.analyzer.sample_rate
        self.duration = self.analyzer.duration
        self.is_recording = False
        
    def detect_cry_level(self, audio):
        """
        Simple cry detection based on volume
        """
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.02  # Adjust threshold as needed
    
    def monitor(self, callback=None):
        """
        Monitor microphone for crying sounds
        """
        print("Starting real-time cry monitoring...")
        print("Listening for baby cries...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Record audio chunk
                audio = sd.rec(
                    int(self.duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                audio = audio.flatten()
                
                # Check if crying detected
                if self.detect_cry_level(audio):
                    print("Cry detected! Analyzing...")
                    
                    # Predict cry type
                    label, confidence, probs = self.analyzer.predict(audio)
                    
                    if confidence > self.threshold:
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{timestamp}] Baby is crying: {label.upper()} ({confidence*100:.1f}%)")
                        
                        if callback:
                            callback(label, confidence, timestamp)
                    else:
                        print("Low confidence, continuing to monitor...")
                else:
                    print(".", end="", flush=True)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


def main():
    """
    Main function with usage examples
    """
    import sys
    
    model_path = './cry_model'
    
    if len(sys.argv) < 2:
        print("Baby Cry Analyzer - Raspberry Pi")
        print("\nUsage:")
        print("  python raspberry_pi_inference.py <audio_file>    - Analyze a single file")
        print("  python raspberry_pi_inference.py --realtime      - Real-time monitoring")
        print("  python raspberry_pi_inference.py --test <folder> - Test on folder")
        return
    
    if sys.argv[1] == '--realtime':
        # Real-time monitoring
        detector = RealtimeCryDetector(model_path=model_path)
        detector.monitor()
        
    elif sys.argv[1] == '--test' and len(sys.argv) > 2:
        # Test on folder
        analyzer = CryAnalyzer(model_path=model_path)
        test_folder = Path(sys.argv[2])
        
        for audio_file in test_folder.glob('*.wav'):
            print(f"\nAnalyzing: {audio_file.name}")
            analyzer.analyze_and_display(audio_file)
            
    else:
        # Single file analysis
        analyzer = CryAnalyzer(model_path=model_path)
        audio_file = sys.argv[1]
        analyzer.analyze_and_display(audio_file)


if __name__ == '__main__':
    main()