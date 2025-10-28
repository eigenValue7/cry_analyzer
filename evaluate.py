"""
Model Evaluation and Visualization Script
Evaluate trained model and visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
import librosa
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model_path='cry_model', data_dir='cry_data'):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        
        # Load configuration
        with open(self.model_path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load label mapping
        with open(self.model_path / 'label_mapping.json', 'r') as f:
            label_map = json.load(f)
            self.labels = [label_map[str(i)] for i in range(len(label_map))]
        
        # Load model
        self.model = tf.keras.models.load_model(self.model_path / 'cry_model.h5')
        
        print("Model loaded successfully!")
        print(f"Classes: {self.labels}")
    
    def extract_features(self, file_path):
        """Extract features from audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.config['sample_rate'], 
                                    duration=self.config['duration'])
            
            if len(audio) < self.config['sample_rate'] * self.config['duration']:
                audio = np.pad(audio, (0, self.config['sample_rate'] * 
                                     self.config['duration'] - len(audio)))
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, 
                                        n_mfcc=self.config['n_mfcc'],
                                        hop_length=self.config['hop_length'])
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
            return features.T
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def evaluate_on_test_set(self, test_dir=None):
        """
        Evaluate model on test set and generate metrics
        """
        if test_dir is None:
            test_dir = self.data_dir
        else:
            test_dir = Path(test_dir)
        
        X_test = []
        y_test = []
        y_pred = []
        
        print("\nEvaluating model on test data...")
        
        for idx, class_name in enumerate(self.labels):
            class_dir = test_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found!")
                continue
            
            wav_files = list(class_dir.glob('*.wav'))
            print(f"Processing {len(wav_files)} files from {class_name}...")
            
            for wav_file in wav_files:
                features = self.extract_features(wav_file)
                
                if features is not None:
                    X_test.append(features)
                    y_test.append(idx)
        
        # Convert to numpy arrays
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Normalize
        X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)
        X_test = X_test[..., np.newaxis]
        
        # Predict
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        return y_test, y_pred, predictions
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels,
                   yticklabels=self.labels)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, save_path='classification_report.png'):
        """
        Create visual classification report
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=self.labels,
                                      output_dict=True)
        
        # Extract metrics
        classes = self.labels
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        # Plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Metrics by Class', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification report saved to {save_path}")
        plt.close()
    
    def plot_prediction_confidence(self, predictions, y_true, save_path='confidence_distribution.png'):
        """
        Plot confidence distribution
        """
        confidences = np.max(predictions, axis=1)
        correct = (np.argmax(predictions, axis=1) == y_true)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of confidences
        axes[0].hist(confidences[correct], bins=20, alpha=0.6, label='Correct', color='green')
        axes[0].hist(confidences[~correct], bins=20, alpha=0.6, label='Incorrect', color='red')
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot by class
        confidence_by_class = [confidences[y_true == i] for i in range(len(self.labels))]
        axes[1].boxplot(confidence_by_class, labels=self.labels)
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence by Class', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution saved to {save_path}")
        plt.close()
    
    def generate_full_report(self, test_dir=None, output_dir='evaluation_results'):
        """
        Generate complete evaluation report with all visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        # Evaluate
        y_true, y_pred, predictions = self.evaluate_on_test_set(test_dir)
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        print("\nDetailed Classification Report:")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=self.labels)
        print(report)
        
        # Save text report
        with open(output_dir / 'evaluation_report.txt', 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
            f.write("Detailed Classification Report:\n")
            f.write("="*60 + "\n")
            f.write(report)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(y_true, y_pred, 
                                  output_dir / 'confusion_matrix.png')
        self.plot_classification_report(y_true, y_pred, 
                                       output_dir / 'classification_report.png')
        self.plot_prediction_confidence(predictions, y_true, 
                                       output_dir / 'confidence_distribution.png')
        
        print("\n" + "="*60)
        print(f"Evaluation complete! Results saved to {output_dir}/")
        print("="*60)
        
        return accuracy, report


def visualize_audio_features(audio_file, output_dir='feature_visualization'):
    """
    Visualize audio features for a single file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Waveform
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    axes[0].plot(time, audio)
    axes[0].set_title('Waveform', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img1 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[3])
    axes[3].set_title('MFCCs', fontsize=12, fontweight='bold')
    fig.colorbar(img3, ax=axes[3])
    
    plt.tight_layout()
    output_file = output_dir / f"{Path(audio_file).stem}_features.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Feature visualization saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Model Evaluation Script")
        print("\nUsage:")
        print("  python evaluate_model.py --evaluate [test_dir]")
        print("  python evaluate_model.py --visualize <audio_file>")
        print("\nExamples:")
        print("  python evaluate_model.py --evaluate")
        print("  python evaluate_model.py --evaluate ./test_data")
        print("  python evaluate_model.py --visualize baby_cry.wav")
        sys.exit(0)
    
    if '--evaluate' in sys.argv:
        test_dir = sys.argv[2] if len(sys.argv) > 2 else None
        evaluator = ModelEvaluator()
        evaluator.generate_full_report(test_dir=test_dir)
    
    elif '--visualize' in sys.argv and len(sys.argv) > 2:
        audio_file = sys.argv[2]
        visualize_audio_features(audio_file)
    
    else:
        print("Invalid arguments. Use --evaluate or --visualize")