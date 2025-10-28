"""
Data Balancing Script for Imbalanced Baby Cry Dataset
Handles severe class imbalance using augmentation and class weights
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
from collections import Counter

class AudioAugmentation:
    """Audio data augmentation techniques"""
    
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio + noise
    
    @staticmethod
    def time_stretch(audio, rate=None):
        """Stretch or compress time"""
        if rate is None:
            rate = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    @staticmethod
    def pitch_shift(audio, sr, n_steps=None):
        """Shift pitch"""
        if n_steps is None:
            n_steps = random.randint(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def change_volume(audio, factor=None):
        """Change volume"""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        return audio * factor
    
    @staticmethod
    def time_shift(audio, shift=None):
        """Shift audio in time"""
        if shift is None:
            shift = random.randint(-len(audio)//10, len(audio)//10)
        return np.roll(audio, shift)
    
    @staticmethod
    def add_background_noise(audio, noise_factor=0.003):
        """Add low-level background noise"""
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise


class DataBalancer:
    def __init__(self, data_dir, target_samples_per_class=100):
        self.data_dir = Path(data_dir)
        self.target_samples = target_samples_per_class
        self.augmenter = AudioAugmentation()
        self.classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    
    def analyze_distribution(self):
        """Analyze current data distribution"""
        print("\n" + "="*60)
        print("CURRENT DATA DISTRIBUTION")
        print("="*60)
        
        distribution = {}
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.wav')))
                distribution[class_name] = count
                print(f"{class_name:15s}: {count:3d} samples")
            else:
                distribution[class_name] = 0
                print(f"{class_name:15s}: Folder not found!")
        
        print(f"\n{'='*60}")
        print(f"Total samples: {sum(distribution.values())}")
        print(f"Max class: {max(distribution.values())}")
        print(f"Min class: {min(distribution.values())}")
        print(f"Imbalance ratio: {max(distribution.values()) / max(min(distribution.values()), 1):.1f}:1")
        print(f"{'='*60}\n")
        
        return distribution
    
    def augment_audio(self, audio, sr, num_augmentations=1):
        """Create augmented versions of audio"""
        augmented = []
        
        for _ in range(num_augmentations):
            aug_audio = audio.copy()
            
            # Randomly apply 2-3 augmentations
            augmentation_choices = random.sample([
                lambda x: self.augmenter.add_noise(x),
                lambda x: self.augmenter.time_stretch(x),
                lambda x: self.augmenter.pitch_shift(x, sr),
                lambda x: self.augmenter.change_volume(x),
                lambda x: self.augmenter.time_shift(x),
                lambda x: self.augmenter.add_background_noise(x)
            ], k=random.randint(2, 3))
            
            for aug_func in augmentation_choices:
                aug_audio = aug_func(aug_audio)
            
            augmented.append(aug_audio)
        
        return augmented
    
    def balance_dataset(self, output_dir=None, strategy='augment'):
        """
        Balance dataset using different strategies
        
        Strategies:
        - 'augment': Augment minority classes to match target
        - 'undersample': Reduce majority class
        - 'hybrid': Combination of both
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / (self.data_dir.name + "_balanced")
        else:
            output_dir = Path(output_dir)
        
        distribution = self.analyze_distribution()
        
        if strategy == 'undersample':
            target = self.target_samples
        elif strategy == 'augment':
            target = min(max(distribution.values()), self.target_samples)
        else:  # hybrid
            target = self.target_samples
        
        print(f"Strategy: {strategy}")
        print(f"Target samples per class: {target}\n")
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            if not class_dir.exists():
                print(f"Skipping {class_name} (not found)")
                continue
            
            wav_files = list(class_dir.glob('*.wav'))
            current_count = len(wav_files)
            
            print(f"\n{class_name}:")
            print(f"  Current: {current_count} samples")
            
            if current_count == 0:
                print(f"  ⚠ No files to process!")
                continue
            
            # If too many samples, undersample
            if current_count > target and strategy in ['undersample', 'hybrid']:
                selected_files = random.sample(wav_files, target)
                print(f"  Action: Undersampling to {target}")
                
                for i, wav_file in enumerate(selected_files):
                    audio, sr = librosa.load(wav_file, sr=None)
                    output_file = output_class_dir / f"{class_name}_{i:04d}.wav"
                    sf.write(output_file, audio, sr)
            
            # If too few samples, augment
            elif current_count < target:
                needed = target - current_count
                augmentations_per_file = max(1, needed // current_count)
                print(f"  Action: Augmenting (need {needed} more samples)")
                print(f"  Creating {augmentations_per_file} augmentation(s) per file")
                
                # Copy original files
                for i, wav_file in enumerate(wav_files):
                    audio, sr = librosa.load(wav_file, sr=None)
                    output_file = output_class_dir / f"{class_name}_{i:04d}_orig.wav"
                    sf.write(output_file, audio, sr)
                
                # Create augmented versions
                aug_count = 0
                for wav_file in wav_files:
                    audio, sr = librosa.load(wav_file, sr=None)
                    
                    # Create augmentations
                    augmented_audios = self.augment_audio(audio, sr, augmentations_per_file)
                    
                    for j, aug_audio in enumerate(augmented_audios):
                        if len(aug_audio) > 0:  # Check valid audio
                            output_file = output_class_dir / f"{class_name}_{current_count + aug_count:04d}_aug.wav"
                            sf.write(output_file, aug_audio, sr)
                            aug_count += 1
                            
                            if current_count + aug_count >= target:
                                break
                    
                    if current_count + aug_count >= target:
                        break
                
                print(f"  Created {aug_count} augmented samples")
            
            # If just right, copy as-is
            else:
                print(f"  Action: Copying (already balanced)")
                for i, wav_file in enumerate(wav_files):
                    audio, sr = librosa.load(wav_file, sr=None)
                    output_file = output_class_dir / f"{class_name}_{i:04d}.wav"
                    sf.write(output_file, audio, sr)
            
            final_count = len(list(output_class_dir.glob('*.wav')))
            print(f"  Final: {final_count} samples ✓")
        
        print("\n" + "="*60)
        print(f"BALANCED DATASET SAVED TO: {output_dir}")
        print("="*60)
        
        # Show new distribution
        print("\nNEW DISTRIBUTION:")
        for class_name in self.classes:
            class_dir = output_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.wav')))
                print(f"{class_name:15s}: {count:3d} samples")
        
        return output_dir


def main():
    """Main function with usage examples"""
    import sys
    
    if len(sys.argv) < 2:
        print("Data Balancing Script for Baby Cry Analyzer")
        print("\nUsage:")
        print("  python balance_data.py <data_dir> [options]")
        print("\nOptions:")
        print("  --target N          Target samples per class (default: 100)")
        print("  --strategy S        Strategy: augment, undersample, hybrid (default: hybrid)")
        print("  --output DIR        Output directory (default: <data_dir>_balanced)")
        print("\nExamples:")
        print("  python balance_data.py cry_data")
        print("  python balance_data.py cry_data --target 80 --strategy hybrid")
        print("  python balance_data.py cry_data --target 100 --output balanced_cry_data")
        return
    
    data_dir = sys.argv[1]
    
    # Parse arguments
    target = 100
    strategy = 'hybrid'
    output_dir = None
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--target' and i + 1 < len(sys.argv):
            target = int(sys.argv[i + 1])
        elif sys.argv[i] == '--strategy' and i + 1 < len(sys.argv):
            strategy = sys.argv[i + 1]
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
    
    # Balance dataset
    balancer = DataBalancer(data_dir, target_samples_per_class=target)
    output_path = balancer.balance_dataset(output_dir=output_dir, strategy=strategy)
    
    print(f"\n✓ Dataset balanced successfully!")
    print(f"\nNext steps:")
    print(f"1. Review the balanced data in: {output_path}")
    print(f"2. Train your model using the balanced dataset")
    print(f"3. python train_cry_analyzer.py  # Update DATA_DIR to balanced folder")


if __name__ == '__main__':
    main()