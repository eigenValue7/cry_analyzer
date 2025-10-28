"""
Data Preparation Script for Baby Cry Analyzer
Helps verify and prepare your audio data for training
"""

import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

class DataPreparation:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        
    def verify_structure(self):
        """
        Verify that the data directory has correct structure
        """
        print("="*60)
        print("DATA STRUCTURE VERIFICATION")
        print("="*60)
        
        if not self.data_dir.exists():
            print(f"❌ Error: Directory '{self.data_dir}' does not exist!")
            print(f"\nPlease create the directory and organize your files as:")
            print(f"\n{self.data_dir}/")
            for class_name in self.classes:
                print(f"  ├── {class_name}/")
                print(f"  │   ├── audio1.wav")
                print(f"  │   ├── audio2.wav")
                print(f"  │   └── ...")
            return False
        
        all_ok = True
        total_files = 0
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"❌ Missing folder: {class_name}")
                all_ok = False
                continue
            
            wav_files = list(class_dir.glob('*.wav'))
            total_files += len(wav_files)
            
            status = "✓" if len(wav_files) > 0 else "⚠"
            print(f"{status} {class_name:15s}: {len(wav_files):3d} files")
            
            if len(wav_files) == 0:
                print(f"  └─ Warning: No .wav files found!")
        
        print(f"\n{'='*60}")
        print(f"Total audio files: {total_files}")
        
        if total_files < 50:
            print(f"⚠ Warning: Only {total_files} files total.")
            print(f"  Recommended: At least 50-100 files per class for good accuracy")
        
        if all_ok:
            print("\n✓ Data structure is correct!")
        else:
            print("\n❌ Please fix the issues above before training.")
        
        print(f"{'='*60}\n")
        return all_ok
    
    def analyze_audio_files(self):
        """
        Analyze audio files for potential issues
        """
        print("\n" + "="*60)
        print("AUDIO FILE ANALYSIS")
        print("="*60)
        
        issues = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                continue
            
            print(f"\nAnalyzing {class_name}...")
            wav_files = list(class_dir.glob('*.wav'))
            
            durations = []
            sample_rates = []
            corrupted = []
            
            for wav_file in wav_files:
                try:
                    audio, sr = librosa.load(wav_file, sr=None)
                    duration = len(audio) / sr
                    
                    durations.append(duration)
                    sample_rates.append(sr)
                    
                    # Check for silence
                    if np.max(np.abs(audio)) < 0.001:
                        issues.append(f"  ⚠ {wav_file.name}: File appears to be silent")
                    
                except Exception as e:
                    corrupted.append(wav_file.name)
                    issues.append(f"  ❌ {wav_file.name}: Error - {str(e)}")
            
            if durations:
                print(f"  Duration: {np.mean(durations):.2f}s (avg), "
                      f"{np.min(durations):.2f}s (min), "
                      f"{np.max(durations):.2f}s (max)")
                print(f"  Sample rates: {set(sample_rates)}")
                
                if len(corrupted) > 0:
                    print(f"  ❌ Corrupted files: {len(corrupted)}")
        
        if issues:
            print(f"\n{'='*60}")
            print("ISSUES FOUND:")
            print("="*60)
            for issue in issues:
                print(issue)
        else:
            print("\n✓ All audio files look good!")
        
        print(f"\n{'='*60}\n")
    
    def convert_to_standard_format(self, target_sr=16000, output_dir=None):
        """
        Convert all audio files to standard format (16kHz, mono, 16-bit)
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / (self.data_dir.name + "_processed")
        else:
            output_dir = Path(output_dir)
        
        print(f"\nConverting audio files to standard format...")
        print(f"Target sample rate: {target_sr} Hz")
        print(f"Output directory: {output_dir}\n")
        
        converted_count = 0
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Create output directory
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            wav_files = list(class_dir.glob('*.wav'))
            print(f"Converting {len(wav_files)} files from {class_name}...")
            
            for wav_file in wav_files:
                try:
                    # Load audio
                    audio, sr = librosa.load(wav_file, sr=target_sr, mono=True)
                    
                    # Save in standard format
                    output_file = output_class_dir / wav_file.name
                    sf.write(output_file, audio, target_sr, subtype='PCM_16')
                    
                    converted_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error converting {wav_file.name}: {e}")
        
        print(f"\n✓ Converted {converted_count} files successfully!")
        print(f"Processed data saved to: {output_dir}")
    
    def split_long_recordings(self, chunk_duration=3.0, output_dir=None):
        """
        Split long recordings into smaller chunks
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / (self.data_dir.name + "_chunked")
        else:
            output_dir = Path(output_dir)
        
        print(f"\nSplitting recordings into {chunk_duration}s chunks...")
        print(f"Output directory: {output_dir}\n")
        
        total_chunks = 0
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                continue
            
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            wav_files = list(class_dir.glob('*.wav'))
            
            for wav_file in wav_files:
                try:
                    audio, sr = librosa.load(wav_file, sr=None)
                    duration = len(audio) / sr
                    
                    # If file is shorter than chunk_duration, just copy it
                    if duration <= chunk_duration:
                        output_file = output_class_dir / wav_file.name
                        sf.write(output_file, audio, sr)
                        total_chunks += 1
                        continue
                    
                    # Split into chunks
                    chunk_samples = int(chunk_duration * sr)
                    num_chunks = int(duration / chunk_duration)
                    
                    for i in range(num_chunks):
                        start_sample = i * chunk_samples
                        end_sample = start_sample + chunk_samples
                        chunk = audio[start_sample:end_sample]
                        
                        # Save chunk
                        output_file = output_class_dir / f"{wav_file.stem}_chunk{i:03d}.wav"
                        sf.write(output_file, chunk, sr)
                        total_chunks += 1
                
                except Exception as e:
                    print(f"  ❌ Error processing {wav_file.name}: {e}")
        
        print(f"\n✓ Created {total_chunks} chunks!")
        print(f"Chunked data saved to: {output_dir}")


def main():
    """
    Main function for data preparation
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Baby Cry Analyzer - Data Preparation Tool")
        print("\nUsage:")
        print("  python prepare_data.py <data_directory> --verify")
        print("  python prepare_data.py <data_directory> --analyze")
        print("  python prepare_data.py <data_directory> --convert [output_dir]")
        print("  python prepare_data.py <data_directory> --split [output_dir]")
        return
    
    data_dir = sys.argv[1]
    prep = DataPreparation(data_dir)
    
    if '--verify' in sys.argv:
        prep.verify_structure()
    
    if '--analyze' in sys.argv:
        prep.analyze_audio_files()
    
    if '--convert' in sys.argv:
        output_dir = sys.argv[sys.argv.index('--convert') + 1] if len(sys.argv) > sys.argv.index('--convert') + 1 else None
        prep.convert_to_standard_format(output_dir=output_dir)
    
    if '--split' in sys.argv:
        output_dir = sys.argv[sys.argv.index('--split') + 1] if len(sys.argv) > sys.argv.index('--split') + 1 else None
        prep.split_long_recordings(output_dir=output_dir)
    
    # If no specific command, run verification
    if len(sys.argv) == 2:
        prep.verify_structure()
        prep.analyze_audio_files()


if __name__ == '__main__':
    main()