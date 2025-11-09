#!/usr/bin/env python3
"""
Debug Feature Values
Shows actual extracted features and compares to training data
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.audio_feature_extractor import AudioFeatureExtractor

def debug_features(audio_file):
    """Show detailed feature breakdown"""

    print("="*80)
    print(f"FEATURE DEBUG: {audio_file}")
    print("="*80)

    # Extract features
    extractor = AudioFeatureExtractor(validate_quality=False)
    features = extractor.extract_features(audio_file, return_dict=True)

    # Load training stats for comparison
    stats_file = "data/processed/feature_stats.json"
    with open(stats_file, 'r') as f:
        training_stats = json.load(f)

    print(f"\n📊 KEY FEATURES (compared to UCI training data):\n")
    print(f"{'Feature':<15} {'Your Value':>12} {'Training Mean':>15} {'Difference':>12} {'Status':>10}")
    print("-"*80)

    key_features = [
        'Jitter_rel', 'Jitter_abs', 'Shimmer_dB', 'Shim_loc',
        'HNR05', 'HNR15', 'HNR25', 'HNR35',
        'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4'
    ]

    for feat in key_features:
        if feat in features and feat in training_stats:
            your_val = features[feat]
            train_mean = training_stats[feat]['mean']
            train_std = training_stats[feat]['std']
            diff = your_val - train_mean

            # How many standard deviations away?
            if train_std > 0:
                z_score = diff / train_std
            else:
                z_score = 0

            # Status
            if abs(z_score) < 1:
                status = "✓ Normal"
            elif abs(z_score) < 2:
                status = "⚠️  Unusual"
            else:
                status = "🔴 Very off"

            print(f"{feat:<15} {your_val:>12.4f} {train_mean:>15.4f} {diff:>12.4f} {status:>10}")

    # Load real data stats too
    real_stats_file = "data/processed/real_data_feature_stats.json"
    if Path(real_stats_file).exists():
        with open(real_stats_file, 'r') as f:
            real_stats = json.load(f)

        print(f"\n📊 COMPARISON TO REAL PHONE-QUALITY DATA:\n")
        print(f"{'Feature':<15} {'Your Value':>12} {'Real Data Mean':>16} {'Difference':>12} {'Status':>10}")
        print("-"*80)

        for feat in key_features:
            if feat in features and feat in real_stats:
                your_val = features[feat]
                real_mean = real_stats[feat]['mean']
                real_std = real_stats[feat]['std']
                diff = your_val - real_mean

                # How many standard deviations away?
                if real_std > 0:
                    z_score = diff / real_std
                else:
                    z_score = 0

                # Status
                if abs(z_score) < 1:
                    status = "✓ Normal"
                elif abs(z_score) < 2:
                    status = "⚠️  Unusual"
                else:
                    status = "🔴 Very off"

                print(f"{feat:<15} {your_val:>12.4f} {real_mean:>16.4f} {diff:>12.4f} {status:>10}")

    print("\n" + "="*80)
    print("💡 INTERPRETATION:")
    print("="*80)
    print("""
✓ Normal:     Within 1 standard deviation (typical)
⚠️  Unusual:   1-2 standard deviations away (uncommon)
🔴 Very off:  >2 standard deviations (rare)

Key Features to Watch:
- Jitter: Voice frequency instability (higher in PD)
- Shimmer: Amplitude variation (higher in PD)
- HNR: Harmonic-to-noise ratio (lower in PD)
- MFCCs: Spectral characteristics of voice
""")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_features.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    debug_features(audio_file)
