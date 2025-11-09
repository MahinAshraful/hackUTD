#!/usr/bin/env python3
"""
Debug Prediction Script
Shows detailed feature extraction and analysis
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.audio_feature_extractor import AudioFeatureExtractor
from src.parkinson_predictor import ParkinsonPredictor

def debug_predict(audio_file):
    """Predict with detailed debugging output"""

    print("\n" + "="*80)
    print("🔍 DEBUG MODE - DETAILED PREDICTION ANALYSIS")
    print("="*80)

    # Initialize
    extractor = AudioFeatureExtractor(validate_quality=False)
    predictor = ParkinsonPredictor()

    # Load training stats
    with open('data/processed/feature_stats.json', 'r') as f:
        train_stats = json.load(f)

    print(f"\n📁 Audio file: {audio_file}")

    # Extract features
    print("\n" + "-"*80)
    print("STEP 1: EXTRACTING FEATURES")
    print("-"*80)

    features_dict = extractor.extract_features(audio_file, return_dict=True)
    features_array = np.array([features_dict[name] for name in extractor.feature_names])

    print(f"✓ Extracted {len(features_array)} features")

    # Show raw features vs training mean
    print("\n" + "-"*80)
    print("STEP 2: COMPARING TO TRAINING DATA")
    print("-"*80)
    print(f"{'Feature':<15} {'Your Value':>12} {'Train Mean':>12} {'Difference':>12} {'Z-Score':>10}")
    print("-"*80)

    problematic_features = []
    for i, name in enumerate(extractor.feature_names):
        your_value = features_array[i]

        if name in train_stats:
            train_mean = train_stats[name]['mean']
            train_std = train_stats[name]['std']

            diff = your_value - train_mean
            z_score = diff / train_std if train_std > 0 else 0

            # Flag if more than 2 standard deviations away
            flag = "⚠️ " if abs(z_score) > 2 else "   "

            print(f"{flag}{name:<15} {your_value:>12.4f} {train_mean:>12.4f} {diff:>12.4f} {z_score:>10.2f}")

            if abs(z_score) > 2:
                problematic_features.append((name, z_score))

    # Normalize features
    print("\n" + "-"*80)
    print("STEP 3: NORMALIZING FEATURES")
    print("-"*80)

    normalized = predictor.normalize_features(features_array)
    print(f"✓ Normalized {len(normalized)} features")

    # Show which features are most extreme
    if problematic_features:
        print(f"\n⚠️  Found {len(problematic_features)} features >2 std deviations from training:")
        for name, z in sorted(problematic_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"   • {name}: {z:.2f} σ")

    # Make prediction
    print("\n" + "-"*80)
    print("STEP 4: MODEL PREDICTION")
    print("-"*80)

    prediction = predictor.model.predict(normalized.reshape(1, -1))[0]
    probabilities = predictor.model.predict_proba(normalized.reshape(1, -1))[0]

    print(f"Prediction: {prediction} (0=Healthy, 1=Parkinson's)")
    print(f"Healthy probability: {probabilities[0]*100:.1f}%")
    print(f"PD probability: {probabilities[1]*100:.1f}%")

    # Show feature importance from model
    if hasattr(predictor.model, 'coef_'):
        print("\n" + "-"*80)
        print("STEP 5: FEATURE IMPORTANCE (Model Coefficients)")
        print("-"*80)

        coefficients = predictor.model.coef_[0]

        # Calculate contribution: coefficient * normalized_feature
        contributions = coefficients * normalized

        # Sort by absolute contribution
        feature_contributions = [(name, coef, norm, contrib)
                                for name, coef, norm, contrib
                                in zip(extractor.feature_names, coefficients, normalized, contributions)]

        feature_contributions.sort(key=lambda x: abs(x[3]), reverse=True)

        print(f"{'Feature':<15} {'Coefficient':>12} {'Normalized':>12} {'Contribution':>14}")
        print("-"*80)

        total_positive = 0
        total_negative = 0

        for name, coef, norm, contrib in feature_contributions[:15]:
            arrow = "→ PD" if contrib > 0 else "→ Healthy"
            print(f"{name:<15} {coef:>12.4f} {norm:>12.4f} {contrib:>14.4f} {arrow}")

            if contrib > 0:
                total_positive += contrib
            else:
                total_negative += contrib

        print("-"*80)
        print(f"{'TOTAL POSITIVE':.<30} {total_positive:>14.4f} (pushes to PD)")
        print(f"{'TOTAL NEGATIVE':.<30} {total_negative:>14.4f} (pushes to Healthy)")
        print(f"{'NET EFFECT':.<30} {total_positive + total_negative:>14.4f}")

    # Identify placeholder features
    print("\n" + "-"*80)
    print("STEP 6: CHECKING FOR PLACEHOLDER FEATURES")
    print("-"*80)

    placeholder_features = ['RPDE', 'DFA', 'PPE', 'GNE']
    has_placeholders = False

    for name in placeholder_features:
        if name in features_dict:
            value = features_dict[name]
            train_mean = train_stats[name]['mean']

            # Check if it's close to a known placeholder value
            is_placeholder = (abs(value - 0.5) < 0.01 or
                            abs(value - 0.7) < 0.01 or
                            abs(value - 0.2) < 0.01 or
                            abs(value - train_mean) < 0.01)

            status = "⚠️  PLACEHOLDER" if is_placeholder else "✓ Real"
            print(f"{status} {name}: {value:.4f}")

            if is_placeholder:
                has_placeholders = True

    if has_placeholders:
        print("\n❌ WARNING: Placeholder features detected!")
        print("   These features are NOT extracted from your audio.")
        print("   They are hardcoded values causing false predictions.")
        print("\n   Solution: Retrain model without these 4 features")

    # Final assessment
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    if probabilities[1] > 0.8:
        print("🚨 VERY HIGH RISK")
    elif probabilities[1] > 0.6:
        print("🔴 HIGH RISK")
    elif probabilities[1] > 0.3:
        print("⚠️  MODERATE RISK")
    else:
        print("✅ LOW RISK")

    print(f"\nParkinson's Probability: {probabilities[1]*100:.1f}%")

    if has_placeholders:
        print("\n⚠️  NOTE: This prediction is UNRELIABLE due to placeholder features!")
        print("   Run: python3 retrain_40_features.py")
        print("   Then test again with your voice.")

    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_predict.py <audio_file>")
        print("\nExample:")
        print("  python3 debug_predict.py my_voice.wav")
        print("  python3 debug_predict.py outputs/audio/test.wav")
        sys.exit(1)

    debug_predict(sys.argv[1])
