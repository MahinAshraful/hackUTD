#!/usr/bin/env python3
"""
Simple Prediction with Acoustic Features Only
Uses: Jitter, Shimmer, HNR (14 features)
No MFCCs - avoids extraction mismatch
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.audio_feature_extractor import AudioFeatureExtractor

def predict_simple(audio_file, debug=False):
    """Predict using simple acoustic features only"""

    print("\n" + "="*70)
    print("PARKINSON'S DETECTION (Simple Acoustic Features)")
    print("="*70)

    # Load model
    with open('models/saved_models/SimpleAcoustic_best.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('data/processed/feature_stats_simple.json', 'r') as f:
        feature_stats = json.load(f)

    with open('data/processed/feature_list_simple.json', 'r') as f:
        feature_list = json.load(f)

    print(f"\n🎵 Processing: {audio_file}")
    print(f"📊 Using {len(feature_list)} features: Jitter, Shimmer, HNR")

    # Extract features
    extractor = AudioFeatureExtractor(validate_quality=True)

    try:
        features_dict = extractor.extract_features(audio_file, return_dict=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

    # Get simple features only
    features = np.array([features_dict[name] for name in feature_list])

    if debug:
        print(f"\n🔍 Extracted Features:")
        print("-" * 70)
        print(f"{'Feature':<15} {'Value':>12} {'Train Mean':>12} {'Difference':>12}")
        print("-" * 70)
        for i, name in enumerate(feature_list):
            value = features[i]
            mean = feature_stats[name]['mean']
            diff = value - mean
            print(f"{name:<15} {value:>12.4f} {mean:>12.4f} {diff:>12.4f}")

    # Normalize
    normalized = np.zeros_like(features)
    for i, name in enumerate(feature_list):
        mean = feature_stats[name]['mean']
        std = feature_stats[name]['std']
        if std > 0:
            normalized[i] = (features[i] - mean) / std

    # Predict
    prediction = model.predict(normalized.reshape(1, -1))[0]
    probabilities = model.predict_proba(normalized.reshape(1, -1))[0]

    pd_prob = probabilities[1]
    healthy_prob = probabilities[0]

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if pd_prob < 0.3:
        risk = "LOW"
        emoji = "✅"
        rec = "Normal voice. Routine checkup recommended."
    elif pd_prob < 0.6:
        risk = "MODERATE"
        emoji = "⚠️ "
        rec = "Some indicators. Monitor, retest in 3-6 months."
    elif pd_prob < 0.8:
        risk = "HIGH"
        emoji = "🔴"
        rec = "Significant indicators. Consult neurologist."
    else:
        risk = "VERY HIGH"
        emoji = "🚨"
        rec = "Strong indicators. Urgent referral recommended."

    print(f"\n{emoji} RISK LEVEL: {risk}")
    print(f"\n📊 Parkinson's Probability: {pd_prob*100:.1f}%")
    print(f"   Healthy Probability: {healthy_prob*100:.1f}%")
    print(f"\n💡 Recommendation: {rec}")

    if debug and hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        contributions = coefficients * normalized

        print(f"\n🔍 Feature Contributions:")
        print("-" * 70)
        fc = list(zip(feature_list, features, normalized, contributions))
        fc.sort(key=lambda x: abs(x[3]), reverse=True)

        for name, raw, norm, contrib in fc[:10]:
            direction = "→PD" if contrib > 0 else "→H"
            print(f"{name:<15} {raw:>10.4f} {norm:>10.2f} {contrib:>10.3f} {direction}")

    print("\n" + "="*70)
    print("ℹ️  Model: Logistic Regression (14 features)")
    print("   Accuracy: 70%, ROC-AUC: 79%")
    print("   Features: Only Jitter, Shimmer, HNR (no MFCCs)")
    print("="*70 + "\n")

    return {
        'prediction': int(prediction),
        'pd_probability': float(pd_prob),
        'risk_level': risk
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 predict_simple.py <audio_file> [--debug]")
        print("\nExamples:")
        print("  python3 predict_simple.py tanvir.wav")
        print("  python3 predict_simple.py mahin.wav --debug")
        sys.exit(1)

    audio_file = sys.argv[1]
    debug_mode = '--debug' in sys.argv

    predict_simple(audio_file, debug=debug_mode)
