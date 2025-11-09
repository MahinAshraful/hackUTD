#!/usr/bin/env python3
"""
Simple Prediction Script (40 Features - Fixed Model)
Usage: python3 predict.py my_voice.wav
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.audio_feature_extractor import AudioFeatureExtractor

def predict(audio_file, debug=False):
    """Predict using the 40-feature model (without placeholders)"""

    print("\n" + "="*60)
    print("PARKINSON'S DISEASE PREDICTION (40-Feature Model)")
    print("="*60)

    # Load model and stats
    with open('models/saved_models/LogisticRegression_L2_40feat.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('data/processed/feature_stats_40.json', 'r') as f:
        feature_stats = json.load(f)

    with open('data/processed/feature_list_40.json', 'r') as f:
        feature_list = json.load(f)

    # Extract features
    print(f"\n🎵 Processing: {audio_file}")
    extractor = AudioFeatureExtractor(validate_quality=True)

    try:
        features_dict = extractor.extract_features(audio_file, return_dict=True)
    except Exception as e:
        print(f"\n❌ Error extracting features: {e}")
        return None

    # Get only the 40 features (skip RPDE, DFA, PPE, GNE)
    features = np.array([features_dict[name] for name in feature_list])

    if debug:
        print(f"\n🔍 Debug Info:")
        print(f"   Features extracted: {len(features)}")
        print(f"   Features expected: {len(feature_list)}")

    # Normalize
    print("\n   → Normalizing features...")
    normalized = np.zeros_like(features)
    for i, name in enumerate(feature_list):
        mean = feature_stats[name]['mean']
        std = feature_stats[name]['std']
        if std > 0:
            normalized[i] = (features[i] - mean) / std
        else:
            normalized[i] = 0

    # Predict
    print("   → Running ML model...")
    prediction = model.predict(normalized.reshape(1, -1))[0]
    probabilities = model.predict_proba(normalized.reshape(1, -1))[0]

    pd_prob = probabilities[1]
    healthy_prob = probabilities[0]

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Risk level
    if pd_prob < 0.3:
        risk_level = "LOW"
        emoji = "✅"
        recommendation = "Normal voice characteristics. Routine checkup recommended."
    elif pd_prob < 0.6:
        risk_level = "MODERATE"
        emoji = "⚠️ "
        recommendation = "Some indicators present. Monitor and retest in 3-6 months."
    elif pd_prob < 0.8:
        risk_level = "HIGH"
        emoji = "🔴"
        recommendation = "Significant indicators detected. Consult neurologist for clinical assessment."
    else:
        risk_level = "VERY HIGH"
        emoji = "🚨"
        recommendation = "Strong indicators present. Urgent neurologist referral recommended."

    print(f"\n{emoji} RISK LEVEL: {risk_level}")
    print(f"\n📊 Parkinson's Probability: {pd_prob*100:.1f}%")
    print(f"   Healthy Probability: {healthy_prob*100:.1f}%")
    print(f"\n💡 Recommendation:")
    print(f"   {recommendation}")

    if debug:
        # Show top contributing features
        coefficients = model.coef_[0]
        contributions = coefficients * normalized

        feature_contributions = list(zip(feature_list, contributions))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n🔍 Top 5 Contributing Features:")
        for name, contrib in feature_contributions[:5]:
            direction = "→ PD" if contrib > 0 else "→ Healthy"
            print(f"   • {name:<15} {contrib:>8.3f} {direction}")

    print("\n" + "="*60)
    print("ℹ️  This model uses 40 features (RPDE, DFA, PPE, GNE removed)")
    print("   No placeholder features - all values extracted from your audio")
    print("="*60 + "\n")

    return {
        'prediction': int(prediction),
        'pd_probability': float(pd_prob),
        'healthy_probability': float(healthy_prob),
        'risk_level': risk_level,
        'recommendation': recommendation
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 predict.py <audio_file> [--debug]")
        print("\nExamples:")
        print("  python3 predict.py my_voice.wav")
        print("  python3 predict.py outputs/audio/test.wav --debug")
        print("\nFor detailed debugging:")
        print("  python3 debug_predict.py my_voice.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    debug_mode = '--debug' in sys.argv

    predict(audio_file, debug=debug_mode)
