#!/usr/bin/env python3
"""
Predict using Real-Data-Trained Model
Model trained on phone-quality HC vs PD recordings
"""

import numpy as np
import pickle
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.audio_feature_extractor import AudioFeatureExtractor

def predict_with_realdata_model(audio_file, verbose=True):
    """
    Predict using model trained on real phone-quality data

    Args:
        audio_file: Path to audio file
        verbose: Print detailed output

    Returns:
        dict: Prediction results
    """

    # Load model
    model_path = "models/saved_models/RealData_best.pkl"
    scaler_path = "models/saved_models/RealData_scaler.pkl"
    stats_path = "data/processed/real_data_feature_stats.json"

    if verbose:
        print("="*80)
        print("PARKINSON'S PREDICTION - REAL DATA MODEL")
        print("="*80)
        print(f"\n📁 Audio file: {audio_file}")
        print(f"🤖 Model: Trained on 81 phone-quality HC/PD recordings")

    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Extract features
    if verbose:
        print(f"\n🔊 Extracting features...")

    extractor = AudioFeatureExtractor(validate_quality=False)
    features_dict = extractor.extract_features(audio_file, return_dict=True)

    # Convert to array (match training order)
    with open(stats_path, 'r') as f:
        feature_stats = json.load(f)

    feature_names = list(feature_stats.keys())
    features = np.array([features_dict.get(name, 0.0) for name in feature_names])

    if verbose:
        print(f"   ✓ Extracted {len(features)} features")

    # Normalize
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    hc_prob = probabilities[0]
    pd_prob = probabilities[1]

    # Results
    result = {
        'file': audio_file,
        'prediction': int(prediction),
        'hc_probability': float(hc_prob),
        'pd_probability': float(pd_prob),
        'predicted_label': 'PD' if prediction == 1 else 'HC'
    }

    if verbose:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        pred_label = "🔴 PARKINSON'S (PD)" if prediction == 1 else "🔵 HEALTHY (HC)"
        print(f"\n📊 Prediction: {pred_label}")
        print(f"\n   Probability Breakdown:")
        print(f"      Healthy (HC):      {hc_prob*100:5.1f}%")
        print(f"      Parkinson's (PD):  {pd_prob*100:5.1f}%")

        # Risk assessment
        if pd_prob < 0.30:
            risk = "LOW"
            emoji = "✅"
            advice = "Voice features appear normal for phone-quality recording"
        elif pd_prob < 0.50:
            risk = "MODERATE-LOW"
            emoji = "⚠️"
            advice = "Some indicators present, but inconclusive"
        elif pd_prob < 0.70:
            risk = "MODERATE-HIGH"
            emoji = "🟠"
            advice = "Several indicators detected, consider medical consultation"
        else:
            risk = "HIGH"
            emoji = "🔴"
            advice = "Strong indicators present, recommend neurologist assessment"

        print(f"\n{emoji} Risk Level: {risk}")
        print(f"\n💡 Interpretation:")
        print(f"   {advice}")

        print("\n" + "-"*80)
        print("⚠️  IMPORTANT NOTES:")
        print("   • Model trained on only 81 samples (limited accuracy)")
        print("   • Test accuracy was ~48-60% (low due to small dataset)")
        print("   • Phone recordings show minimal HC vs PD differences")
        print("   • This is NOT a medical diagnosis - consult a doctor!")
        print("="*80 + "\n")

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_realdata.py <audio_file.wav>")
        print("\nExample:")
        print("  python3 predict_realdata.py tanvir.wav")
        print("  python3 predict_realdata.py mahin.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not Path(audio_file).exists():
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)

    predict_with_realdata_model(audio_file, verbose=True)
