"""Test RealData model with its scaler"""
import numpy as np
import pickle
from src.audio_feature_extractor import AudioFeatureExtractor

# Load model and scaler
with open('models/saved_models/RealData_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/saved_models/RealData_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"Model: {model.__class__.__name__}")
print(f"Scaler: {scaler.__class__.__name__}")

# Extract features
extractor = AudioFeatureExtractor(validate_quality=True)

test_files = ['mahintest.wav', 'mahin.wav']

for audio_file in test_files:
    print(f"\n{'=' * 60}")
    print(f"File: {audio_file}")
    print("=" * 60)

    # Extract features
    features = extractor.extract_features(audio_file, return_dict=False)

    # Normalize with scaler
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    print(f"Prediction: {prediction} ({'PD' if prediction == 1 else 'Healthy'})")
    print(f"Healthy: {probability[0]:.2%}, PD: {probability[1]:.2%}")
