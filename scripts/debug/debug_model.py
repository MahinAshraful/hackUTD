"""Debug script to examine feature extraction and normalization"""
import numpy as np
import pickle
import json
from src.audio_feature_extractor import AudioFeatureExtractor
from src.parkinson_predictor import ParkinsonPredictor

# Test files
test_files = ['mahintest.wav', 'mahin.wav']

print("=" * 80)
print("DEBUGGING MODEL PREDICTIONS")
print("=" * 80)

# Initialize predictor
predictor = ParkinsonPredictor()

for audio_file in test_files:
    print(f"\n{'=' * 80}")
    print(f"FILE: {audio_file}")
    print("=" * 80)

    # Extract features
    features = predictor.extractor.extract_features(audio_file, return_dict=True)

    # Show raw features
    print("\nRAW FEATURES (first 10):")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name:15s}: {value:10.4f}")

    # Convert to array
    feature_array = np.array([features.get(name, 0.0) for name in predictor.extractor.feature_names])

    # Normalize
    normalized = predictor.normalize_features(feature_array)

    print("\nNORMALIZED FEATURES (first 10):")
    for i, name in enumerate(predictor.extractor.feature_names[:10]):
        print(f"  {name:15s}: {normalized[i]:10.4f}")

    # Make prediction
    prediction = predictor.model.predict(normalized.reshape(1, -1))[0]
    probability = predictor.model.predict_proba(normalized.reshape(1, -1))[0]

    print(f"\nMODEL OUTPUT:")
    print(f"  Prediction: {prediction} ({'Parkinson' if prediction == 1 else 'Healthy'})")
    print(f"  Probabilities: Healthy={probability[0]:.4f}, PD={probability[1]:.4f}")

    # Check for issues
    print(f"\nDIAGNOSTICS:")
    print(f"  Zero features in raw: {np.sum(feature_array == 0)}/44")
    print(f"  NaN in normalized: {np.sum(np.isnan(normalized))}")
    print(f"  Inf in normalized: {np.sum(np.isinf(normalized))}")
    print(f"  Min normalized: {np.min(normalized):.4f}")
    print(f"  Max normalized: {np.max(normalized):.4f}")
    print(f"  Mean normalized: {np.mean(normalized):.4f}")

    # Check specific problem features
    print(f"\nKEY FEATURES:")
    for feat_name in ['Jitter_rel', 'HNR05', 'MFCC0', 'RPDE', 'DFA', 'PPE']:
        idx = predictor.extractor.feature_names.index(feat_name)
        raw_val = feature_array[idx]
        norm_val = normalized[idx]
        print(f"  {feat_name:15s}: raw={raw_val:10.4f}, normalized={norm_val:10.4f}")

print("\n" + "=" * 80)
print("CHECKING TRAINING STATISTICS")
print("=" * 80)

# Load feature stats
with open('data/processed/feature_stats.json', 'r') as f:
    stats = json.load(f)

print("\nTRAINING STATS (first 10 features):")
for name in predictor.extractor.feature_names[:10]:
    if name in stats:
        print(f"  {name:15s}: mean={stats[name]['mean']:10.4f}, std={stats[name]['std']:10.4f}")
