"""Compare feature extraction with CSV data"""
import pandas as pd
import numpy as np
from src.audio_feature_extractor import AudioFeatureExtractor
from pathlib import Path

# Load pre-extracted features
csv_features = pd.read_csv('data/real_data_features.csv')
print("Pre-extracted features (first HC sample):")
print(csv_features.iloc[0][['Jitter_rel', 'Shim_loc', 'HNR05', 'MFCC0']].to_dict())

# Extract features from first HC audio file
hc_files = list(Path('data/new_data/HC_AH').glob('*.wav'))
if hc_files:
    print(f"\nExtracting features from: {hc_files[0].name}")
    extractor = AudioFeatureExtractor(validate_quality=False)
    features = extractor.extract_features(str(hc_files[0]), return_dict=True)
    print(f"Newly extracted features:")
    print({
        'Jitter_rel': features['Jitter_rel'],
        'Shim_loc': features['Shim_loc'],
        'HNR05': features['HNR05'],
        'MFCC0': features['MFCC0']
    })

# Compare
print("\nDifference analysis:")
print(f"CSV has {len(csv_features)} rows")
print(f"CSV columns: {list(csv_features.columns[:10])}")
