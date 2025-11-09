#!/usr/bin/env python3
"""
Comprehensive analysis: Why are models detecting Parkinson's?
Shows feature values, importance, and decision reasoning
"""

import numpy as np
from src.parkinson_predictor import ParkinsonPredictor
from src.audio_feature_extractor import AudioFeatureExtractor

# Test files
test_files = ['mahintest.wav', 'mahin.wav', 'tanvir.wav']

# All phone models
models = [
    ('Phone_RandomForest', 'models/phone_models/Phone_RandomForest.pkl'),
    ('Phone_SVM_Linear', 'models/phone_models/Phone_SVM_Linear.pkl'),
    ('Phone_SVM_RBF', 'models/phone_models/Phone_SVM_RBF.pkl'),
    ('Phone_LogisticRegression_L2', 'models/phone_models/Phone_LogisticRegression_L2.pkl'),
]

print("="*80)
print("COMPREHENSIVE ANALYSIS: WHY MODELS DETECT PARKINSON'S")
print("="*80)

# First, extract and show raw features
extractor = AudioFeatureExtractor(validate_quality=False)

for audio_file in test_files:
    print(f"\n{'='*80}")
    print(f"FILE: {audio_file}")
    print("="*80)

    # Extract features
    print("\n📊 EXTRACTED VOICE FEATURES:")
    print("-" * 80)
    features_dict = extractor.extract_features(audio_file, return_dict=True)

    # Show key features
    key_features = [
        'Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ',
        'Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5',
        'HNR05', 'HNR15', 'HNR25',
        'RPDE', 'DFA', 'PPE',
        'MFCC0', 'MFCC1', 'MFCC2'
    ]

    for feat in key_features:
        if feat in features_dict:
            value = features_dict[feat]
            print(f"  {feat:20s}: {value:10.4f}")

    print("\n" + "="*80)
    print("MODEL PREDICTIONS & REASONING")
    print("="*80)

    # Test each model
    for model_name, model_path in models:
        print(f"\n🤖 {model_name}")
        print("-" * 80)

        try:
            predictor = ParkinsonPredictor(
                model_path=model_path,
                scaler_path='models/phone_models/Phone_scaler.pkl',
                model_type='phone'
            )

            result = predictor.predict(audio_file, return_details=True)

            if result['success']:
                pd_prob = result['pd_probability']
                print(f"  Prediction: {pd_prob:.1%} Parkinson's → {result['risk_level']}")

                # Show feature importance if available
                if 'feature_importance' in result and result['feature_importance']:
                    print(f"\n  🔍 Top Contributing Features:")
                    for i, (feat, imp) in enumerate(list(result['feature_importance'].items())[:8], 1):
                        # Get raw feature value
                        raw_val = features_dict.get(feat, 0)
                        print(f"    {i}. {feat:15s}: importance={imp:.4f}, value={raw_val:8.4f}")
                elif hasattr(predictor.model, 'feature_importances_'):
                    # RandomForest feature importances
                    importances = predictor.model.feature_importances_
                    feat_imp = {
                        name: float(imp)
                        for name, imp in zip(extractor.feature_names, importances)
                    }
                    sorted_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8])

                    print(f"\n  🔍 Top Important Features (RandomForest):")
                    for i, (feat, imp) in enumerate(sorted_imp.items(), 1):
                        raw_val = features_dict.get(feat, 0)
                        print(f"    {i}. {feat:15s}: importance={imp:.4f}, value={raw_val:8.4f}")
                else:
                    print(f"\n  (Feature importance not available for this model type)")
            else:
                print(f"  ERROR: {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  ERROR: {e}")

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print("\n🔍 KEY FINDINGS:")
print("\n1. VOICE QUALITY METRICS:")
print("   - Jitter (frequency stability): Higher = more unstable voice")
print("   - Shimmer (amplitude variation): Higher = less control")
print("   - HNR (Harmonic-to-Noise Ratio): Lower = more noise")
print("\n2. PHONE RECORDING CHARACTERISTICS:")
print("   - HNR typically 7-20 dB (vs 60 dB in lab)")
print("   - More background noise and variation")
print("\n3. MODEL BEHAVIOR:")
print("   - Models trained on 81 phone samples (small dataset)")
print("   - May have learned to be overly sensitive")
print("   - False positive rate still ~50-70% on healthy samples")
print("\n⚠️  RECOMMENDATION:")
print("   - Collect 200+ more diverse training samples")
print("   - Calibrate probability thresholds")
print("   - Add confidence intervals to predictions")
print("="*80 + "\n")
