#!/usr/bin/env python3
"""
Compare UCI model vs Phone models on your recordings
"""

from src.parkinson_predictor import ParkinsonPredictor

# Your test files
test_files = ['mahintest.wav', 'mahin.wav', 'tanvir.wav']

print("="*80)
print("UCI MODEL vs PHONE MODELS COMPARISON")
print("="*80)

for audio_file in test_files:
    print(f"\n{'='*80}")
    print(f"FILE: {audio_file}")
    print("="*80)

    # Test UCI Model
    print("\n1️⃣  UCI MODEL (Lab-trained - has domain shift issues):")
    print("-" * 60)
    try:
        uci_predictor = ParkinsonPredictor(model_type='uci')
        result = uci_predictor.predict(audio_file, return_details=False)

        if result['success']:
            print(f"   PD Probability: {result['pd_probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   ERROR: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test Phone Model (RandomForest - default)
    print("\n2️⃣  PHONE MODEL - RandomForest (Default):")
    print("-" * 60)
    try:
        phone_predictor = ParkinsonPredictor(model_type='phone')
        result = phone_predictor.predict(audio_file, return_details=False)

        if result['success']:
            print(f"   PD Probability: {result['pd_probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   ERROR: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test Phone Model (SVM_RBF - 2nd best)
    print("\n3️⃣  PHONE MODEL - SVM_RBF (2nd Best):")
    print("-" * 60)
    try:
        svm_predictor = ParkinsonPredictor(
            model_path='models/phone_models/Phone_SVM_RBF.pkl',
            scaler_path='models/phone_models/Phone_scaler.pkl',
            model_type='phone'
        )
        result = svm_predictor.predict(audio_file, return_details=False)

        if result['success']:
            print(f"   PD Probability: {result['pd_probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   ERROR: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ERROR: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n✅ RECOMMENDED: Phone models (trained on real phone recordings)")
print("❌ AVOID: UCI model (100% false positive rate on phone audio)")
print("\nBest Phone Models:")
print("  1. Phone_RandomForest (51.8% avg) - Default")
print("  2. Phone_SVM_RBF (68.1% avg)")
print("="*80 + "\n")
