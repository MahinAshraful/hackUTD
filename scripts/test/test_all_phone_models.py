#!/usr/bin/env python3
"""
Test all phone models on your test recordings
"""

import numpy as np
from pathlib import Path
from src.parkinson_predictor import ParkinsonPredictor

# Test files
test_files = ['mahintest.wav', 'mahin.wav', 'tanvir.wav']

# All phone models
phone_models = [
    'Phone_LogisticRegression_L2',
    'Phone_LogisticRegression_L1',
    'Phone_RandomForest',
    'Phone_GradientBoosting',
    'Phone_SVM_Linear',
    'Phone_SVM_RBF',
    'Phone_NeuralNet'
]

print("="*80)
print("TESTING ALL PHONE MODELS ON YOUR RECORDINGS")
print("="*80)

results = []

for model_name in phone_models:
    model_path = f'models/phone_models/{model_name}.pkl'
    scaler_path = 'models/phone_models/Phone_scaler.pkl'

    if not Path(model_path).exists():
        print(f"\n⚠️  {model_name} not found, skipping...")
        continue

    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print("="*80)

    try:
        predictor = ParkinsonPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            model_type='phone'
        )

        for audio_file in test_files:
            result = predictor.predict(audio_file, return_details=False)

            if result['success']:
                pd_prob = result['pd_probability']
                risk = result['risk_level']

                print(f"  {audio_file:15s} → PD: {pd_prob:5.1%}  Risk: {risk}")

                results.append({
                    'model': model_name,
                    'file': audio_file,
                    'pd_prob': pd_prob,
                    'risk': risk
                })
            else:
                print(f"  {audio_file:15s} → ERROR: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"  ERROR loading model: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY - Average PD Probability by Model")
print("="*80)

import pandas as pd
if results:
    df = pd.DataFrame(results)
    summary = df.groupby('model')['pd_prob'].agg(['mean', 'std', 'min', 'max'])
    summary['mean'] = summary['mean'] * 100
    summary['std'] = summary['std'] * 100
    summary['min'] = summary['min'] * 100
    summary['max'] = summary['max'] * 100
    summary.columns = ['Mean %', 'Std %', 'Min %', 'Max %']

    print("\n" + summary.to_string())

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find model with lowest average prediction (closest to healthy)
    best_model = summary['Mean %'].idxmin()
    best_mean = summary.loc[best_model, 'Mean %']

    print(f"\n✅ Model with lowest false positive rate: {best_model}")
    print(f"   Average PD Probability: {best_mean:.1f}%")

    if best_mean < 30:
        print(f"   ✓ Good performance - classifies your samples as LOW risk")
    elif best_mean < 60:
        print(f"   ⚠️  Moderate performance - classifies as MODERATE risk")
    else:
        print(f"   ❌ Poor performance - still has high false positive rate")
