"""Test with the RealData model trained on phone recordings"""
import numpy as np
import pickle
from src.parkinson_predictor import ParkinsonPredictor

# Test both models
models_to_test = [
    ('UCI Dataset Model (Lab Quality)', 'models/saved_models/LogisticRegression_L2_best.pkl'),
    ('Real Data Model (Phone Quality)', 'models/saved_models/RealData_best.pkl'),
]

test_files = ['mahintest.wav', 'mahin.wav', 'tanvir.wav']

for model_name, model_path in models_to_test:
    print("=" * 80)
    print(f"TESTING: {model_name}")
    print("=" * 80)

    try:
        predictor = ParkinsonPredictor(model_path=model_path)

        for audio_file in test_files:
            print(f"\n  File: {audio_file}")
            result = predictor.predict(audio_file, return_details=False)

            if result['success']:
                print(f"    PD Probability: {result['pd_probability']:.2%}")
                print(f"    Risk Level: {result['risk_level']}")
            else:
                print(f"    ERROR: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"  ERROR loading model: {e}")

    print()
