#!/usr/bin/env python3
"""
Test Existing Models on Real HC vs PD Recordings
Evaluate how well our trained models perform on the real dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.audio_feature_extractor import AudioFeatureExtractor
from src.parkinson_predictor import ParkinsonPredictor

def load_model(model_path):
    """Load a trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def test_real_recordings():
    """Test existing models on real HC vs PD recordings"""

    print("\n" + "="*80)
    print("TESTING EXISTING MODELS ON REAL HC vs PD RECORDINGS")
    print("="*80)

    # Paths
    hc_dir = Path("data/new_data/HC_AH")
    pd_dir = Path("data/new_data/PD_AH")

    hc_files = sorted(list(hc_dir.glob("*.wav")))
    pd_files = sorted(list(pd_dir.glob("*.wav")))

    print(f"\n📊 Dataset:")
    print(f"   Healthy Controls (HC): {len(hc_files)} files")
    print(f"   Parkinson's (PD):      {len(pd_files)} files")

    # Load models
    print(f"\n🔧 Loading models...")

    models_to_test = []

    # Main model (44 features)
    main_model_path = Path("models/saved_models/LogisticRegression_L2_best.pkl")
    if main_model_path.exists():
        try:
            # Create predictor (will initialize with validate_quality=True internally)
            predictor = ParkinsonPredictor(model_path=str(main_model_path))
            # Disable quality validation for batch processing
            predictor.extractor.validate_quality = False
            models_to_test.append({
                'name': 'Logistic Regression (44 features)',
                'path': main_model_path,
                'predictor': predictor
            })
            print(f"   ✓ Loaded: Logistic Regression (44 features)")
        except Exception as e:
            print(f"   ✗ Failed to load Logistic Regression: {e}")

    # Simple model (14 features)
    simple_model_path = Path("models/saved_models/SimpleAcoustic_best.pkl")
    if simple_model_path.exists():
        try:
            # Check if this model has different feature stats
            simple_stats_path = Path("data/processed/simple_feature_stats.json")
            if simple_stats_path.exists():
                predictor = ParkinsonPredictor(
                    model_path=str(simple_model_path),
                    feature_stats_path=str(simple_stats_path)
                )
            else:
                # Use default feature stats
                predictor = ParkinsonPredictor(model_path=str(simple_model_path))

            predictor.extractor.validate_quality = False
            models_to_test.append({
                'name': 'Simple Acoustic (14 features)',
                'path': simple_model_path,
                'predictor': predictor
            })
            print(f"   ✓ Loaded: Simple Acoustic (14 features)")
        except Exception as e:
            print(f"   ✗ Failed to load Simple Acoustic: {e}")

    if not models_to_test:
        print("   ❌ No models found!")
        return

    # Test each model
    for model_info in models_to_test:
        print("\n" + "="*80)
        print(f"MODEL: {model_info['name']}")
        print("="*80)

        predictor = model_info['predictor']

        # Test HC files
        print(f"\n📊 Testing on {len(hc_files)} HC (Healthy) recordings...")
        hc_results = []
        hc_failed = 0

        for i, file in enumerate(hc_files, 1):
            try:
                # Suppress print output during batch processing
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    result = predictor.predict(str(file), return_details=False)

                if result.get('success'):
                    hc_results.append(result['pd_probability'])
                else:
                    hc_failed += 1

                if i % 10 == 0:
                    print(f"   {i}/{len(hc_files)} done...")
            except Exception as e:
                hc_failed += 1

        # Test PD files
        print(f"\n📊 Testing on {len(pd_files)} PD (Parkinson's) recordings...")
        pd_results = []
        pd_failed = 0

        for i, file in enumerate(pd_files, 1):
            try:
                # Suppress print output during batch processing
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    result = predictor.predict(str(file), return_details=False)

                if result.get('success'):
                    pd_results.append(result['pd_probability'])
                else:
                    pd_failed += 1

                if i % 10 == 0:
                    print(f"   {i}/{len(pd_files)} done...")
            except Exception as e:
                pd_failed += 1

        # Calculate statistics
        print(f"\n{'='*80}")
        print(f"RESULTS FOR: {model_info['name']}")
        print(f"{'='*80}")

        if hc_results:
            hc_mean = np.mean(hc_results) * 100
            hc_std = np.std(hc_results) * 100
            hc_classified_healthy = sum(1 for p in hc_results if p < 0.5)
            hc_accuracy = (hc_classified_healthy / len(hc_results)) * 100

            print(f"\n🔵 HEALTHY CONTROLS (HC) - Should predict LOW risk:")
            print(f"   Files tested: {len(hc_results)}/{len(hc_files)}")
            print(f"   Average PD probability: {hc_mean:.1f}% ± {hc_std:.1f}%")
            print(f"   Classified as Healthy (<50%): {hc_classified_healthy}/{len(hc_results)} ({hc_accuracy:.1f}%)")
            print(f"   Classified as PD (≥50%): {len(hc_results) - hc_classified_healthy}/{len(hc_results)} ({100-hc_accuracy:.1f}%)")

            if hc_accuracy < 50:
                print(f"   ❌ POOR: Model thinks healthy people have Parkinson's!")
            elif hc_accuracy < 70:
                print(f"   ⚠️  FAIR: Some healthy misclassified")
            else:
                print(f"   ✅ GOOD: Most healthy correctly identified")

        if pd_results:
            pd_mean = np.mean(pd_results) * 100
            pd_std = np.std(pd_results) * 100
            pd_classified_pd = sum(1 for p in pd_results if p >= 0.5)
            pd_accuracy = (pd_classified_pd / len(pd_results)) * 100

            print(f"\n🔴 PARKINSON'S DISEASE (PD) - Should predict HIGH risk:")
            print(f"   Files tested: {len(pd_results)}/{len(pd_files)}")
            print(f"   Average PD probability: {pd_mean:.1f}% ± {pd_std:.1f}%")
            print(f"   Classified as PD (≥50%): {pd_classified_pd}/{len(pd_results)} ({pd_accuracy:.1f}%)")
            print(f"   Classified as Healthy (<50%): {len(pd_results) - pd_classified_pd}/{len(pd_results)} ({100-pd_accuracy:.1f}%)")

            if pd_accuracy < 50:
                print(f"   ❌ POOR: Model misses Parkinson's cases!")
            elif pd_accuracy < 70:
                print(f"   ⚠️  FAIR: Some PD cases missed")
            else:
                print(f"   ✅ GOOD: Most PD correctly identified")

        if hc_results and pd_results:
            overall_accuracy = ((hc_classified_healthy + pd_classified_pd) /
                               (len(hc_results) + len(pd_results))) * 100

            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   Total accuracy: {overall_accuracy:.1f}%")
            print(f"   HC accuracy: {hc_accuracy:.1f}%")
            print(f"   PD accuracy: {pd_accuracy:.1f}%")

            # Interpretation
            print(f"\n💡 INTERPRETATION:")
            if overall_accuracy < 60:
                print(f"   ❌ Model performs POORLY on this real-world data")
                print(f"   Likely reason: Training data (UCI) doesn't match real recordings")
            elif overall_accuracy < 75:
                print(f"   ⚠️  Model performance is MEDIOCRE")
                print(f"   Could be improved with better training data")
            else:
                print(f"   ✅ Model performs WELL on this data!")

            # Show distribution
            print(f"\n📈 PREDICTION DISTRIBUTION:")
            print(f"\n   HC (Healthy) predictions:")
            bins_hc = [0, 25, 50, 75, 100]
            for i in range(len(bins_hc)-1):
                count = sum(1 for p in hc_results if bins_hc[i] <= p*100 < bins_hc[i+1])
                pct = (count / len(hc_results)) * 100
                bar = "█" * int(pct / 5)
                print(f"      {bins_hc[i]:3d}-{bins_hc[i+1]:3d}%: {count:2d} files ({pct:5.1f}%) {bar}")

            print(f"\n   PD (Parkinson's) predictions:")
            for i in range(len(bins_hc)-1):
                count = sum(1 for p in pd_results if bins_hc[i] <= p*100 < bins_hc[i+1])
                pct = (count / len(pd_results)) * 100
                bar = "█" * int(pct / 5)
                print(f"      {bins_hc[i]:3d}-{bins_hc[i+1]:3d}%: {count:2d} files ({pct:5.1f}%) {bar}")

        # Save detailed results
        results_df = pd.DataFrame({
            'filename': [f.name for f in hc_files[:len(hc_results)]] + [f.name for f in pd_files[:len(pd_results)]],
            'true_label': ['HC'] * len(hc_results) + ['PD'] * len(pd_results),
            'pd_probability': hc_results + pd_results,
            'predicted_label': ['HC' if p < 0.5 else 'PD' for p in hc_results + pd_results]
        })

        output_file = f"data/test_results_{model_info['name'].replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")

    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print("\n💡 Key Insights:")
    print("   • If HC files are predicted as PD: Model is too sensitive (false positives)")
    print("   • If PD files are predicted as HC: Model misses disease (false negatives)")
    print("   • Low accuracy on both: Training data doesn't match real recordings")
    print("\n🎯 This tells us whether the UCI-trained model works on real phone recordings!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_real_recordings()
