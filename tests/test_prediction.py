"""
End-to-End Test: Generate Audio → Extract Features → Predict Risk
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.generate_test_audio import save_test_audio
from src.parkinson_predictor import ParkinsonPredictor

def run_full_test():
    """
    Complete end-to-end test of the Parkinson's prediction system
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "PARKINSON'S PREDICTION SYSTEM - FULL TEST")
    print("=" * 70)

    # Step 1: Generate test audio
    print("\n[STEP 1/3] Generating test audio file...")
    print("-" * 70)
    audio_file = save_test_audio('outputs/audio/test.wav')

    # Verify file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Could not create {audio_file}")
        return

    # Step 2: Initialize predictor
    print("\n[STEP 2/3] Initializing Parkinson's Predictor...")
    print("-" * 70)
    try:
        predictor = ParkinsonPredictor()
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return

    # Step 3: Run prediction
    print("\n[STEP 3/3] Running Prediction...")
    print("-" * 70)
    try:
        result = predictor.predict(audio_file, return_details=True)

        # Save detailed report
        if result['success']:
            report_file = 'outputs/reports/test_prediction_report.json'
            predictor.save_report(result, report_file)
            print(f"\n📄 Detailed report saved: {report_file}")

            # Print feature importance
            if 'feature_importance' in result and result['feature_importance']:
                print("\n📊 Top 5 Contributing Features:")
                for i, (feature, importance) in enumerate(
                    list(result['feature_importance'].items())[:5], 1
                ):
                    print(f"   {i}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    print("\n" + "=" * 70)
    print("✅ FULL TEST COMPLETE!")
    print("=" * 70)
    print("\n📁 Files created:")
    print(f"   • outputs/audio/test.wav - Synthetic voice recording")
    print(f"   • outputs/reports/test_prediction_report.json - Detailed results")
    print("\n💡 Next steps:")
    print("   • Try with real voice recordings")
    print("   • Test batch processing with multiple files")
    print("   • Build Phase 3: Nemotron AI Agent integration")
    print()

if __name__ == "__main__":
    run_full_test()
