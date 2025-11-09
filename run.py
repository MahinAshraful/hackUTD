#!/usr/bin/env python3
"""
Parkinson's Disease Detection - Main Entry Point
Simple command-line interface for predictions
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parkinson_predictor import ParkinsonPredictor

def main():
    parser = argparse.ArgumentParser(
        description='Parkinson\'s Disease Detection from Voice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s predict voice.wav
  %(prog)s predict voice.wav --output report.json
  %(prog)s test
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict from audio file')
    predict_parser.add_argument('audio_file', help='Path to audio file (WAV recommended)')
    predict_parser.add_argument('-o', '--output', help='Save report to JSON file')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run test with synthetic audio')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')

    args = parser.parse_args()

    if args.command == 'predict':
        run_prediction(args.audio_file, args.output)
    elif args.command == 'test':
        run_test()
    elif args.command == 'info':
        show_info()
    else:
        parser.print_help()

def run_prediction(audio_file, output_file=None):
    """Run prediction on audio file"""
    print("\n" + "="*60)
    print("PARKINSON'S DISEASE PREDICTION")
    print("="*60)

    try:
        # Initialize predictor
        predictor = ParkinsonPredictor()

        # Run prediction
        result = predictor.predict(audio_file, return_details=True)

        # Save report if requested
        if output_file and result['success']:
            predictor.save_report(result, output_file)

    except FileNotFoundError:
        print(f"\n❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

def run_test():
    """Run end-to-end test"""
    print("\n" + "="*60)
    print("RUNNING END-TO-END TEST")
    print("="*60)

    try:
        from tests.test_prediction import run_full_test
        run_full_test()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_info():
    """Show system information"""
    print("\n" + "="*60)
    print("PARKINSON'S DISEASE DETECTION SYSTEM")
    print("="*60)
    print("\n📊 Model: Logistic Regression (L2)")
    print("   • ROC-AUC: 91%")
    print("   • Accuracy: 90%")
    print("   • Recall: 100% (catches all PD cases)")
    print("   • Precision: 83%")

    print("\n🎙️  Audio Requirements:")
    print("   • Format: WAV (recommended), MP3, or common formats")
    print("   • Duration: Minimum 3 seconds, 5+ seconds ideal")
    print("   • Content: Sustained vowel 'Ahhhhh' or speech")
    print("   • Quality: Clear recording, minimal noise")

    print("\n📁 Project Structure:")
    print("   • src/         - Source code")
    print("   • tests/       - Test files")
    print("   • data/        - Training data")
    print("   • models/      - Trained models")
    print("   • outputs/     - Generated results")
    print("   • docs/        - Documentation")

    print("\n💡 Quick Start:")
    print("   python3 run.py test          # Run test")
    print("   python3 run.py predict voice.wav  # Predict")
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()
