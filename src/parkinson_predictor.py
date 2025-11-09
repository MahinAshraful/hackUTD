"""
End-to-End Parkinson's Disease Prediction Pipeline
Audio → Features → ML Model → Risk Assessment
"""

import numpy as np
import pickle
import json
import os
from pathlib import Path
from src.audio_feature_extractor import AudioFeatureExtractor, AudioQualityError

class ParkinsonPredictor:
    """
    Complete pipeline for Parkinson's disease risk prediction from voice
    """

    def __init__(self, model_path=None, feature_stats_path=None):
        # Set default paths relative to project root
        # Using RealData model trained on phone recordings (not UCI lab data)
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / 'models/saved_models/RealData_best.pkl'
        if feature_stats_path is None:
            project_root = Path(__file__).parent.parent
            feature_stats_path = project_root / 'data/processed/real_data_feature_stats.json'
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained model file
            feature_stats_path: Path to feature normalization stats
        """
        print("🚀 Initializing Parkinson's Predictor...")

        # Load trained model
        print(f"   → Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature statistics for normalization
        print(f"   → Loading feature stats: {feature_stats_path}")
        with open(feature_stats_path, 'r') as f:
            self.feature_stats = json.load(f)

        # Initialize feature extractor
        self.extractor = AudioFeatureExtractor(
            validate_quality=True,
            min_duration=3.0,
            target_sr=22050,
            min_snr=10.0
        )

        # Get model info
        model_name = self.model.__class__.__name__
        print(f"   ✓ Model loaded: {model_name}")
        print(f"   ✓ Ready to predict!\n")

    def normalize_features(self, features):
        """
        Normalize features using training statistics

        Args:
            features: numpy array of 44 features

        Returns:
            numpy array: normalized features
        """
        normalized = np.zeros_like(features)

        for i, feature_name in enumerate(self.extractor.feature_names):
            if feature_name in self.feature_stats:
                mean = self.feature_stats[feature_name]['mean']
                std = self.feature_stats[feature_name]['std']

                if std > 0:
                    normalized[i] = (features[i] - mean) / std
                else:
                    normalized[i] = 0
            else:
                normalized[i] = features[i]

        return normalized

    def predict(self, audio_path, return_details=True):
        """
        Predict Parkinson's disease risk from audio file

        Args:
            audio_path: Path to audio file
            return_details: If True, return detailed results

        Returns:
            dict: Prediction results
        """
        print("=" * 60)
        print("PARKINSON'S DISEASE RISK ASSESSMENT")
        print("=" * 60)

        results = {
            'audio_path': str(audio_path),
            'success': False
        }

        try:
            # Step 1: Extract features
            features = self.extractor.extract_features(audio_path, return_dict=False)
            results['raw_features'] = features.tolist()

            # Step 2: Normalize features
            print("\n   → Normalizing features...")
            normalized_features = self.normalize_features(features)
            results['normalized_features'] = normalized_features.tolist()

            # Step 3: Make prediction
            print("   → Running ML model...")
            prediction = self.model.predict(normalized_features.reshape(1, -1))[0]
            probability = self.model.predict_proba(normalized_features.reshape(1, -1))[0]

            # Step 4: Calculate risk score
            pd_probability = probability[1]  # Probability of Parkinson's
            healthy_probability = probability[0]

            results['prediction'] = int(prediction)
            results['pd_probability'] = float(pd_probability)
            results['healthy_probability'] = float(healthy_probability)
            results['success'] = True

            # Step 5: Risk assessment
            risk_level, recommendation = self._assess_risk(pd_probability)
            results['risk_level'] = risk_level
            results['recommendation'] = recommendation

            # Print results
            self._print_results(results)

            # Get feature importance if available
            if return_details and hasattr(self.model, 'coef_'):
                results['feature_importance'] = self._get_feature_importance()

            return results

        except AudioQualityError as e:
            print(f"\n❌ Audio Quality Error: {e}")
            results['error'] = str(e)
            results['error_type'] = 'quality'
            return results

        except Exception as e:
            print(f"\n❌ Prediction Error: {e}")
            results['error'] = str(e)
            results['error_type'] = 'processing'
            return results

    def _assess_risk(self, pd_probability):
        """
        Assess risk level and provide recommendation

        Args:
            pd_probability: Probability of Parkinson's (0-1)

        Returns:
            tuple: (risk_level, recommendation)
        """
        if pd_probability < 0.3:
            return "LOW", "Normal voice characteristics. Routine checkup recommended."

        elif pd_probability < 0.6:
            return "MODERATE", "Some indicators present. Monitor and retest in 3-6 months."

        elif pd_probability < 0.8:
            return "HIGH", "Significant indicators detected. Consult neurologist for clinical assessment."

        else:
            return "VERY HIGH", "Strong indicators present. Urgent neurologist referral recommended."

    def _print_results(self, results):
        """Print formatted prediction results"""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if not results['success']:
            print(f"❌ Prediction failed: {results.get('error', 'Unknown error')}")
            return

        pd_prob = results['pd_probability']
        risk = results['risk_level']

        # Risk indicator
        risk_colors = {
            'LOW': '✅',
            'MODERATE': '⚠️',
            'HIGH': '🔴',
            'VERY HIGH': '🚨'
        }

        print(f"\n{risk_colors.get(risk, '❓')} RISK LEVEL: {risk}")
        print(f"\n📊 Parkinson's Probability: {pd_prob * 100:.1f}%")
        print(f"   Healthy Probability: {results['healthy_probability'] * 100:.1f}%")

        print(f"\n💡 Recommendation:")
        print(f"   {results['recommendation']}")

        print("\n" + "=" * 60)

    def _get_feature_importance(self):
        """
        Get feature importance from model coefficients

        Returns:
            dict: Top features contributing to prediction
        """
        if not hasattr(self.model, 'coef_'):
            return None

        coefficients = self.model.coef_[0]
        feature_importance = {
            name: abs(float(coef))
            for name, coef in zip(self.extractor.feature_names, coefficients)
        }

        # Sort by importance
        sorted_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10

        return sorted_features

    def predict_batch(self, audio_paths):
        """
        Predict for multiple audio files

        Args:
            audio_paths: List of audio file paths

        Returns:
            list: List of prediction results
        """
        print(f"\n{'=' * 60}")
        print(f"BATCH PREDICTION: {len(audio_paths)} files")
        print("=" * 60)

        results = []
        for i, path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Processing: {path}")
            result = self.predict(path, return_details=False)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'=' * 60}")
        print(f"BATCH COMPLETE: {successful}/{len(audio_paths)} successful")
        print("=" * 60)

        return results

    def save_report(self, results, output_path):
        """
        Save prediction results to JSON file

        Args:
            results: Prediction results dict
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Report saved: {output_path}")


def demo_predictor():
    """Demo function showing usage"""
    print("=" * 60)
    print("PARKINSON'S PREDICTOR DEMO")
    print("=" * 60)

    print("\n📋 How to Use:")
    print("""
    # Initialize predictor
    predictor = ParkinsonPredictor()

    # Predict from audio file
    result = predictor.predict('voice_recording.wav')

    # Result contains:
    {
        'prediction': 0 or 1,              # 0=Healthy, 1=Parkinson's
        'pd_probability': 0.87,             # 87% chance of PD
        'risk_level': 'HIGH',               # Risk category
        'recommendation': '...',            # Action to take
        'feature_importance': {...}         # Top contributing features
    }

    # Batch prediction
    results = predictor.predict_batch(['file1.wav', 'file2.wav'])

    # Save report
    predictor.save_report(result, 'patient_report.json')
    """)

    print("\n" + "=" * 60)
    print("System Requirements:")
    print("   • Audio file: WAV format recommended")
    print("   • Duration: Minimum 3 seconds")
    print("   • Content: Sustained vowel 'Ahhhhh' or speech")
    print("   • Quality: Clear recording, minimal background noise")
    print("=" * 60)


if __name__ == "__main__":
    demo_predictor()
