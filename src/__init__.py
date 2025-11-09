"""
Parkinson's Disease Detection - Source Package
"""

# DEMO MODE: Optional imports - only load if dependencies available
try:
    from .audio_feature_extractor import AudioFeatureExtractor
    from .parkinson_predictor import ParkinsonPredictor
    __all__ = ['AudioFeatureExtractor', 'ParkinsonPredictor']
except ImportError as e:
    # Running in demo mode without ML dependencies
    print(f"⚠️  ML dependencies not available: {e}")
    print("   Running in DEMO MODE")
    __all__ = []
