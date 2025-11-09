# Scripts Directory

Organized collection of scripts for the Parkinson's Disease Detector.

## Directory Structure

```
scripts/
├── analysis/      Analysis and comparison scripts
├── debug/         Debugging and diagnostic tools
├── predict/       Prediction and inference scripts
├── training/      Model training scripts
├── test/          Testing and evaluation scripts
└── utils/         Utility and helper scripts
```

## Quick Reference

### Analysis Scripts (`analysis/`)
- `analyze_detection_reasons.py` - Shows why models detect Parkinson's with feature importance
- `analyze_mahintest.py` - Deep dive analysis on specific test recording
- `compare_uci_vs_phone.py` - Compare UCI vs Phone model predictions
- `analyze_preprocessing.py` - Analyze preprocessing pipeline
- `analyze_real_data.py` - Analyze real phone recording data
- `compare_extraction.py` - Compare feature extraction methods

### Training Scripts (`training/`)
- `retrain_phone_models.py` - **Main training script** for phone models (recommended)
- `train_clinical_only.py` - Train models using only clinical features (18 features)
- `train_on_real_data.py` - Train on real phone recording dataset
- `retrain_40_features.py` - Retrain with 40 features
- `retrain_simple_features.py` - Retrain with simplified feature set

### Testing Scripts (`test/`)
- `test_all_phone_models.py` - Test all phone models on recordings
- `test_real_data.py` - Test on real phone data
- `test_realdata_model.py` - Test real data model
- `test_realdata_with_scaler.py` - Test with scaler normalization

### Prediction Scripts (`predict/`)
- `predict.py` - Main prediction script
- `predict_realdata.py` - Predict on real data
- `predict_simple.py` - Simple prediction interface

### Debug Scripts (`debug/`)
- `debug_features.py` - Debug feature extraction
- `debug_model.py` - Debug model behavior
- `debug_predict.py` - Debug prediction pipeline
- `fix_features.py` - Fix feature issues

### Utility Scripts (`utils/`)
- `record_voice.py` - Record voice samples
- `run.py` - Main application runner
