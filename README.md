# Parkinson's Disease Detection from Voice

AI-powered early detection of Parkinson's disease using voice analysis. Record 5 seconds of "Ahhhhh" and get instant risk assessment.

## 🎯 Quick Start

```bash
# Run test
python3 run.py test

# Predict from your voice
python3 run.py predict my_voice.wav

# Get help
python3 run.py --help
```

## 📊 System Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 91% |
| **Accuracy** | 90% |
| **Recall** | 100% (catches all PD cases) |
| **Precision** | 83% |
| **Speed** | <2s per prediction |

**Model**: Logistic Regression (L2) - chosen for interpretability and clinical trust

## 🏗️ Project Structure

```
hackUTD/
├── src/                              # Source code
│   ├── audio_feature_extractor.py    # 44-feature extraction
│   ├── parkinson_predictor.py        # End-to-end prediction
│   └── train_models.py               # Model training pipeline
├── tests/                            # Test files
│   ├── generate_test_audio.py        # Synthetic audio generator
│   └── test_prediction.py            # End-to-end test
├── data/                             # Data files
│   ├── raw/                          # Original dataset
│   │   └── parkinsons_data.csv
│   └── processed/                    # Processed data
│       ├── train.csv
│       ├── test.csv
│       └── feature_stats.json
├── models/                           # Trained models
│   ├── saved_models/                 # Model files (.pkl)
│   ├── results/                      # Training results & visualizations
│   └── hyperparameters/              # Best hyperparameters
├── outputs/                          # Generated outputs
│   ├── audio/                        # Test audio files
│   └── reports/                      # Prediction reports
├── docs/                             # Documentation
│   ├── MODEL_SELECTION_PITCH.md      # Why we chose this model
│   └── PHASE2_COMPLETE.md            # Feature extraction docs
├── run.py                            # Main entry point
└── requirements.txt                  # Python dependencies
```

## 🚀 Installation

```bash
# Clone repository
cd hackUTD

# Install dependencies
pip3 install -r requirements.txt

# Verify installation
python3 run.py info
```

## 📖 Usage

### Basic Prediction

```python
from src.parkinson_predictor import ParkinsonPredictor

# Initialize
predictor = ParkinsonPredictor()

# Predict
result = predictor.predict('voice_recording.wav')

# Result:
# ✅ RISK LEVEL: MODERATE
# 📊 Parkinson's Probability: 54.2%
# 💡 Recommendation: Monitor and retest in 3-6 months
```

### Command Line

```bash
# Single prediction
python3 run.py predict voice.wav

# Save detailed report
python3 run.py predict voice.wav --output report.json

# Run system test
python3 run.py test

# Show system info
python3 run.py info
```

### Batch Processing

```python
from src.parkinson_predictor import ParkinsonPredictor

predictor = ParkinsonPredictor()

files = ['patient1.wav', 'patient2.wav', 'patient3.wav']
results = predictor.predict_batch(files)

for i, result in enumerate(results):
    print(f"Patient {i+1}: {result['risk_level']}")
```

## 🎙️ Audio Requirements

| Requirement | Specification |
|-------------|---------------|
| **Format** | WAV (recommended), MP3, or common formats |
| **Duration** | Minimum 3 seconds, 5+ seconds ideal |
| **Sample Rate** | 22,050 Hz or higher |
| **Content** | Sustained vowel "Ahhhhh" or continuous speech |
| **Quality** | Clear recording, minimal background noise |
| **SNR** | Minimum 10 dB signal-to-noise ratio |

## 🔬 How It Works

```
Audio Recording
       ↓
Quality Validation (duration, SNR, clipping)
       ↓
Feature Extraction (44 features)
   ↙         ↘
Parselmouth  Librosa
(18 features)(26 features)
   ↘         ↙
Combine & Normalize
       ↓
Logistic Regression Model
       ↓
Risk Assessment
(LOW/MODERATE/HIGH/VERY HIGH)
```

### 44 Voice Features

- **Jitter (4)**: Voice frequency stability
- **Shimmer (4)**: Voice amplitude variation
- **HNR (5)**: Harmonic-to-noise ratio
- **Other (5)**: RPDE, DFA, PPE, GNE, Shi_APQ11
- **MFCCs (13)**: Mel-frequency cepstral coefficients
- **Deltas (13)**: Temporal dynamics

## 📊 Risk Levels

| Level | Probability | Clinical Action |
|-------|-------------|-----------------|
| **LOW** | < 30% | Normal characteristics. Routine checkup. |
| **MODERATE** | 30-60% | Some indicators. Monitor, retest in 3-6 months. |
| **HIGH** | 60-80% | Significant indicators. Neurologist consultation. |
| **VERY HIGH** | > 80% | Strong indicators. Urgent referral. |

## 🧪 Testing

```bash
# Run full test suite
python3 run.py test

# Test individual components
python3 -m pytest tests/

# Generate test audio
python3 tests/generate_test_audio.py
```

## 📈 Model Training

We tested 14 models across 4 tiers:

1. **Gradient Boosting** (XGBoost, LightGBM, CatBoost)
2. **Tree Ensembles** (Random Forest, Extra Trees)
3. **Traditional ML** (SVM, Logistic Regression)
4. **Neural Networks** (2-layer, 3-layer)
5. **Ensembles** (Voting, Stacking)

**Winner**: Logistic Regression (L2)

**Why?**
- 91% ROC-AUC (nearly best)
- 100% recall (catches all PD cases)
- 90% accuracy (highest)
- Fast (0.05s training, <10ms inference)
- Interpretable (doctors can see why)

See [docs/MODEL_SELECTION_PITCH.md](docs/MODEL_SELECTION_PITCH.md) for details.

To retrain models:

```bash
python3 src/train_models.py
```

## 📄 Output Format

Prediction results are returned as JSON:

```json
{
  "success": true,
  "prediction": 1,
  "pd_probability": 0.87,
  "healthy_probability": 0.13,
  "risk_level": "HIGH",
  "recommendation": "Significant indicators detected. Consult neurologist.",
  "feature_importance": {
    "MFCC2": 0.243,
    "RPDE": 0.235,
    "PPE": 0.214
  },
  "raw_features": [...],
  "normalized_features": [...]
}
```

## 🔧 Development

```bash
# Project structure
make structure    # (or manually create folders)

# Code style
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

## 📚 Documentation

- [Model Selection Pitch](docs/MODEL_SELECTION_PITCH.md) - Why Logistic Regression won
- [Phase 2 Complete](docs/PHASE2_COMPLETE.md) - Audio feature extraction details

## 🛣️ Roadmap

- [x] **Phase 1**: ML Model Training (14 models tested)
- [x] **Phase 2**: Audio Feature Extraction (44 features)
- [ ] **Phase 3**: Nemotron AI Agent (intelligent reasoning)
- [ ] **Phase 4**: Backend API (FastAPI + PostgreSQL)
- [ ] **Phase 5**: Frontend (React web app)

## ⚠️ Disclaimer

This system is a **screening tool**, not a diagnostic device. Always consult qualified medical professionals for clinical diagnosis and treatment of Parkinson's disease.

## 📜 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with**: Python, Parselmouth, Librosa, Scikit-learn
**Dataset**: UCI Parkinson's Dataset (80 patients, 44 features)
**Model**: Logistic Regression (L2) - 91% ROC-AUC
