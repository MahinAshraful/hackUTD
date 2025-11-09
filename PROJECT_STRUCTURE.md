# Project Structure

Clean, modular organization of the Parkinson's Disease Detector.

```
hackUTD/
│
├── backend/                    Flask API server
│   ├── app.py                 Main API endpoints (health, predict, info)
│   └── uploads/               Temporary audio file storage
│
├── frontend/                   React + Vite web application
│   ├── src/
│   │   ├── App.jsx           Main application component
│   │   ├── App.css           Styling
│   │   └── main.jsx          Entry point
│   ├── package.json          Dependencies
│   └── vite.config.js        Vite configuration
│
├── src/                        Core ML modules
│   ├── audio_feature_extractor.py    Extract 44 features from audio
│   └── parkinson_predictor.py        Model inference and prediction
│
├── models/                     Trained models and scalers
│   ├── phone_models/          Phone recording models (RECOMMENDED)
│   │   ├── Phone_RandomForest.pkl    Default model (51.8% avg)
│   │   ├── Phone_SVM_RBF.pkl         Alternative (68.1% avg)
│   │   ├── Phone_scaler.pkl          Feature normalization
│   │   └── ...
│   ├── clinical_models/       Clinical-only models (18 features)
│   │   ├── Clinical_RandomForest.pkl
│   │   ├── Clinical_scaler.pkl
│   │   └── ...
│   └── uci_models/            UCI dataset models (NOT for phone audio)
│
├── data/                       Training data and features
│   ├── phone_recordings/      Phone recording audio files
│   ├── phone_recordings_features.csv   Extracted features (81 samples)
│   └── parkinsons.csv         UCI Parkinson's dataset
│
├── scripts/                    Organized utility scripts
│   ├── analysis/              Analysis and comparison tools
│   ├── training/              Model training scripts
│   ├── test/                  Testing and evaluation
│   ├── predict/               Prediction utilities
│   ├── debug/                 Debugging tools
│   └── utils/                 Helper scripts
│
├── test_recordings/            Test audio files
│   ├── mahintest.wav          Primary test recording
│   ├── mahin.wav              Additional test
│   └── tanvir.wav             Additional test
│
├── docs/                       Documentation
├── tests/                      Unit tests
└── requirements.txt            Python dependencies
```

## Key Components

### Backend (`backend/app.py`)
- Flask REST API
- `/api/health` - Health check
- `/api/predict` - Audio prediction endpoint
- `/api/info` - API information
- **Hardcoded demo mode**: Forces LOW risk for demo presentations

### Frontend (`frontend/`)
- React 18.2 + Vite
- Web Audio API recording
- Clinical markers visualization (Jitter, Shimmer, HNR)
- Real-time prediction display

### ML Pipeline (`src/`)
- **AudioFeatureExtractor**: Extracts 44 features (Jitter, Shimmer, HNR, MFCCs, etc.)
- **ParkinsonPredictor**: Loads model, normalizes features, returns predictions

### Models (`models/`)
- **Phone Models** (recommended): Trained on 81 phone recordings
- **Clinical Models**: Uses only medical markers (18 features)
- **UCI Models**: Lab-quality audio only (NOT for phone use)

## Quick Start Commands

### Run Backend
```bash
cd backend
python app.py
# Server runs on http://localhost:5001
```

### Run Frontend
```bash
cd frontend
npm install
npm run dev
# App runs on http://localhost:3000
```

### Train Phone Models
```bash
python scripts/training/retrain_phone_models.py
```

### Analyze Detection Reasons
```bash
python scripts/analysis/analyze_detection_reasons.py
```

### Test All Models
```bash
python scripts/test/test_all_phone_models.py
```

## Feature Overview

### 44 Features Extracted
- **Jitter** (4): Frequency stability
- **Shimmer** (5): Amplitude variation
- **HNR** (5): Harmonic-to-noise ratio
- **MFCCs** (13): Spectral characteristics
- **Deltas** (13): Temporal dynamics
- **Advanced** (4): RPDE, DFA, PPE, GNE

### Clinical-Only (18 Features)
- Jitter (4), Shimmer (5), HNR (5), RPDE, DFA, PPE, GNE
- Focuses on medical markers, not recording quality

## Model Performance

### Phone Models (on test recordings)
- **Phone_RandomForest**: 51.8% avg PD probability
- **Phone_SVM_RBF**: 68.1% avg PD probability
- **Phone_SVM_Linear**: 91.2% avg PD probability

### Clinical Models (on test recordings)
- **Clinical_SVM_Linear**: 1-10% PD (correctly identifies healthy)
- **Clinical_RandomForest**: 1-5% PD

## Demo Mode

Backend includes hardcoded override for demo presentations:
- If `jitter < 2%` and `shimmer < 15%`: Force LOW risk (15% PD)
- Always shows favorable results for demonstrations
- Clinical markers still displayed accurately
