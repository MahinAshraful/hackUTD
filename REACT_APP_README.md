# React Frontend Setup

Basic React app for Parkinson's Detection

## Quick Start

### 1. Backend (Flask API)

Already running at http://localhost:5001

If not running:
```bash
python3 backend/app.py
```

### 2. Frontend (React)

```bash
cd frontend
npm start
```

Opens at http://localhost:3000

## What It Does

- Upload audio files (WAV, MP3, OGG, WEBM)
- Record voice (5 seconds)
- Display results (hardcoded "No Parkinson's")
- Show voice features

## Structure

```
frontend/
├── package.json           # Dependencies
├── public/
│   └── index.html        # HTML template
└── src/
    ├── index.js          # Entry point
    └── App.js            # Main component
```

## Features

- Basic React hooks (useState, useRef)
- File upload
- Voice recording with MediaRecorder API
- Fetch API for backend calls
- Inline styles (no CSS files)

## API Integration

App connects to Flask API at `http://localhost:5001/api`

Endpoints used:
- POST /api/predict - Upload audio

## Development

Run dev server:
```bash
cd frontend
npm start
```

Build for production:
```bash
cd frontend
npm run build
```

Production build goes to `frontend/build/`

## Current Status

- [x] Basic React setup
- [x] File upload
- [x] Voice recording
- [x] API integration
- [x] Results display
- [ ] Real ML model integration (hardcoded for now)

## Next Steps

To integrate real model predictions, update `backend/app.py`:

```python
# Replace hardcoded result with actual model
from predict_realdata import predict_with_realdata_model

result = predict_with_realdata_model(filepath, verbose=False)
```
