# Parkinson's Detection Web App

A simple web application for voice-based Parkinson's disease detection.

## Architecture

```
Frontend (HTML/JS) <---> Flask API <---> ML Model (Future)
     :5500                :5000
```

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip3 install -r requirements.txt
```

### 2. Start Backend Server

```bash
cd backend
python3 app.py
```

You should see:
```
PARKINSON'S DETECTION API
Starting Flask server...
API will be available at: http://localhost:5000
```

### 3. Start Frontend

Option A - Using Python's built-in server:
```bash
cd frontend
python3 -m http.server 5500
```

Option B - Just open the HTML file:
```bash
open frontend/index.html
```

### 4. Access the App

Open your browser and go to:
- **Frontend**: http://localhost:5500
- **API Health Check**: http://localhost:5000/api/health

## Features

### Current (Hardcoded Demo)

- Upload audio files (WAV, MP3, OGG, WEBM)
- Record voice directly from browser (5 seconds)
- Display results with confidence scores
- Show voice features (Jitter, Shimmer, HNR)
- Clean, modern UI
- Mobile responsive

### Planned (Integration Phase)

- Real ML model prediction
- Feature extraction from audio
- Detailed analysis report
- History tracking
- Export results as PDF

## API Endpoints

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "message": "Parkinson's Detection API is running"
}
```

### POST /api/predict
Upload audio for prediction

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `audio` file field

**Response:**
```json
{
  "success": true,
  "prediction": "healthy",
  "confidence": 0.92,
  "details": {
    "parkinson_probability": 0.08,
    "healthy_probability": 0.92,
    "risk_level": "low",
    "message": "Voice characteristics appear normal"
  },
  "features": {
    "jitter": 0.0043,
    "shimmer": 0.0689,
    "hnr": 17.05
  },
  "filename": "recording.wav"
}
```

### GET /api/info
Get API information

**Response:**
```json
{
  "version": "1.0.0",
  "model": "Parkinson's Voice Analysis",
  "features": [
    "Voice recording analysis",
    "Real-time prediction",
    "Audio quality assessment"
  ],
  "supported_formats": ["wav", "mp3", "ogg", "webm"]
}
```

## Testing

### Test with cURL

```bash
# Health check
curl http://localhost:5000/api/health

# API info
curl http://localhost:5000/api/info

# Upload audio for prediction
curl -X POST http://localhost:5000/api/predict \
  -F "audio=@tanvir.wav"
```

### Test with Browser

1. Go to http://localhost:5500
2. Click "Record Voice" or upload a file
3. See hardcoded results (always shows "No Parkinson's")

## Directory Structure

```
backend/
├── app.py              # Flask API server
├── requirements.txt    # Python dependencies
└── uploads/           # Temporary upload folder (auto-created)

frontend/
└── index.html         # Single-page web app
```

## Integration with ML Model

To integrate the real ML model (future):

### Option 1: Use Existing Predictor

Replace hardcoded response in `backend/app.py`:

```python
from src.parkinson_predictor import ParkinsonPredictor

# Initialize predictor (do this once at startup)
predictor = ParkinsonPredictor()

@app.route('/api/predict', methods=['POST'])
def predict():
    # ... file handling code ...

    # Real prediction
    result = predictor.predict(filepath, return_details=True)

    # Convert to API response format
    response = {
        'success': result['success'],
        'prediction': 'parkinsons' if result['prediction'] == 1 else 'healthy',
        'confidence': result['pd_probability'] if result['prediction'] == 1 else result['healthy_probability'],
        'details': {
            'parkinson_probability': result['pd_probability'],
            'healthy_probability': result['healthy_probability'],
            'risk_level': result['risk_level'].lower(),
            'message': result['recommendation']
        }
    }

    return jsonify(response)
```

### Option 2: Use Real-Data Model

```python
from predict_realdata import predict_with_realdata_model

@app.route('/api/predict', methods=['POST'])
def predict():
    # ... file handling code ...

    result = predict_with_realdata_model(filepath, verbose=False)

    # Format response...
```

## Troubleshooting

### CORS Errors

If you see CORS errors in browser console:
- Make sure Flask CORS is installed: `pip3 install flask-cors`
- Backend must be running on port 5000
- Frontend must be on different port (5500) or different domain

### Port Already in Use

If port 5000 or 5500 is already in use:

**Backend:**
```python
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Frontend:**
```javascript
// Change API_URL in index.html
const API_URL = 'http://localhost:5001/api';
```

### Microphone Not Working

- Grant microphone permissions in browser
- Use HTTPS or localhost (mic requires secure context)
- Check browser console for error messages

## Production Deployment

For production deployment:

1. **Backend:**
   - Use production WSGI server (Gunicorn, uWSGI)
   - Enable HTTPS
   - Set proper CORS origins (not `*`)
   - Add authentication
   - Use environment variables for config

2. **Frontend:**
   - Build/minify assets
   - Use CDN for hosting
   - Enable caching
   - Add analytics

3. **Infrastructure:**
   - Deploy on cloud (AWS, GCP, Azure, Heroku)
   - Use managed database for history
   - Add load balancer
   - Set up monitoring

## Current Status

- [x] Flask API backend
- [x] HTML/JS frontend
- [x] File upload
- [x] Voice recording
- [x] Hardcoded response (demo)
- [ ] ML model integration
- [ ] Feature extraction
- [ ] Database for history
- [ ] User authentication
- [ ] Production deployment

## Next Steps

1. Test the current hardcoded demo
2. Integrate real ML model (UCI or Real-Data)
3. Add audio preprocessing
4. Build quality detection
5. Add user accounts and history
6. Deploy to production
