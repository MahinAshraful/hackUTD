#!/usr/bin/env python3
"""
Flask API for Parkinson's Detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Parkinson\'s Detection API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict Parkinson's disease from audio file
    Accepts: multipart/form-data with 'audio' file
    Returns: JSON with prediction results
    """

    # Check if file is in request
    if 'audio' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    file = request.files['audio']

    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed: WAV, MP3, OGG, WEBM'
        }), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # TODO: Replace this with actual model prediction
    # For now, hardcoded to return "No Parkinson's"

    result = {
        'success': True,
        'prediction': 'healthy',
        'confidence': 0.92,
        'details': {
            'parkinson_probability': 0.08,
            'healthy_probability': 0.92,
            'risk_level': 'low',
            'message': 'Voice characteristics appear normal'
        },
        'features': {
            'jitter': 0.0043,
            'shimmer': 0.0689,
            'hnr': 17.05
        },
        'filename': filename
    }

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify(result)

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'version': '1.0.0',
        'model': 'Parkinson\'s Voice Analysis',
        'features': [
            'Voice recording analysis',
            'Real-time prediction',
            'Audio quality assessment'
        ],
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    print("="*80)
    print("PARKINSON'S DETECTION API")
    print("="*80)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /api/health   - Health check")
    print("  POST /api/predict  - Upload audio for prediction")
    print("  GET  /api/info     - API information")
    print("\n" + "="*80)

    app.run(debug=True, host='0.0.0.0', port=5001)
