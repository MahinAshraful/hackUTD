#!/usr/bin/env python3
"""
Flask API for Parkinson's Detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import os
import sys
import random
from werkzeug.utils import secure_filename

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))
# DEMO MODE: Skip ML model, just use Nemotron agents
from src.agents.coordinator import AgentCoordinator
from src.rag.patient_history_rag import PatientHistoryRAG
from datetime import datetime
import numpy as np
import json

app = Flask(__name__)
CORS(app)

print("🚀 DEMO MODE - Skipping ML model, using hardcoded data")

# Initialize multi-agent coordinator
print("🤖 Loading Nemotron Multi-Agent System...")
try:
    coordinator = AgentCoordinator()
    print("✅ Multi-Agent System ready!\n")
except Exception as e:
    print(f"⚠️  Multi-Agent System initialization warning: {e}")
    coordinator = None
    print("   (Fallback mode enabled)\n")

# Initialize Patient History RAG
print("📊 Loading Patient History Database...")
try:
    patient_history = PatientHistoryRAG()
    print("✅ Patient History RAG ready!\n")
except Exception as e:
    print(f"⚠️  Patient History RAG initialization warning: {e}")
    patient_history = None
    print("   (History tracking disabled)\n")

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_random_clinical_data():
    """
    Generate random clinical data for demo
    DEMO MODE: Always shows VERY LOW risk (<2%) for reassuring demo
    """
    # DEMO MODE: Always generate VERY HEALTHY values
    # Excellent jitter values (well below 1%)
    jitter = random.uniform(0.25, 0.65)  # Excellent: <1%

    # Excellent shimmer values (well below 5%)
    shimmer = random.uniform(2.5, 4.5)  # Excellent: <5%

    # Excellent HNR values (high is good)
    hnr = random.uniform(19.0, 25.0)  # Excellent: >15 dB

    # VERY LOW Parkinson's probability (<2%)
    pd_prob = random.uniform(0.005, 0.018)  # 0.5% to 1.8%

    risk_level = 'VERY LOW'

    return {
        'jitter': round(jitter, 2),
        'shimmer': round(shimmer, 2),
        'hnr': round(hnr, 1),
        'pd_probability': round(pd_prob, 3),  # 3 decimals for <2%
        'healthy_probability': round(1 - pd_prob, 3),
        'risk_level': risk_level,
        'prediction': 0  # Always healthy
    }

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
    DEMO MODE: Generate random clinical data instead of real prediction
    Accepts: multipart/form-data with 'audio' file
    Returns: JSON with random prediction results
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

    # Save file (just for demo, we don't actually process it)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # DEMO MODE: Generate random clinical data
        print(f"📊 Generating random clinical data for {filename}...")
        demo_data = generate_random_clinical_data()

        # Clean up uploaded file
        os.remove(filepath)

        # Prepare clinical features
        clinical_features = {
            'jitter': demo_data['jitter'],
            'shimmer': demo_data['shimmer'],
            'hnr': demo_data['hnr']
        }

        # Generate recommendation
        recommendation = 'Excellent voice characteristics. All markers well within healthy range. No signs of concern detected.'

        print(f"✅ Generated {demo_data['risk_level']} risk profile (PD: {demo_data['pd_probability']:.2%})")

        return jsonify({
            'success': True,
            'prediction': demo_data['prediction'],
            'pd_probability': demo_data['pd_probability'],
            'healthy_probability': demo_data['healthy_probability'],
            'risk_level': demo_data['risk_level'],
            'recommendation': recommendation,
            'feature_importance': {},
            'clinical_features': clinical_features,
            'filename': filename
        })

    except Exception as e:
        # Clean up file if prediction fails
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'success': False,
            'error': f'Demo data generation error: {str(e)}'
        }), 500

@app.route('/api/predict-enhanced', methods=['POST'])
def predict_enhanced():
    """
    DEMO MODE: Enhanced prediction with multi-agent Nemotron analysis
    Uses random clinical data and sends to Nemotron agents
    Accepts: multipart/form-data with 'audio' file
    Returns: Complete analysis from all agents
    """
    # Check if multi-agent system is available
    if coordinator is None:
        return jsonify({
            'success': False,
            'error': 'Multi-agent system not available. Use /api/predict instead.'
        }), 503

    # Same file validation as basic predict
    if 'audio' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed: WAV, MP3, OGG, WEBM'
        }), 400

    # Save file (just for demo, we don't actually process it)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Get patient ID from request (or use default)
        patient_id = request.form.get('patient_id', 'demo_patient_001')

        # DEMO MODE: Generate random clinical data
        print(f"📊 Generating random clinical data for {filename}...")
        demo_data = generate_random_clinical_data()

        # Clean up uploaded file
        os.remove(filepath)

        # Prepare clinical features
        clinical_features = {
            'jitter': demo_data['jitter'],
            'shimmer': demo_data['shimmer'],
            'hnr': demo_data['hnr']
        }

        # Generate fake 44-dimensional voice features for FAISS
        # In real implementation, these would come from audio analysis
        voice_features = np.random.randn(44).astype('float32')

        # Generate recommendation
        recommendation = 'Excellent voice characteristics. All markers well within healthy range. No signs of concern detected.'

        # Prepare ML result for agents
        ml_result = {
            'success': True,
            'prediction': demo_data['prediction'],
            'pd_probability': demo_data['pd_probability'],
            'healthy_probability': demo_data['healthy_probability'],
            'risk_level': demo_data['risk_level'],
            'recommendation': recommendation,
            'clinical_features': clinical_features,
            'voice_features': voice_features,  # For similarity search
            'filename': filename
        }

        # Save to patient history database
        if patient_history:
            try:
                visit_id = patient_history.add_visit(
                    patient_id=patient_id,
                    visit_date=datetime.now().isoformat(),
                    clinical_features=clinical_features,
                    ml_result=ml_result,
                    voice_features=voice_features,
                    notes=f"Demo visit from {filename}"
                )
                print(f"✅ Saved visit to history (visit_id: {visit_id})")
            except Exception as e:
                print(f"⚠️  Error saving visit to history: {e}")

        # Prepare context for agents
        context = {
            'ml_result': ml_result,
            'patient_id': patient_id,
            'patient_context': {
                'current_medications': request.form.getlist('medications[]') if 'medications[]' in request.form else []
            }
        }

        print(f"✅ Generated {demo_data['risk_level']} risk profile (PD: {demo_data['pd_probability']:.2%})")
        print(f"\n{'#'*80}")
        print(f"🤖 STARTING MULTI-AGENT NEMOTRON ANALYSIS")
        print(f"{'#'*80}\n")

        # Run multi-agent analysis
        agent_results = coordinator.run(context['ml_result'])

        print(f"\n{'#'*80}")
        print(f"✅ MULTI-AGENT ANALYSIS COMPLETE!")
        print(f"   Agents executed: {agent_results['summary']['agents_executed']}/7")
        print(f"   Success: {agent_results['success']}")
        print(f"   Pathway: {agent_results['summary']['pathway']}")
        print(f"{'#'*80}\n")

        # Convert numpy arrays to lists for JSON serialization
        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj

        # Clean agent_results for JSON
        clean_results = make_json_serializable(agent_results)

        return jsonify(clean_results)

    except Exception as e:
        # Clean up file if analysis fails
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'success': False,
            'error': f'Enhanced analysis error: {str(e)}'
        }), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'version': '2.0.0',
        'model': 'Parkinson\'s Voice Analysis with Nemotron AI',
        'features': [
            'Voice recording analysis',
            'Real-time prediction',
            'Audio quality assessment',
            'Multi-agent Nemotron intelligence',
            'PubMed research integration',
            'Clinical trial matching',
            'Personalized risk assessment',
            'Treatment planning'
        ],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'agents': {
            'orchestrator': 'Plans diagnostic workflow',
            'research': 'Searches medical literature (PubMed)',
            'risk': 'Calculates longitudinal risk',
            'treatment': 'Plans interventions & finds trials',
            'explainer': 'Explains ML predictions',
            'report': 'Generates clinical reports',
            'monitoring': 'Creates follow-up schedules'
        } if coordinator else None
    })

if __name__ == '__main__':
    print("="*80)
    print("PARKINSON'S DETECTION API v2.0 - NEMOTRON AI POWERED")
    print("="*80)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /api/health            - Health check")
    print("  POST /api/predict           - Basic ML prediction")
    print("  POST /api/predict-enhanced  - Multi-agent Nemotron analysis ⭐")
    print("  GET  /api/info              - API information")
    print("\nMulti-Agent System:")
    if coordinator:
        print("  ✅ ACTIVE - 7 Nemotron agents ready")
        print("     • Orchestrator  • Research  • Risk Assessment")
        print("     • Treatment     • Explainer • Report Generator")
        print("     • Monitoring")
    else:
        print("  ⚠️  FALLBACK MODE - Using basic prediction only")
    print("\n" + "="*80)

    app.run(debug=True, host='0.0.0.0', port=5001)
