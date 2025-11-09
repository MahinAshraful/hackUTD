import React, { useState, useRef } from 'react';

const API_URL = 'http://localhost:5001/api';

function App() {
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      uploadFile(selectedFile);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.addEventListener('dataavailable', (e) => {
        audioChunksRef.current.push(e.data);
      });

      mediaRecorderRef.current.addEventListener('stop', () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        uploadFile(audioFile);
        stream.getTracks().forEach(track => track.stop());
      });

      mediaRecorderRef.current.start();
      setRecording(true);

      setTimeout(() => {
        stopRecording();
      }, 5000);

    } catch (err) {
      setError('Error accessing microphone: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const uploadFile = async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Error connecting to server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div style={{ maxWidth: '600px', margin: '50px auto', padding: '20px' }}>
      <h1>Parkinson's Voice Detection</h1>
      <p>AI-powered voice analysis for early detection</p>

      {error && (
        <div style={{ padding: '10px', background: '#ffcccc', border: '1px solid red', marginBottom: '20px' }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{ padding: '10px', background: '#ffffcc', border: '1px solid orange', marginBottom: '20px' }}>
          Processing audio...
        </div>
      )}

      {!result && (
        <div>
          <div style={{ marginBottom: '20px' }}>
            <h3>Upload Audio File</h3>
            <input
              type="file"
              accept=".wav,.mp3,.ogg,.webm"
              onChange={handleFileChange}
              style={{ display: 'block', marginBottom: '10px' }}
            />
            <small>Supported: WAV, MP3, OGG, WEBM</small>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h3>Or Record Voice</h3>
            <button
              onClick={recording ? stopRecording : startRecording}
              style={{
                padding: '10px 20px',
                fontSize: '16px',
                background: recording ? '#ff4444' : '#4444ff',
                color: 'white',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              {recording ? 'Recording... (5s)' : 'Record Voice (5 seconds)'}
            </button>
          </div>
        </div>
      )}

      {result && (
        <div>
          <div style={{ padding: '20px', background: '#f0f0f0', marginBottom: '20px' }}>
            <h2>Results</h2>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: result.prediction === 'healthy' ? 'green' : 'red' }}>
              {result.prediction === 'healthy' ? 'No Parkinson\'s Detected' : 'Parkinson\'s Indicators Detected'}
            </div>
            <p>Confidence: {Math.round(result.confidence * 100)}%</p>
          </div>

          <div style={{ padding: '20px', background: '#f0f0f0', marginBottom: '20px' }}>
            <h3>Voice Features</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>Jitter:</td>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{(result.features.jitter * 100).toFixed(2)}%</td>
                </tr>
                <tr>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>Shimmer:</td>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{(result.features.shimmer * 100).toFixed(2)}%</td>
                </tr>
                <tr>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>HNR:</td>
                  <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{result.features.hnr.toFixed(1)} dB</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{ padding: '15px', background: '#fff3cd', border: '1px solid #ffc107', marginBottom: '20px' }}>
            <strong>Disclaimer:</strong> This tool is for educational and screening purposes only.
            It is not a substitute for professional medical diagnosis. Please consult a healthcare provider.
          </div>

          <button
            onClick={resetApp}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              background: '#666',
              color: 'white',
              border: 'none',
              cursor: 'pointer',
              width: '100%'
            }}
          >
            Analyze Another Recording
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
