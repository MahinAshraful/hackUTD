import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [audioFile, setAudioFile] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const timerRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setAudioFile(file)
      setResult(null)
      setError(null)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        setAudioFile(audioFile)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      setResult(null)
      setError(null)

      // Timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 4) {
            stopRecording()
            return 5
          }
          return prev + 1
        })
      }, 1000)

      // Auto-stop after 5 seconds
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          stopRecording()
        }
      }, 5000)
    } catch (err) {
      setError('Failed to access microphone. Please check permissions.')
      console.error('Error accessing microphone:', err)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  const analyzeAudio = async () => {
    if (!audioFile) {
      setError('Please upload or record an audio file first')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('audio', audioFile)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      if (data.success) {
        setResult(data)
      } else {
        setError(data.error || 'Analysis failed')
      }
    } catch (err) {
      setError(`Failed to analyze audio: ${err.message}`)
      console.error('Error analyzing audio:', err)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setAudioFile(null)
    setResult(null)
    setError(null)
    setRecordingTime(0)
  }

  const getRiskColor = (riskLevel) => {
    const colors = {
      'LOW': '#10b981',
      'MODERATE': '#f59e0b',
      'HIGH': '#f97316',
      'VERY HIGH': '#ef4444'
    }
    return colors[riskLevel] || '#6b7280'
  }

  return (
    <div className="app">
      <div className="container">
        <header>
          <h1>🎤 Parkinson's Disease Voice Detector</h1>
          <p className="subtitle">AI-powered early screening through voice analysis</p>
        </header>

        <div className="card">
          <h2>Step 1: Provide Audio Sample</h2>

          <div className="input-section">
            <div className="upload-area">
              <label htmlFor="file-upload" className="upload-label">
                📁 Upload Audio File
                <input
                  id="file-upload"
                  type="file"
                  accept="audio/*"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </label>
              {audioFile && (
                <p className="file-name">Selected: {audioFile.name}</p>
              )}
            </div>

            <div className="divider">OR</div>

            <div className="record-area">
              <button
                className={`record-button ${isRecording ? 'recording' : ''}`}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={loading}
              >
                {isRecording ? '⏹️ Stop Recording' : '🎙️ Record Voice (5s)'}
              </button>
              {isRecording && (
                <p className="recording-timer">Recording... {recordingTime}s / 5s</p>
              )}
            </div>
          </div>

          <button
            className="analyze-button"
            onClick={analyzeAudio}
            disabled={!audioFile || loading || isRecording}
          >
            {loading ? 'Analyzing...' : '🔬 Analyze Audio'}
          </button>
        </div>

        {error && (
          <div className="card error-card">
            <p className="error-message">❌ {error}</p>
          </div>
        )}

        {result && (
          <div className="card result-card">
            <h2>Analysis Results</h2>

            <div className="result-main">
              <div
                className="risk-badge"
                style={{ backgroundColor: getRiskColor(result.risk_level) }}
              >
                {result.risk_level} RISK
              </div>

              <div className="probability">
                <div className="prob-label">Parkinson's Probability</div>
                <div className="prob-value">
                  {(result.pd_probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="recommendation">
              <strong>Recommendation:</strong> {result.recommendation}
            </div>

            {result.feature_importance && (
              <div className="features">
                <h3>Top Contributing Features</h3>
                <div className="feature-list">
                  {Object.entries(result.feature_importance)
                    .slice(0, 5)
                    .map(([feature, value]) => (
                      <div key={feature} className="feature-item">
                        <span className="feature-name">{feature}</span>
                        <div className="feature-bar-container">
                          <div
                            className="feature-bar"
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                        <span className="feature-value">{value.toFixed(3)}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            <button className="reset-button" onClick={reset}>
              🔄 Analyze Another Recording
            </button>

            <div className="disclaimer">
              <strong>Medical Disclaimer:</strong> This tool is for screening purposes only
              and is not a diagnostic device. Please consult a healthcare professional for
              proper diagnosis and treatment.
            </div>
          </div>
        )}

        <footer>
          <p>Powered by Machine Learning • 91% Accuracy • 100% Recall</p>
        </footer>
      </div>
    </div>
  )
}

export default App
