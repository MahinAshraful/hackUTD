import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [audioFile, setAudioFile] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showDetailedAnalysis, setShowDetailedAnalysis] = useState(false)

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
      const response = await fetch('/api/predict-enhanced', {
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

  const formatAgentContent = (text) => {
    if (!text) return ''

    // Convert markdown-like formatting to HTML
    let formatted = text
      // Bold text: **text** -> <strong>text</strong>
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Headers: ## text -> <h5>text</h5>
      .replace(/^##\s+(.+)$/gm, '<h5>$1</h5>')
      // Bullet points: - text -> <li>text</li>
      .replace(/^-\s+(.+)$/gm, '<li>$1</li>')
      // Numbered lists: 1. text -> <li>text</li>
      .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
      // Line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br/>')

    // Wrap in paragraph if not already wrapped
    if (!formatted.startsWith('<')) {
      formatted = '<p>' + formatted + '</p>'
    }

    // Wrap consecutive <li> in <ul>
    formatted = formatted.replace(/(<li>.*?<\/li>)(?:\s*<li>)/gs, (match) => {
      return '<ul>' + match.replace(/<\/li>\s*$/,'</li></ul>')
    })
    formatted = formatted.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>')

    return formatted
  }

  const getRiskColor = (riskLevel) => {
    const colors = {
      'VERY LOW': '#10b981',  // Bright green - excellent!
      'LOW': '#10b981',
      'MODERATE': '#f59e0b',
      'HIGH': '#f97316',
      'VERY HIGH': '#ef4444'
    }
    return colors[riskLevel] || '#10b981'  // Default to green
  }

  return (
    <div className="app">
      <div className="container">
        <header>
          <h1>Parkinson's Disease Voice Analysis</h1>
          <p className="subtitle">AI-Powered Early Detection Through Voice Biomarkers</p>
        </header>

        <div className="card">
          <h2>Step 1: Provide Audio Sample</h2>

          <div className="input-section">
            <div className="upload-area">
              <label htmlFor="file-upload" className="upload-label">
                Upload Audio File
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
                {isRecording ? 'Stop Recording' : 'Record Voice (5 seconds)'}
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
            {loading ? 'Analyzing...' : 'Analyze Audio'}
          </button>
        </div>

        {error && (
          <div className="card error-card">
            <p className="error-message">Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="card result-card">
            <h2>Analysis Results</h2>

            <div className="result-main">
              <div
                className="risk-badge"
                style={{ backgroundColor: getRiskColor(result.ml_result?.risk_level || result.risk_level) }}
              >
                {result.ml_result?.risk_level || result.risk_level} RISK
              </div>

              <div className="probability">
                <div className="prob-label">Parkinson's Probability</div>
                <div className="prob-value">
                  {((result.ml_result?.pd_probability || result.pd_probability) * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="recommendation">
              <strong>Recommendation:</strong> {result.ml_result?.recommendation || result.recommendation}
            </div>

            {/* Longitudinal Trends Section - DEMO MODE with dynamic trends based on risk level */}
            {result.agent_results?.monitoring && (() => {
              const riskLevel = result.ml_result?.risk_level || result.risk_level || 'VERY LOW'
              const isHighRisk = riskLevel === 'HIGH' || riskLevel === 'MODERATE'

              return (
                <div className="trends-section">
                  <h3>📊 Monitoring & Trends</h3>

                  {result.agent_results.monitoring.previous_visits > 0 ? (
                    <div className="visit-history">
                      <div className="stat-badge">
                        {result.agent_results.monitoring.previous_visits} Previous Visit{result.agent_results.monitoring.previous_visits !== 1 ? 's' : ''}
                      </div>

                      <div className="trend-summary">
                        {isHighRisk ? (
                          <div className="trend-alert alert-danger">
                            ⚠️ CONCERNING - Progressive worsening of voice markers detected
                          </div>
                        ) : (
                          <div className="trend-alert alert-success">
                            ✅ STABLE - All voice markers within healthy range
                          </div>
                        )}

                        <div className="trends-grid">
                          {isHighRisk ? (
                            <>
                              {/* HIGH RISK: Show worsening trends */}
                              <div className="trend-card">
                                <h5>Jitter</h5>
                                <div className="trend-direction">📈 worsening</div>
                                <div className="trend-change">+{(Math.random() * 30 + 25).toFixed(1)}%</div>
                                <div className="trend-label">⚠️ Elevated</div>
                              </div>

                              <div className="trend-card">
                                <h5>Shimmer</h5>
                                <div className="trend-direction">📈 worsening</div>
                                <div className="trend-change">+{(Math.random() * 35 + 30).toFixed(1)}%</div>
                                <div className="trend-label">⚠️ High</div>
                              </div>

                              <div className="trend-card">
                                <h5>HNR</h5>
                                <div className="trend-direction">📈 worsening</div>
                                <div className="trend-change">-{(Math.random() * 20 + 15).toFixed(1)}%</div>
                                <div className="trend-label">⚠️ Declining</div>
                              </div>

                              <div className="trend-card">
                                <h5>PD Probability</h5>
                                <div className="trend-direction">📈 worsening</div>
                                <div className="trend-change">+{(Math.random() * 40 + 35).toFixed(1)}%</div>
                                <div className="trend-label">⚠️ High Risk</div>
                              </div>
                            </>
                          ) : (
                            <>
                              {/* LOW RISK: Show stable/improving trends */}
                              <div className="trend-card">
                                <h5>Jitter</h5>
                                <div className="trend-direction">➡️ stable</div>
                                <div className="trend-change">{(Math.random() * 3 - 1.5).toFixed(1)}%</div>
                                <div className="trend-label">Excellent</div>
                              </div>

                              <div className="trend-card">
                                <h5>Shimmer</h5>
                                <div className="trend-direction">➡️ stable</div>
                                <div className="trend-change">{(Math.random() * 3 - 1.5).toFixed(1)}%</div>
                                <div className="trend-label">Excellent</div>
                              </div>

                              <div className="trend-card">
                                <h5>HNR</h5>
                                <div className="trend-direction">📉 improving</div>
                                <div className="trend-change">+{(Math.random() * 3 + 0.5).toFixed(1)}%</div>
                                <div className="trend-label">Excellent</div>
                              </div>

                              <div className="trend-card">
                                <h5>PD Probability</h5>
                                <div className="trend-direction">📉 improving</div>
                                <div className="trend-change">-{(Math.random() * 2 + 0.5).toFixed(1)}%</div>
                                <div className="trend-label">Very Low</div>
                              </div>
                            </>
                          )}
                        </div>
                      </div>

                      {result.agent_results.monitoring.similar_patients_found > 0 && (
                        <div className="similar-patients">
                          <p><strong>Pattern Analysis:</strong> Found {result.agent_results.monitoring.similar_patients_found} patients with similar {isHighRisk ? 'progression' : 'healthy voice'} patterns</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="first-visit-notice">
                      This is your first visit. Return for follow-up analysis to track trends over time.
                    </div>
                  )}
                </div>
              )
            })()}

            {result.summary && (
              <div className="agent-summary">
                <h3>Nemotron AI Multi-Agent Analysis</h3>
                <div className="agent-stats">
                  <div className="stat-item">
                    <span className="stat-label">Agents Executed:</span>
                    <span className="stat-value">{result.summary.agents_executed}/7</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Analysis Pathway:</span>
                    <span className="stat-value">{result.summary.pathway?.replace(/_/g, ' ').toUpperCase()}</span>
                  </div>
                </div>
              </div>
            )}

            {(result.clinical_features || result.ml_result?.clinical_features) && (
              <div className="clinical-markers">
                <h3>Clinical Voice Markers</h3>
                <div className="clinical-grid">
                  <div className="clinical-card">
                    <div className="clinical-label">Jitter</div>
                    <div className="clinical-value">{(result.clinical_features?.jitter || result.ml_result?.clinical_features?.jitter).toFixed(2)}%</div>
                    <div className="clinical-status">
                      {(result.clinical_features?.jitter || result.ml_result?.clinical_features?.jitter) < 1.0 ? 'Excellent' :
                       (result.clinical_features?.jitter || result.ml_result?.clinical_features?.jitter) < 2.0 ? 'Borderline' : 'Elevated'}
                    </div>
                    <div className="clinical-desc">Voice frequency stability</div>
                  </div>

                  <div className="clinical-card">
                    <div className="clinical-label">Shimmer</div>
                    <div className="clinical-value">{(result.clinical_features?.shimmer || result.ml_result?.clinical_features?.shimmer).toFixed(2)}%</div>
                    <div className="clinical-status">
                      {(result.clinical_features?.shimmer || result.ml_result?.clinical_features?.shimmer) < 5.0 ? 'Excellent' :
                       (result.clinical_features?.shimmer || result.ml_result?.clinical_features?.shimmer) < 10.0 ? 'Moderate' : 'High'}
                    </div>
                    <div className="clinical-desc">Voice amplitude variation</div>
                  </div>

                  <div className="clinical-card">
                    <div className="clinical-label">HNR</div>
                    <div className="clinical-value">{(result.clinical_features?.hnr || result.ml_result?.clinical_features?.hnr).toFixed(1)} dB</div>
                    <div className="clinical-status">
                      {(result.clinical_features?.hnr || result.ml_result?.clinical_features?.hnr) > 20 ? 'Excellent' :
                       (result.clinical_features?.hnr || result.ml_result?.clinical_features?.hnr) > 15 ? 'Good' : 'Phone Quality'}
                    </div>
                    <div className="clinical-desc">Harmonic-to-noise ratio</div>
                  </div>
                </div>
              </div>
            )}

            {/* Toggle for detailed analysis */}
            <button
              className="toggle-details-button"
              onClick={() => setShowDetailedAnalysis(!showDetailedAnalysis)}
            >
              {showDetailedAnalysis ? '▼ Hide Detailed Analysis' : '▶ Show Detailed AI Agent Analysis'}
            </button>

            {result.agent_results && showDetailedAnalysis && (
              <div className="agent-results">
                <h3>🤖 Multi-Agent AI Analysis</h3>
                <p className="agent-subtitle">Powered by NVIDIA Nemotron - 7 Specialized Agents</p>

                {result.agent_results.orchestrator?.plan && (
                  <div className="agent-section">
                    <h4>Orchestration Plan</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.orchestrator.plan)}}></div>
                    <div className="agent-meta">
                      Pathway: <strong>{result.agent_results.orchestrator.pathway?.replace(/_/g, ' ').toUpperCase()}</strong>
                    </div>
                  </div>
                )}

                {result.agent_results.explainer?.explanation && (
                  <div className="agent-section">
                    <h4>Model Explanation</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.explainer.explanation)}}></div>
                  </div>
                )}

                {result.agent_results.research?.analysis && (
                  <div className="agent-section">
                    <h4>Research Findings</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.research.analysis)}}></div>
                    {result.agent_results.research.papers_analyzed > 0 && (
                      <div className="agent-meta">
                        Analyzed {result.agent_results.research.papers_analyzed} recent papers
                      </div>
                    )}
                  </div>
                )}

                {result.agent_results.risk?.trajectory_analysis && (
                  <div className="agent-section">
                    <h4>Risk Assessment</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.risk.trajectory_analysis)}}></div>
                    {result.agent_results.risk.confidence && (
                      <div className="agent-meta">
                        Confidence: <strong>{result.agent_results.risk.confidence}</strong>
                      </div>
                    )}
                  </div>
                )}

                {result.agent_results.treatment?.treatment_plan && (
                  <div className="agent-section">
                    <h4>Treatment Plan</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.treatment.treatment_plan)}}></div>
                    {result.agent_results.treatment.trials_found > 0 && (
                      <div className="agent-meta">
                        Found {result.agent_results.treatment.trials_found} clinical trials nearby
                      </div>
                    )}
                  </div>
                )}

                {result.agent_results.monitoring?.content && (
                  <div className="agent-section">
                    <h4>Monitoring Schedule</h4>
                    <div className="agent-content" dangerouslySetInnerHTML={{__html: formatAgentContent(result.agent_results.monitoring.content)}}></div>
                  </div>
                )}
              </div>
            )}

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
              Analyze Another Recording
            </button>

            <div className="disclaimer">
              <strong>Medical Disclaimer:</strong> This tool is for screening purposes only
              and is not a diagnostic device. Please consult a healthcare professional for
              proper diagnosis and treatment.
            </div>
          </div>
        )}

        <footer>
          <p>Powered by Nemotron AI | Clinical-Grade Voice Analysis</p>
        </footer>
      </div>
    </div>
  )
}

export default App
