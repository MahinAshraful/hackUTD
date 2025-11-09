# Demo Mode Guide - Risk Level Differentiation

## Overview

The system now differentiates between **recorded voice** and **uploaded audio files** to generate appropriate risk levels for demo purposes.

## Behavior

### 🎤 Record Voice (Live Recording)
- **Detection**: File named `recording.webm`
- **Risk Level**: VERY LOW (0.5% - 1.8% PD probability)
- **Clinical Markers**:
  - Jitter: 0.25% - 0.65% (Excellent)
  - Shimmer: 2.5% - 4.5% (Excellent)
  - HNR: 19.0 - 25.0 dB (Excellent)
- **Prediction**: Healthy (0)
- **Recommendation**: "Excellent voice characteristics. All markers well within healthy range."
- **Purpose**: Reassuring demo for live user recordings

### 📁 Upload Audio File
- **Detection**: Any filename except `recording.webm`
- **Risk Level**: MODERATE to HIGH (40% - 75% PD probability)
- **Clinical Markers**:
  - Jitter: 1.8% - 3.5% (Elevated)
  - Shimmer: 6.5% - 12.0% (Elevated)
  - HNR: 10.0 - 16.0 dB (Borderline to Low)
- **Prediction**: Positive for PD (1)
- **Recommendation**: "Voice analysis shows elevated markers for Parkinson's disease. Recommend comprehensive neurological evaluation and specialist consultation."
- **Purpose**: Demonstrates full multi-agent Nemotron analysis with high-risk pathway

## Why This Approach?

### User Experience
- **Live recordings** should be reassuring (user is testing their own voice)
- **Uploaded files** demonstrate the system's capabilities with high-risk scenarios

### Demo Impact
- **Upload file** → Triggers full Nemotron agent workflow:
  - HIGH/MODERATE risk pathway
  - Research Agent searches PubMed for relevant papers
  - Treatment Agent uses ReAct loop to find interventions
  - Risk Agent models 5-year progression
  - All 7 agents fully engaged

- **Record voice** → Shows healthy baseline:
  - LOW risk pathway (lighter workflow)
  - Still runs agents but with preventive care focus
  - Demonstrates monitoring capabilities

## Technical Implementation

### Backend Detection Logic

```python
# In backend/app.py
is_recorded = filename.lower() == 'recording.webm'

if is_recorded:
    demo_data = generate_random_clinical_data(high_risk=False)  # VERY LOW
else:
    demo_data = generate_random_clinical_data(high_risk=True)   # HIGH/MODERATE
```

### Frontend File Naming

```javascript
// In frontend/src/App.jsx
// Recorded audio (line 39)
const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })

// Uploaded files keep their original names
// e.g., "patient_sample.wav", "test_audio.mp3"
```

## Risk Level Thresholds

| Risk Level | PD Probability | Jitter | Shimmer | HNR | Prediction |
|------------|----------------|--------|---------|-----|------------|
| VERY LOW   | 0.5% - 1.8%    | <1%    | <5%     | >19 | Healthy (0) |
| MODERATE   | 40% - 60%      | >1.8%  | >6.5%   | <16 | PD (1) |
| HIGH       | 60% - 75%      | >1.8%  | >6.5%   | <16 | PD (1) |

## Agent Workflow Pathways

### LOW Risk (Recorded Voice)
```
Orchestrator → Light Monitoring Protocol
├── Explainer: Interprets low-risk markers
├── Research: General preventive studies
├── Risk: Minimal progression likelihood
├── Treatment: Preventive recommendations
├── Monitoring: Annual follow-up schedule
└── Report: Reassuring clinical summary
```

### MODERATE/HIGH Risk (Uploaded File)
```
Orchestrator → Urgent Intervention Protocol
├── Explainer: Detailed marker analysis
├── Research: PubMed search for recent PD studies
├── Risk: 5-year progression modeling
├── Treatment: ReAct loop (3 iterations)
│   ├── Iteration 1: Query guidelines for MODERATE/HIGH risk
│   ├── Iteration 2: Check drug interactions
│   └── Iteration 3: Find clinical trials + synthesize plan
├── Monitoring: Quarterly follow-up with DaTscan
└── Report: Comprehensive diagnostic report
```

## Testing

### Test Recorded Voice (VERY LOW)
```bash
# Start backend
cd backend && python3 app.py

# Frontend: Click "Start Recording" → Record 5 seconds → "Analyze Voice"
# Expected: ~1% PD probability, VERY LOW risk, green indicators
```

### Test Uploaded File (HIGH/MODERATE)
```bash
# Frontend: Click "Choose File" → Upload any .wav/.mp3 file → "Analyze Voice"
# Expected: 40-75% PD probability, MODERATE/HIGH risk, full agent analysis
```

## Console Output Examples

### Recorded Voice
```
🎤 Recorded voice detected: recording.webm → Generating VERY LOW risk...
✅ Generated VERY LOW risk profile (PD: 1.2%)
```

### Uploaded File
```
📁 Uploaded file detected: patient_sample.wav → Generating HIGH risk...
✅ Generated HIGH risk profile (PD: 68.5%)
🚀 Running 3 agents in parallel: explainer, research, risk
🔄 Starting ReAct Loop (max 3 iterations)...
```

## Notes

- **Random variation**: Each analysis generates slightly different values within the range
- **Realistic demo**: High-risk uploads show off the full Nemotron intelligence system
- **Safe testing**: Live recordings always show healthy results
- **NVIDIA integration**: High-risk pathway maximizes usage of Nemotron Super 49B model
