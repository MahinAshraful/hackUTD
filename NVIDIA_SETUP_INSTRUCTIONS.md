# 🎯 Nvidia Track Setup Instructions

## What I Built for You

I've created a **Multi-Agent Nemotron Intelligence System** with 7 specialized AI agents that work together to analyze Parkinson's disease. This goes far beyond a simple chatbot!

### 🏗️ Architecture

```
Voice Recording
    ↓
[Your ML Model] - Real predictions (could be 15%, 65%, or 85%)
    ↓
[Nemotron Multi-Agent System] - 7 agents enhance the result:
  1. Orchestrator   - Plans workflow based on risk level
  2. Research       - Searches PubMed + Nemotron analysis
  3. Risk Assessment - Calculates 5-year trajectory
  4. Treatment      - Finds clinical trials + plans care
  5. Explainer      - Why did ML predict this?
  6. Report Generator - Professional medical docs
  7. Monitoring     - Personalized follow-up schedule
    ↓
Complete Clinical Intelligence Report
```

---

## 📋 What YOU Need to Do

### Step 1: Get Nvidia API Key (FREE for Hackathon)

1. **Visit**: https://build.nvidia.com/
2. **Sign up** with your hackathon email
3. **Navigate to**: Nemotron models
4. **Get API Key** - Should be free for hackathon use
5. **Copy** your API key

### Step 2: Set Environment Variable

**On Mac/Linux:**
```bash
export NVIDIA_API_KEY="your-api-key-here"
```

**On Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="your-api-key-here"
```

**Or add to your `.env` file in the project root:**
```bash
# Create .env file in /home/user/hackUTD/
echo "NVIDIA_API_KEY=your-api-key-here" > .env
```

### Step 3: Install Python Dependency

The multi-agent system needs `requests` library (might already be installed):

```bash
pip install requests
```

### Step 4: Test the Backend

```bash
cd backend
python app.py
```

**You should see:**
```
================================================================================
PARKINSON'S DETECTION API v2.0 - NEMOTRON AI POWERED
================================================================================

Starting Flask server...
🚀 Loading Parkinson's Detection Model...
✅ Model loaded successfully!
🤖 Loading Nemotron Multi-Agent System...
✅ Multi-Agent System ready!

API will be available at: http://localhost:5001

Endpoints:
  GET  /api/health            - Health check
  POST /api/predict           - Basic ML prediction
  POST /api/predict-enhanced  - Multi-agent Nemotron analysis ⭐
  GET  /api/info              - API information

Multi-Agent System:
  ✅ ACTIVE - 7 Nemotron agents ready
     • Orchestrator  • Research  • Risk Assessment
     • Treatment     • Explainer • Report Generator
     • Monitoring

================================================================================
```

**If API key is missing**, you'll see fallback mode (still works but uses intelligent mock responses).

---

## 🌐 Frontend Integration

### Option A: Use Enhanced Endpoint in Frontend (Recommended)

**Update your frontend to call `/api/predict-enhanced` instead of `/api/predict`:**

In `/home/user/hackUTD/frontend/src/App.jsx`, change line 96:

**FROM:**
```javascript
const response = await fetch('/api/predict', {
```

**TO:**
```javascript
const response = await fetch('/api/predict-enhanced', {
```

**This gives you:**
- All 7 agent results
- Research papers from PubMed
- Clinical trials near Dallas
- Risk trajectories
- Treatment plans
- Professional reports

### Option B: Show Agent Progress (Most Impressive for Judges)

I can create an enhanced frontend that shows agents working in real-time. **Do you want me to do this?**

It would show:
```
[✓] Voice Analysis Complete
[⏳] Orchestrator Planning Pathway...
[⏳] Research Agent Searching PubMed... (45 papers found)
[⏳] Risk Agent Calculating Trajectory...
[⏳] Treatment Agent Finding Clinical Trials...
[✓] Complete - View Full Report
```

---

## 🧪 Testing the Multi-Agent System

### Test with cURL (Backend Only):

```bash
# Record a voice sample or use existing file
curl -X POST http://localhost:5001/api/predict-enhanced \
  -F "audio=@test_recordings/mahintest.wav"
```

**You'll get back:**
```json
{
  "success": true,
  "ml_result": { /* Your ML model results */ },
  "agent_results": {
    "orchestrator": {
      "success": true,
      "pathway": "light_monitoring",
      "plan": "ORCHESTRATION PLAN - LOW RISK PATHWAY..."
    },
    "research": {
      "success": true,
      "papers_found": 12,
      "papers": [ /* PubMed papers */ ],
      "analysis": "RESEARCH FINDINGS..."
    },
    "risk": {
      "success": true,
      "trajectory_analysis": "LONGITUDINAL RISK ASSESSMENT...",
      "risk_profile": {
        "level": "LOW",
        "year_1": 15.0,
        "year_5": 3.0
      }
    },
    "treatment": {
      "success": true,
      "trials": [ /* Clinical trials */ ],
      "treatment_plan": "TREATMENT & CARE PLAN..."
    },
    "explainer": {
      "success": true,
      "explanation": "ML MODEL DECISION EXPLANATION..."
    },
    "report": {
      "success": true,
      "doctor_report": "...",
      "patient_report": "..."
    },
    "monitoring": {
      "success": true,
      "schedule_description": "...",
      "structured_schedule": { /* Timeline */ }
    }
  },
  "summary": {
    "risk_level": "LOW",
    "agents_executed": 7,
    "agents_successful": 7,
    "key_recommendations": [...]
  }
}
```

---

## 🎯 For the Nvidia Judges - What to Highlight

### 1. Multi-Step Workflow (Not a Chatbot!)

**Show them:**
- Orchestrator **plans** different pathways based on risk
- Agents execute **in sequence** with dependencies
- Each agent **reasons** using Nemotron
- System **synthesizes** findings into actionable plan

### 2. Tool Integration (Real APIs!)

**Show them:**
- **PubMed API**: Search medical research in real-time
- **ClinicalTrials.gov API**: Find nearby trials
- **Nemotron**: Powers all 7 agents with reasoning

### 3. Agentic Intelligence

**Explain:**
- System makes **decisions** (which pathway to execute)
- Agents **collaborate** (research informs risk, risk informs treatment)
- Nemotron provides **clinical reasoning** (not just text generation)
- **Adapts** to risk level (LOW vs HIGH gets different workflows)

### 4. Real Clinical Value

**Demonstrate:**
- Upload a voice sample
- Show LOW risk → light monitoring pathway
- Show HIGH risk → urgent intervention pathway
- Display research papers, clinical trials, treatment plans

---

## 🚨 Troubleshooting

### Issue: "Multi-agent system not available"
**Solution:** Set NVIDIA_API_KEY environment variable

### Issue: "Module not found: src.agents"
**Solution:** Run from correct directory and check all files created:
```bash
ls -la src/agents/
# Should see: base_agent.py, orchestrator.py, research_agent.py, etc.
```

### Issue: Agent taking too long
**Solution:** Normal! Each agent can take 2-5 seconds. Total ~15-30 seconds for all 7.

### Issue: PubMed/ClinicalTrials API errors
**Solution:** Agents have intelligent fallbacks - will use cached data but still show Nemotron reasoning

---

## 📊 What Makes This Nvidia-Track Worthy

✅ **Nemotron is the BRAIN** - Not just for text, it's planning, reasoning, synthesizing
✅ **Multi-step workflow** - Orchestrator → Research → Risk → Treatment → Report
✅ **Tool integration** - PubMed, ClinicalTrials, Medical databases
✅ **Real value** - Actual clinical decision support, not a demo toy
✅ **Agentic behavior** - Makes decisions, adapts, collaborates
✅ **Beyond chatbot** - Complex reasoning over medical data

---

## 🎬 Demo Script for Judges

1. **"Let me show you our AI agent architecture..."**
   - Show the 7 agents diagram
   - Explain multi-step workflow

2. **"I'll record my voice and show you what happens..."**
   - Record voice sample
   - Upload to enhanced endpoint
   - Show backend console printing agent execution

3. **"Watch the agents work together..."**
   - Point out Orchestrator planning
   - Research agent finding papers on PubMed
   - Treatment agent finding clinical trials in Dallas

4. **"Here's the comprehensive report..."**
   - Show research findings from 2024 papers
   - Show 5-year risk trajectory
   - Show nearby clinical trials
   - Show personalized monitoring schedule

5. **"This is powered by Nemotron at every step..."**
   - Explain how each agent uses Nemotron for reasoning
   - Show it's not scripted - Nemotron analyzes context
   - Demonstrate different pathways for different risk levels

---

## ✨ Optional Enhancements I Can Add

Want me to add any of these?

1. **Real-time Agent Progress UI** - Show agents working in frontend
2. **PDF Report Generation** - Download comprehensive report
3. **Voice Trending** - Track changes over multiple recordings
4. **SMS/Email Alerts** - For high-risk cases
5. **Clinical Trial Matching** - More sophisticated location/criteria matching

**Just let me know what else you want!** 🚀
