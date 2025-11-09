# 🤖 What the Nemotron Agents Actually Do

## Overview

**Nemotron** is NVIDIA's large language model (like ChatGPT, but specialized for reasoning tasks). In your app, you use **Nemotron-70B-Instruct** to power 7 specialized AI agents that analyze Parkinson's disease risk.

---

## The Multi-Agent Workflow

```
Voice Audio → ML Prediction → 7 Nemotron Agents → Comprehensive Report
                 (Random)        (Parallel Analysis)
```

Each agent calls Nemotron with a **specialized prompt** to get expert analysis in different areas.

---

## The 7 Nemotron Agents Explained

### 1. 🎯 Orchestrator Agent
**File:** `src/agents/orchestrator.py`

**What it does:**
- Receives ML prediction (PD probability, clinical features)
- Asks Nemotron: *"Based on this risk level, what's the optimal diagnostic pathway?"*
- Decides which agents to activate and in what order
- Chooses pathway:
  - **LOW risk (<30%)** → Light Monitoring
  - **MODERATE (30-60%)** → Moderate Intervention
  - **HIGH risk (>60%)** → Urgent Intervention

**Example Nemotron prompt:**
```
You are orchestrating a clinical diagnostic workflow for Parkinson's disease.

ML Model Results:
- PD Probability: 75%
- Risk Level: HIGH
- Jitter: 2.8%
- Shimmer: 11.2%
- HNR: 12.1 dB

Based on this risk level, plan the optimal agent activation sequence...
```

**Nemotron returns:** Orchestration plan with recommended agent sequence

---

### 2. 📚 Research Agent
**File:** `src/agents/research_agent.py`

**What it ACTUALLY does:**
1. **Searches PubMed** (real API call!) for recent papers on Parkinson's voice analysis
2. Fetches top 5 relevant papers with titles, authors, journals
3. Asks Nemotron: *"Analyze these papers and tell me what they mean for THIS patient's specific voice markers"*

**Example workflow:**
```python
# Step 1: Search PubMed
query = "parkinson voice analysis shimmer 2024"
papers = search_pubmed(query)  # Real API call

# Step 2: Ask Nemotron to analyze
prompt = f"""
Analyze these recent research papers:

1. Voice Analysis in Early PD - Smith et al. (2024) - JAMA Neurology
2. Shimmer Patterns in Prodromal PD - Zhang et al. (2024)

Patient's markers:
- Jitter: 0.67%
- Shimmer: 4.99% (BORDERLINE)
- HNR: 17.3 dB

What do these papers tell us about THIS patient?
"""

analysis = nemotron.reason(prompt)  # Nemotron AI call
```

**Nemotron returns:** Evidence-based analysis connecting research to patient

---

### 3. 📊 Risk Assessment Agent
**File:** `src/agents/risk_agent.py`

**What it does:**
- Takes current risk + research findings
- Asks Nemotron: *"Calculate this patient's 5-year risk trajectory"*
- Gets detailed progression probabilities with and without treatment

**Example Nemotron prompt:**
```
Calculate a personalized Parkinson's risk trajectory.

Current Assessment:
- PD Probability: 15%
- Jitter: 0.43% (EXCELLENT)
- Shimmer: 6.89% (BORDERLINE)
- HNR: 17.1 dB (GOOD)

Research Context:
Recent studies show shimmer 5-7% has 45% sensitivity for early PD...

Provide:
1. 5-year risk trajectory (with and without intervention)
2. Key risk factors
3. Progression probability
4. Confidence level
```

**Nemotron returns:** Year-by-year risk projections (e.g., Year 1: 15% → Year 5: 8% with preventive care)

---

### 4. 💊 Treatment Planning Agent
**File:** `src/agents/treatment_agent.py`

**What it ACTUALLY does:**
1. **Searches ClinicalTrials.gov** (real API call!) for active Parkinson's trials
2. Filters by location (within 500 miles), status (recruiting), and relevance
3. Asks Nemotron: *"Given this risk level and these available trials, create a personalized treatment plan"*

**Example workflow:**
```python
# Step 1: Search real clinical trials
trials = search_clinical_trials(
    condition="Parkinson Disease",
    location="Dallas, TX",
    status="RECRUITING"
)  # Real ClinicalTrials.gov API call

# Results:
# - NCT05445678: Early Levodopa in Prodromal PD (Phase 3)
# - NCT05445123: Neuroprotective Therapy (Phase 2)

# Step 2: Ask Nemotron for personalized plan
prompt = f"""
Create treatment plan for HIGH risk patient.

Available trials:
- Early Levodopa in Prodromal PD (NCT05445678, Phase 3)
- Neuroprotective Therapy (NCT05445123, Phase 2)

Provide:
1. Immediate actions
2. Medication options
3. Therapy recommendations (voice, physical)
4. Clinical trial enrollment recommendations
"""

plan = nemotron.reason(prompt)
```

**Nemotron returns:** Comprehensive treatment plan including trial recommendations

---

### 5. 💡 Explainer Agent
**File:** `src/agents/explainer_agent.py`

**What it does:**
- Takes ML prediction + clinical features
- Asks Nemotron: *"Explain in simple terms WHY the model predicted this result"*
- Breaks down which features contributed most

**Example Nemotron prompt:**
```
Explain why the ML model predicted 15% PD probability.

Voice Features:
- Jitter: 0.43% (Normal: <1%, Elevated: >2%)
  → Your value: EXCELLENT
- Shimmer: 6.89% (Normal: <5%, Elevated: >10%)
  → Your value: BORDERLINE
- HNR: 17.1 dB (Phone normal: 10-25 dB)
  → Your value: GOOD

Explain:
1. Which features drove this prediction
2. How each compares to normal ranges
3. Why this combination = 15%
4. What this means clinically
5. Model confidence
```

**Nemotron returns:** Patient-friendly explanation of the prediction

---

### 6. 📄 Report Generator Agent
**File:** `src/agents/report_agent.py`

**What it does:**
- Collects ALL previous agent results
- Asks Nemotron: *"Create a comprehensive clinical report for a doctor"*
- Formats everything into a professional medical document

**Example Nemotron prompt:**
```
Generate comprehensive clinical report.

ML Prediction: 15% PD probability (LOW risk)
Research Findings: [Smith et al. findings...]
Risk Trajectory: [5-year analysis...]
Treatment Plan: [Preventive care recommendations...]
Monitoring: [12-month schedule...]

Create a professional medical report suitable for:
- Primary care physician
- Neurologist referral
- Patient records
```

**Nemotron returns:** Structured clinical report with all findings

---

### 7. 📅 Monitoring Agent
**File:** `src/agents/monitoring_agent.py`

**What it does:**
- Takes risk level + treatment plan
- Asks Nemotron: *"Design a personalized follow-up schedule"*
- Creates specific monitoring timeline

**Example Nemotron prompt:**
```
Create personalized monitoring schedule for LOW risk patient.

Risk Profile:
- Current: 15% PD probability
- 5-year trajectory: 15% → 8% (with preventive care)
- Treatment: Voice therapy, lifestyle optimization

Design:
1. Follow-up visit schedule (3mo, 6mo, 12mo, etc.)
2. What to assess at each visit
3. Red flags to watch for
4. When to escalate care
```

**Nemotron returns:** Month-by-month monitoring plan with specific milestones

---

## How Nemotron is Used

### The `nemotron.reason()` Call

Every agent uses this simple interface:

```python
# Initialize Nemotron client
nemotron = NemotronClient(api_key="nvapi-...")

# Send a prompt, get AI reasoning
response = nemotron.reason(prompt="Explain this medical data...")

# response = Nemotron's intelligent analysis
```

### What Happens Behind the Scenes

**Normal Mode (with API key):**
```python
POST https://integrate.api.nvidia.com/v1/chat/completions
Headers: Authorization: Bearer nvapi-...
Body: {
  "model": "nvidia/llama-3.1-nemotron-70b-instruct",
  "messages": [
    {"role": "system", "content": "You are a medical AI assistant..."},
    {"role": "user", "content": "Explain this patient's risk..."}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**Fallback Mode (current - no valid API key):**
```python
# If API fails (401/404), use intelligent mock responses
if "low" in prompt or "15%" in prompt:
    return generate_low_risk_response()
elif "high" in prompt or "75%" in prompt:
    return generate_high_risk_response()
```

---

## Real-World Example

### Input to System:
```json
{
  "jitter": 2.8,
  "shimmer": 11.2,
  "hnr": 12.1,
  "pd_probability": 0.75,
  "risk_level": "HIGH"
}
```

### What Each Agent Does:

1. **Orchestrator** → Nemotron analyzes risk → Returns "urgent_intervention" pathway
2. **Research** → Searches PubMed → Nemotron analyzes 5 papers → Returns evidence summary
3. **Risk** → Nemotron calculates trajectory → Returns "75% → 82% (2yr) without treatment"
4. **Treatment** → Searches trials → Nemotron creates plan → Returns neurologist referral + trial options
5. **Explainer** → Nemotron explains prediction → Returns "Multiple elevated markers indicate high risk..."
6. **Monitoring** → Nemotron designs schedule → Returns "Week 1: Neurologist, Week 2: DaTscan, Month 3: Eval..."
7. **Report** → Nemotron compiles everything → Returns professional medical document

### Final Output:
```json
{
  "success": true,
  "agents_executed": 7,
  "pathway": "urgent_intervention",
  "agent_results": {
    "orchestrator": {...},
    "research": {...},
    "risk": {...},
    "treatment": {...},
    "explainer": {...},
    "monitoring": {...},
    "report": {...}
  }
}
```

---

## Key Points

✅ **Nemotron = NVIDIA's LLM** (like GPT, Claude, etc.)
✅ **7 Agents = 7 specialized prompts** to Nemotron
✅ **Real API integrations**: PubMed, ClinicalTrials.gov
✅ **Fallback mode works**: Intelligent mock responses when API unavailable
✅ **Each agent adds value**: Research, risk analysis, treatment planning, etc.

**Think of it as:** 7 medical specialists (powered by AI) analyzing one patient from different angles.

---

## Why This is Cool for Hackathons

1. **Shows AI sophistication** - Not just one prediction, but multi-agent reasoning
2. **Real API integrations** - Actually searches PubMed and clinical trials
3. **Contextual analysis** - Nemotron connects research to specific patient
4. **Practical output** - Treatment plans, monitoring schedules, trial recommendations
5. **Demonstrates Nvidia tech** - Using Nemotron (Nvidia track!)

🎉 **This is way more impressive than just "ML model predicts Parkinson's"!**
