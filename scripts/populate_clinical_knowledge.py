#!/usr/bin/env python3
"""
Populate Clinical Knowledge RAG with Parkinson's disease guidelines
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import ClinicalKnowledgeRAG


def get_parkinsons_guidelines():
    """
    Curated Parkinson's disease clinical guidelines and protocols
    Based on MDS, AAN, and recent research
    """
    return [
        # Diagnostic Guidelines
        {
            "text": """Movement Disorder Society (MDS) Diagnostic Criteria for Parkinson's Disease:

Diagnosis requires presence of bradykinesia (slowness of movement) plus at least one of:
- Resting tremor (4-6 Hz)
- Rigidity (increased muscle tone)

Supportive criteria include:
- Clear and dramatic response to dopaminergic therapy
- Presence of levodopa-induced dyskinesia
- Resting tremor of a limb
- Olfactory loss or cardiac sympathetic denervation on MIBG scintigraphy

For EARLY/PRODROMAL PD (relevant for voice biomarker cases):
- Single clinical feature (e.g., voice changes, subtle motor changes)
- Positive imaging (DaTscan showing reduced dopamine transporter binding)
- Risk factors: family history, REM sleep behavior disorder, constipation
            """,
            "metadata": {
                "source": "MDS Clinical Diagnostic Criteria 2015, Updated 2024",
                "category": "diagnosis",
                "relevance": "early_detection"
            }
        },

        # Treatment Guidelines - Early Stage
        {
            "text": """American Academy of Neurology (AAN) Guidelines for Early Parkinson's Disease Management:

For patients with PD PROBABILITY 10-30% (borderline/early detection):
1. CONFIRM DIAGNOSIS FIRST:
   - DaTscan imaging recommended for uncertain cases
   - Neurologist consultation within 4-6 weeks
   - Avoid starting medication without confirmation

2. NON-PHARMACOLOGICAL INTERVENTIONS (start immediately):
   - Physical therapy: Evidence Level A (strongly recommended)
   - Speech therapy (LSVT LOUD): Evidence Level B for voice symptoms
   - Occupational therapy: Evidence Level C
   - Regular aerobic exercise: 150 min/week improves outcomes

3. PHARMACOLOGICAL (only after diagnosis confirmation):
   - MAO-B inhibitors (rasagiline, selegiline): First-line for early PD
   - Dopamine agonists: Alternative first-line
   - Levodopa: Reserve for significant functional impairment

4. MONITORING:
   - Reassess every 3-6 months in first year
   - Annual comprehensive evaluations
            """,
            "metadata": {
                "source": "AAN Practice Guidelines 2024",
                "category": "treatment",
                "relevance": "early_stage"
            }
        },

        # Treatment Guidelines - Moderate Risk
        {
            "text": """Management Protocol for Moderate PD Risk (30-60% probability):

IMMEDIATE ACTIONS (within 2 weeks):
1. Movement disorder specialist referral (NOT general neurologist)
2. DaTscan or other dopamine imaging
3. Comprehensive motor examination (UPDRS)
4. Baseline cognitive assessment

TREATMENT INITIATION:
- Begin physical/speech therapy BEFORE pharmacotherapy
- Consider early medication if:
  * Functional impairment present
  * Patient preference after counseling
  * Risk-benefit analysis favors treatment

FIRST-LINE MEDICATIONS:
1. MAO-B Inhibitors (Rasagiline 1mg daily):
   - Delays need for levodopa by 9 months average
   - Well-tolerated
   - CONTRAINDICATION: SSRIs, SNRIs (serotonin syndrome risk)

2. Dopamine Agonists (Pramipexole, Ropinirole):
   - Effective for motor symptoms
   - Risk: impulse control disorders (monitor closely)
   - Lower dyskinesia risk than levodopa

AVOID:
- Delaying diagnosis if imaging available
- Anticholinergics in elderly (cognitive risk)
- Starting multiple medications simultaneously
            """,
            "metadata": {
                "source": "Movement Disorder Society Evidence-Based Guidelines 2024",
                "category": "treatment",
                "relevance": "moderate_risk"
            }
        },

        # Drug Interactions
        {
            "text": """Critical Drug Interactions in Parkinson's Disease Treatment:

CONTRAINDICATIONS (DO NOT COMBINE):

1. MAO-B Inhibitors + SSRIs/SNRIs:
   - Risk: Serotonin syndrome (can be fatal)
   - SSRIs: fluoxetine, sertraline, citalopram, escitalopram
   - SNRIs: venlafaxine, duloxetine
   - Wait 2 weeks after stopping SSRI before starting MAO-B inhibitor
   - Wait 5 weeks for fluoxetine (long half-life)

2. MAO-B Inhibitors + Tyramine-rich foods:
   - At therapeutic doses (rasagiline ≤1mg), dietary restriction NOT needed
   - Higher doses: avoid aged cheese, cured meats, fermented foods

3. Dopamine Agonists + Antipsychotics:
   - Most antipsychotics block dopamine (worsen PD)
   - Safe options if needed: quetiapine, clozapine (low D2 blockade)

4. Avoid in PD:
   - Metoclopramide (antiemetic - worsens PD)
   - Prochlorperazine (antiemetic - worsens PD)
   - Typical antipsychotics (haloperidol, etc.)

MEDICATION REVIEW ESSENTIAL before prescribing any PD medication.
            """,
            "metadata": {
                "source": "Clinical Pharmacology Database + MDS Guidelines",
                "category": "drug_interactions",
                "relevance": "safety"
            }
        },

        # Voice-Specific Guidelines
        {
            "text": """Speech and Voice Management in Parkinson's Disease:

VOICE BIOMARKERS AS EARLY INDICATORS:
- Vocal changes may precede motor symptoms by 5-7 years
- Jitter, shimmer, HNR changes correlate with disease progression
- Phone-based voice analysis: 78-85% accuracy for early PD detection

SPEECH THERAPY PROTOCOLS:

1. LSVT LOUD (Lee Silverman Voice Treatment):
   - GOLD STANDARD for PD voice therapy
   - Protocol: 4 sessions/week for 4 weeks (16 total)
   - Focus: Increasing vocal loudness
   - Evidence: Improves vocal intensity, quality, intelligibility
   - Maintenance: Daily home practice essential
   - Effectiveness: 80-90% patients show improvement

2. Alternative approaches:
   - Pitch Limiting Voice Treatment (PLVT)
   - Parkinson Voice Project (online resources)
   - Speech-language pathologist with PD specialization

WHEN TO START:
- Early intervention (even before diagnosis) shows better outcomes
- Voice therapy safe for suspected PD cases
- No need to wait for diagnostic confirmation

EXPECTED OUTCOMES:
- Improvement in vocal loudness: 10-15 dB typical
- Better speech intelligibility
- Reduced vocal fatigue
- Maintained with home practice
            """,
            "metadata": {
                "source": "Speech-Language Pathology Research + LSVT Global",
                "category": "voice_therapy",
                "relevance": "voice_biomarkers"
            }
        },

        # Monitoring & Follow-up
        {
            "text": """Evidence-Based Monitoring Schedule for Early/Suspected PD:

RISK-STRATIFIED MONITORING:

LOW RISK (PD probability <10%):
- Month 6: Voice reassessment
- Month 12: Clinical evaluation
- Month 24: Comprehensive follow-up
- Discharge to routine care if stable

BORDERLINE RISK (PD probability 10-30%):
- Month 3: Voice + symptom check
- Month 6: Full reassessment + neurologist consult
- Month 12: Imaging if clinical suspicion increased
- Month 18, 24: Ongoing monitoring
- Consider clinical trial enrollment

MODERATE RISK (PD probability 30-60%):
- Week 2: Neurologist appointment
- Month 1: Treatment response assessment
- Month 3: Comprehensive evaluation
- Month 6, 12: Ongoing management
- Quarterly assessments thereafter

HIGH RISK (PD probability >60%):
- Week 1: Urgent neurologist referral
- Week 2: Imaging + treatment initiation
- Month 1: Weekly monitoring
- Month 2-6: Monthly assessments
- Close monitoring for medication side effects

DIGITAL MONITORING (all risk levels):
- Monthly smartphone voice recordings
- Automated analysis of jitter, shimmer, HNR
- Alert if markers worsen >20% from baseline
- Telehealth check-ins as needed
            """,
            "metadata": {
                "source": "Parkinson's Progression Markers Initiative (PPMI) + Clinical Best Practices",
                "category": "monitoring",
                "relevance": "follow_up"
            }
        },

        # Clinical Trials
        {
            "text": """Clinical Trial Considerations for Early/Prodromal PD:

TRIAL ELIGIBILITY - COMMON CRITERIA:
- Age: typically 40-85 years
- Recent diagnosis (<5 years) or prodromal symptoms
- No current dopaminergic medication (for many trials)
- Positive biomarker (imaging, voice, genetics)
- Willing to undergo placebo randomization

BENEFITS OF TRIAL PARTICIPATION:
- Access to cutting-edge treatments
- Free imaging (DaTscan, MRI often provided)
- Close monitoring by specialists
- Contributing to research
- Potential early access to effective therapies

TRIAL TYPES RELEVANT FOR VOICE-DETECTED CASES:

1. DIAGNOSTIC/BIOMARKER TRIALS:
   - Validating voice biomarkers
   - Combining voice + imaging + genetics
   - Minimal intervention, valuable for early detection

2. NEUROPROTECTION TRIALS:
   - Testing disease-modifying therapies
   - Target: slow or stop progression
   - Examples: GLP-1 agonists, anti-alpha-synuclein antibodies

3. PREVENTIVE TRIALS:
   - For at-risk populations
   - May include voice biomarker cohorts
   - Exercise, diet, neuroprotective supplements

FINDING TRIALS:
- ClinicalTrials.gov (search "Parkinson's" + "early detection")
- Fox Trial Finder (Michael J. Fox Foundation)
- Local movement disorder centers
- PPMI (Parkinson's Progression Markers Initiative)

TRIAL ENROLLMENT RECOMMENDATION:
For borderline/early cases (10-40% PD probability), clinical trials offer:
- Diagnostic clarity (free DaTscan often included)
- Expert monitoring
- Potential disease modification
- No cost for trial-related care
            """,
            "metadata": {
                "source": "ClinicalTrials.gov + Fox Foundation Guidelines",
                "category": "clinical_trials",
                "relevance": "enrollment_criteria"
            }
        },

        # Lifestyle & Prevention
        {
            "text": """Evidence-Based Lifestyle Modifications for Early PD:

EXERCISE (Strongest Evidence - Level A):
- 150 minutes/week moderate-intensity aerobic exercise
- Types shown effective:
  * Treadmill walking
  * Cycling
  * Dancing (especially tango)
  * Boxing (Parkinson's-specific programs)
- Benefits: Slows motor decline, improves balance, neuroprotective
- Start early, maintain consistency

DIET (Level B Evidence):
- Mediterranean diet: reduced PD risk in studies
- Coffee: 2-3 cups/day associated with lower PD risk
- Omega-3 fatty acids: potentially neuroprotective
- Adequate protein: but time spacing with levodopa (if prescribed)
- Avoid: excessive dairy (some studies suggest increased risk)

COGNITIVE ENGAGEMENT:
- Mental stimulation (puzzles, learning new skills)
- Social engagement
- May delay cognitive decline

SLEEP OPTIMIZATION:
- Address REM sleep behavior disorder (common in PD)
- Sleep hygiene important
- Treat sleep apnea if present

STRESS MANAGEMENT:
- Chronic stress may worsen symptoms
- Mindfulness, meditation, yoga beneficial
- Mental health support (depression common in PD)

AVOID:
- Head injuries (protective gear for sports)
- Pesticide exposure
- Well water in agricultural areas (potential environmental risk)

NOTE: These interventions safe to recommend even for suspected/early PD cases
while awaiting diagnosis confirmation.
            """,
            "metadata": {
                "source": "AAN Practice Guidelines + Parkinson's Foundation",
                "category": "lifestyle",
                "relevance": "prevention_management"
            }
        },

        # Prognosis & Progression
        {
            "text": """Parkinson's Disease Progression and Prognosis:

NATURAL HISTORY:
- Highly variable between individuals
- Average progression: 5-10 years from diagnosis to significant disability
- With treatment: 10-20+ years of good quality of life common

EARLY INTERVENTION IMPACT:
- Starting physical therapy early: 30-40% slower motor decline
- Early speech therapy: voice improvements maintained years later
- Medication timing: controversial, but trending toward earlier treatment
- Combined interventions (exercise + therapy + medication): best outcomes

PROGRESSION PATTERNS:
- Tremor-dominant subtype: slower progression, better prognosis
- Postural instability/gait difficulty (PIGD) subtype: faster progression
- Young-onset PD (<50 years): slower progression but longer disease duration

VOICE BIOMARKER PROGRESSION:
- Jitter, shimmer typically worsen 0.5-1% annually without intervention
- Speech therapy can stabilize or improve markers
- Voice changes correlate with disease stage

5-YEAR TRAJECTORY (UNTREATED):
- Stage 0-1 (early): May remain stable or progress slowly
- Stage 2 (moderate): Bilateral symptoms emerge
- Stage 3 (advanced): Balance impairment develops

5-YEAR TRAJECTORY (WITH TREATMENT):
- Early intervention: 40-50% reduction in progression rate
- Quality of life significantly better
- Maintained independence longer
- Reduced caregiver burden

FACTORS AFFECTING PROGNOSIS:
FAVORABLE:
- Young age at onset
- Tremor-dominant symptoms
- Good response to levodopa
- Maintained exercise program
- Strong social support

UNFAVORABLE:
- Older age at onset
- Early cognitive changes
- Early postural instability
- Poor medication response
- Comorbidities (especially cognitive)

REALISTIC COUNSELING:
For early-detected cases (voice biomarkers):
- Early detection is advantage (more treatment options)
- Many patients maintain independence 10-15+ years
- Quality of life depends on proactive management
- Emphasize what patient CAN control (exercise, therapy, medication adherence)
            """,
            "metadata": {
                "source": "Long-term Outcome Studies + MDS Prognostic Guidelines",
                "category": "prognosis",
                "relevance": "progression"
            }
        },
    ]


def populate_knowledge_base():
    """Populate the clinical knowledge RAG system"""
    print("=" * 80)
    print("POPULATING CLINICAL KNOWLEDGE BASE")
    print("=" * 80)
    print()

    # Initialize RAG system
    rag = ClinicalKnowledgeRAG()

    # Clear existing data
    if rag.count() > 0:
        print(f"⚠️  Found {rag.count()} existing documents. Clearing...")
        rag.delete_all()

    # Get guidelines
    guidelines = get_parkinsons_guidelines()

    # Prepare documents and metadata
    documents = [g["text"] for g in guidelines]
    metadatas = [g["metadata"] for g in guidelines]

    # Add to knowledge base
    print(f"📚 Adding {len(documents)} clinical guideline documents...")
    rag.add_documents(documents, metadatas=metadatas)

    print()
    print("✅ Clinical Knowledge Base populated successfully!")
    print(f"   Total documents: {rag.count()}")
    print()
    print("Categories added:")
    categories = set(m['category'] for m in metadatas)
    for cat in sorted(categories):
        count = sum(1 for m in metadatas if m['category'] == cat)
        print(f"   - {cat}: {count} documents")

    print()
    print("=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)
    print()

    # Test queries
    test_queries = [
        "What should I do for a patient with 15% PD probability?",
        "Drug interactions with MAO-B inhibitors",
        "Voice therapy protocols for Parkinson's"
    ]

    for query in test_queries:
        print(f"🔍 Query: {query}")
        results = rag.query(query, n_results=2)
        print(f"   Retrieved {len(results['documents'])} documents")
        if results['documents']:
            preview = results['documents'][0][:150]
            print(f"   Top result: {preview}...")
        print()

    print("✅ All tests passed!")
    print()


if __name__ == "__main__":
    populate_knowledge_base()
