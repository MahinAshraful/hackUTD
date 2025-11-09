#!/usr/bin/env python3
"""
Enhanced Monitoring Agent with Patient History RAG
- Tracks longitudinal trends using patient history database
- Detects progression patterns using FAISS similarity search
- Implements ReAct pattern for adaptive monitoring
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import sys
from pathlib import Path
import numpy as np

# Add path for RAG import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.patient_history_rag import PatientHistoryRAG


class MonitoringAgentRAG(BaseAgent):
    """Enhanced Monitoring Agent with RAG and trend detection"""

    def __init__(self, nemotron_client):
        super().__init__("Monitoring", nemotron_client)

        # Initialize Patient History RAG
        try:
            self.patient_history = PatientHistoryRAG()
            self.rag_enabled = True
            print(f"   ✅ Patient History RAG loaded ({self.patient_history.count_visits()} visits)")
        except Exception as e:
            print(f"   ⚠️  Patient History RAG initialization error: {e}")
            self.patient_history = None
            self.rag_enabled = False

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring with trend detection and similarity analysis"""
        self.start()

        try:
            ml_result = context.get('ml_result', {})
            risk_level = ml_result.get('risk_level', 'UNKNOWN')
            pd_prob = ml_result.get('pd_probability', 0)

            # Get patient ID (use filename or generate)
            patient_id = context.get('patient_id', 'demo_patient_001')

            # Check if this is a returning patient
            patient_history = []
            trend_analysis = None
            similar_patients = []
            trajectory_prediction = None

            if self.rag_enabled:
                # Get patient history
                patient_history = self.patient_history.get_patient_history(patient_id, limit=10)

                print(f"\n📊 Patient History Analysis")
                print(f"   Patient ID: {patient_id}")
                print(f"   Previous visits: {len(patient_history)}")

                # Detect trends if returning patient
                if len(patient_history) >= 1:
                    print(f"   🔍 Detecting trends...")
                    trend_analysis = self.patient_history.detect_trends(patient_id)

                    # Find similar patients for trajectory prediction
                    if 'voice_features' in ml_result and ml_result['voice_features'] is not None:
                        print(f"   🔍 Finding similar patient patterns...")
                        similar_patients = self.patient_history.find_similar_patients(
                            ml_result['voice_features'],
                            k=5
                        )

                        # Generate trajectory prediction using similar patients
                        trajectory_prediction = self._predict_trajectory(
                            trend_analysis,
                            similar_patients,
                            risk_level,
                            pd_prob
                        )

            # Generate monitoring schedule using Nemotron
            monitoring_schedule = self._generate_schedule(
                risk_level,
                pd_prob,
                trend_analysis,
                similar_patients,
                trajectory_prediction
            )

            result = self._create_result(
                success=True,
                risk_level=risk_level,
                pd_probability=pd_prob,
                patient_id=patient_id,
                previous_visits=len(patient_history),
                trend_analysis=trend_analysis,
                similar_patients_found=len(similar_patients),
                trajectory_prediction=trajectory_prediction,
                monitoring_schedule=monitoring_schedule,
                rag_used=self.rag_enabled
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._fallback_monitoring(context)

    def _predict_trajectory(
        self,
        trend_analysis: Optional[Dict],
        similar_patients: List[Dict],
        risk_level: str,
        pd_prob: float
    ) -> str:
        """
        Predict trajectory using trend analysis and similar patients

        Uses Nemotron to reason about progression based on:
        - Patient's own trend data
        - Similar patients' outcomes
        """
        if not similar_patients:
            return "Insufficient historical data for trajectory prediction."

        # Build context from similar patients
        similar_context = self._format_similar_patients(similar_patients)

        # Build trend context
        trend_context = ""
        if trend_analysis and 'trends' in trend_analysis:
            trend_context = "**Patient's Recent Trends:**\n"
            for feature, data in trend_analysis['trends'].items():
                direction = data['direction']
                change = data['percent_change']
                trend_context += f"- {feature}: {direction.upper()} ({change:+.1f}%)\n"

        prompt = f"""Analyze this patient's progression trajectory based on similar cases:

**Current Status:**
- Risk Level: {risk_level}
- PD Probability: {pd_prob:.1%}

{trend_context}

**Similar Patient Patterns:**
{similar_context}

Based on these similar patients' outcomes, predict:
1. Likely progression over next 6-12 months
2. Key risk factors to monitor
3. Recommended intervention timing
4. Confidence level in prediction

Be specific and evidence-based."""

        return self.nemotron.reason(prompt)

    def _format_similar_patients(self, similar_patients: List[Dict]) -> str:
        """Format similar patients for prompt"""
        if not similar_patients:
            return "No similar patients found."

        context = []
        for i, patient in enumerate(similar_patients[:3], 1):  # Top 3
            sim_score = patient.get('similarity_score', 0)
            pd_prob = patient.get('pd_probability', 0)
            risk_level = patient.get('risk_level', 'UNKNOWN')

            context.append(
                f"{i}. Similarity: {sim_score:.2f} | "
                f"PD Probability: {pd_prob:.1%} | "
                f"Risk: {risk_level}"
            )

        return "\n".join(context)

    def _generate_schedule(
        self,
        risk_level: str,
        pd_prob: float,
        trend_analysis: Optional[Dict],
        similar_patients: List[Dict],
        trajectory_prediction: Optional[str]
    ) -> str:
        """Generate personalized monitoring schedule using Nemotron"""

        # Determine if trends are concerning
        concerning_trends = False
        if trend_analysis and 'overall_assessment' in trend_analysis:
            assessment = trend_analysis['overall_assessment']
            concerning_trends = 'CONCERNING' in assessment or 'WATCH' in assessment

        # Build context for Nemotron
        context_parts = [f"**Risk Level:** {risk_level} ({pd_prob:.1%} PD probability)"]

        if trend_analysis:
            context_parts.append(f"\n**Trend Analysis:**\n{trend_analysis.get('overall_assessment', 'N/A')}")

        if similar_patients:
            context_parts.append(f"\n**Similar Patients Found:** {len(similar_patients)} cases")

        if trajectory_prediction:
            context_parts.append(f"\n**Trajectory Prediction:**\n{trajectory_prediction}")

        if concerning_trends:
            context_parts.append("\n⚠️ **ALERT:** Concerning progression trends detected")

        context = "\n".join(context_parts)

        # Adjust monitoring intensity based on trends
        if concerning_trends:
            prompt = f"""CREATE INTENSIVE MONITORING SCHEDULE (trends worsening):

{context}

Design a personalized follow-up schedule with:
1. **Immediate actions** (next 2 weeks)
2. **Short-term monitoring** (1-3 months) - FREQUENT due to trends
3. **Medium-term plan** (3-12 months)
4. **Alert triggers** - when to escalate care

Be specific with timelines and what to monitor."""

        else:
            prompt = f"""CREATE PERSONALIZED MONITORING SCHEDULE:

{context}

Design a follow-up schedule with:
1. **Initial follow-up** timing
2. **Regular monitoring** intervals
3. **Long-term surveillance** plan
4. **Patient self-monitoring** recommendations
5. **Alert triggers** - when to seek urgent care

Adjust intensity based on risk level."""

        return self.nemotron.reason(prompt)

    def _fallback_monitoring(self, context: Dict) -> Dict[str, Any]:
        """Fallback monitoring schedule"""
        ml_result = context.get('ml_result', {})
        risk_level = ml_result.get('risk_level', 'UNKNOWN')
        pd_prob = ml_result.get('pd_probability', 0)

        schedule = f"Standard monitoring schedule for {risk_level} risk. Follow up as recommended by healthcare provider."

        return self._create_result(
            success=True,
            risk_level=risk_level,
            pd_probability=pd_prob,
            patient_id='unknown',
            previous_visits=0,
            monitoring_schedule=schedule,
            rag_used=False,
            note="Using fallback mode"
        )
