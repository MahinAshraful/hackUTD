#!/usr/bin/env python3
"""
Report Generator Agent - Creates comprehensive clinical reports
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from datetime import datetime


class ReportGeneratorAgent(BaseAgent):
    """Generates professional medical reports"""

    def __init__(self, nemotron_client):
        super().__init__("ReportGenerator", nemotron_client)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report"""
        self.start()

        try:
            # Gather all findings
            ml_result = context.get('ml_result', {})
            research = context.get('research_findings', {})
            risk = context.get('risk_assessment', {})
            treatment = context.get('treatment_plan', {})
            explanation = context.get('explanation', {})

            # Generate doctor report
            doctor_report = self._generate_doctor_report(
                ml_result, research, risk, treatment, explanation
            )

            # Generate patient report
            patient_report = self._generate_patient_report(
                ml_result, risk, treatment
            )

            result = self._create_result(
                success=True,
                doctor_report=doctor_report,
                patient_report=patient_report,
                generated_at=datetime.now().isoformat()
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._create_result(success=False, error=str(e))

    def _generate_doctor_report(self, ml, research, risk, treatment, explanation) -> str:
        """Generate technical report for doctors"""
        prompt = f"""Generate a professional clinical report for a neurologist.

**Voice Analysis Results:**
- PD Probability: {ml.get('pd_probability', 0):.1%}
- Risk Level: {ml.get('risk_level', 'UNKNOWN')}
- Clinical Features: Jitter {ml.get('clinical_features', {}).get('jitter', 0):.2f}%, 
  Shimmer {ml.get('clinical_features', {}).get('shimmer', 0):.2f}%, 
  HNR {ml.get('clinical_features', {}).get('hnr', 0):.1f} dB

**Research Findings:**
{research.get('analysis', 'No research analysis available')[:500]}

**Risk Assessment:**
{risk.get('trajectory_analysis', 'No risk analysis available')[:500]}

**Treatment Recommendations:**
{treatment.get('treatment_plan', 'No treatment plan available')[:500]}

Create a formal medical report with:
1. Executive Summary
2. Clinical Findings
3. Evidence Base
4. Risk Stratification
5. Recommended Actions
6. Follow-up Plan

Use professional medical terminology."""

        return self.nemotron.reason(prompt)

    def _generate_patient_report(self, ml, risk, treatment) -> str:
        """Generate patient-friendly report"""
        prompt = f"""Generate a clear, compassionate report for the patient.

**Results:**
- Risk Level: {ml.get('risk_level', 'UNKNOWN')}
- PD Probability: {ml.get('pd_probability', 0):.1%}

**Risk Information:**
{risk.get('trajectory_analysis', 'No risk analysis available')[:300]}

**Next Steps:**
{treatment.get('treatment_plan', 'No treatment plan available')[:300]}

Create a patient-friendly report with:
1. What the results mean
2. What happens next
3. Answers to common questions
4. When to follow up

Use clear, non-technical language. Be reassuring but honest."""

        return self.nemotron.reason(prompt)
