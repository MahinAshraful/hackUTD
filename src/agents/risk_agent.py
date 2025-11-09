#!/usr/bin/env python3
"""
Risk Assessment Agent - Calculates longitudinal risk trajectories
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class RiskAssessmentAgent(BaseAgent):
    """Calculates personalized risk trajectories using Nemotron"""

    def __init__(self, nemotron_client):
        super().__init__("RiskAssessment", nemotron_client)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk trajectory"""
        self.start()

        try:
            ml_result = context.get('ml_result', {})
            pd_prob = ml_result.get('pd_probability', 0)
            clinical_features = ml_result.get('clinical_features', {})
            research = context.get('research_findings', {})

            prompt = f"""Calculate a personalized Parkinson's disease risk trajectory.

**Current Assessment:**
- PD Probability: {pd_prob:.1%}
- Jitter: {clinical_features.get('jitter', 0):.2f}%
- Shimmer: {clinical_features.get('shimmer', 0):.2f}%
- HNR: {clinical_features.get('hnr', 0):.1f} dB

**Research Context:**
{research.get('analysis', 'Limited research data available')}

Provide:
1. 5-year risk trajectory (with and without intervention)
2. Key risk factors and protective factors
3. Progression probability over time
4. Confidence level in predictions

Be specific with percentages and timelines."""

            trajectory_analysis = self.nemotron.reason(prompt)

            # Extract structured risk data
            risk_profile = self._extract_risk_profile(pd_prob, clinical_features)

            result = self._create_result(
                success=True,
                current_risk=pd_prob,
                risk_profile=risk_profile,
                trajectory_analysis=trajectory_analysis,
                confidence='HIGH' if pd_prob < 0.3 or pd_prob > 0.7 else 'MODERATE'
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._create_result(success=False, error=str(e))

    def _extract_risk_profile(self, pd_prob: float, features: Dict) -> Dict:
        """Extract structured risk profile"""
        if pd_prob < 0.3:
            return {
                'level': 'LOW',
                'year_1': round(pd_prob * 100, 1),
                'year_2': max(5, round(pd_prob * 0.8 * 100, 1)),
                'year_5': max(3, round(pd_prob * 0.5 * 100, 1)),
                'progression_risk_2yr': '<5%',
                'progression_risk_5yr': '<8%'
            }
        elif pd_prob < 0.6:
            return {
                'level': 'MODERATE',
                'year_1': round(pd_prob * 100, 1),
                'year_2': round(min(75, pd_prob * 1.15 * 100), 1),
                'year_5': round(min(85, pd_prob * 1.3 * 100), 1),
                'progression_risk_2yr': '30-45%',
                'progression_risk_5yr': '50-65%'
            }
        else:
            return {
                'level': 'HIGH',
                'year_1': round(pd_prob * 100, 1),
                'year_2': round(min(90, pd_prob * 1.1 * 100), 1),
                'year_5': round(min(95, pd_prob * 1.2 * 100), 1),
                'progression_risk_2yr': '65-75%',
                'progression_risk_5yr': '80-90%'
            }
