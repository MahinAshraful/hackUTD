#!/usr/bin/env python3
"""
Explainer Agent - Explains why ML model made its prediction
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent


class ExplainerAgent(BaseAgent):
    """Explains ML model decisions using Nemotron"""

    def __init__(self, nemotron_client):
        super().__init__("Explainer", nemotron_client)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Explain ML prediction"""
        self.start()

        try:
            ml_result = context.get("ml_result", {})
            pd_prob = ml_result.get("pd_probability", 0)
            clinical_features = ml_result.get("clinical_features", {})

            prompt = f"""Explain why the ML model predicted this Parkinson's disease probability.

**ML Prediction:** {pd_prob:.1%} PD probability

**Voice Features:**
- Jitter: {clinical_features.get('jitter', 0):.2f}% (Normal: <1%, Elevated: >2%)
- Shimmer: {clinical_features.get('shimmer', 0):.2f}% (Normal: <5%, Elevated: >10%)
- HNR: {clinical_features.get('hnr', 0):.1f} dB (Phone: 10-25 dB normal)

Explain:
1. Which features drove this prediction and why
2. How each feature compares to normal ranges
3. Why this combination resulted in {pd_prob:.1%}
4. What this means clinically
5. Model confidence and limitations

Be clear and educational for patients."""

            explanation = self.nemotron.reason(prompt)

            result = self._create_result(
                success=True,
                prediction=pd_prob,
                explanation=explanation,
                key_factors=self._extract_key_factors(clinical_features, pd_prob),
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._create_result(success=False, error=str(e))

    def _extract_key_factors(self, features: Dict, pd_prob: float) -> List[Dict]:
        """Extract key contributing factors"""
        factors = []

        jitter = features.get("jitter", 0)
        shimmer = features.get("shimmer", 0)
        hnr = features.get("hnr", 0)

        # Jitter analysis
        if jitter < 1:
            factors.append(
                {
                    "feature": "Jitter",
                    "value": f"{jitter:.2f}%",
                    "status": "EXCELLENT",
                    "impact": "Protective factor - reduces risk",
                }
            )
        elif jitter > 2:
            factors.append(
                {
                    "feature": "Jitter",
                    "value": f"{jitter:.2f}%",
                    "status": "ELEVATED",
                    "impact": "Risk factor - increases prediction",
                }
            )

        # Shimmer analysis
        if shimmer < 5:
            factors.append(
                {
                    "feature": "Shimmer",
                    "value": f"{shimmer:.2f}%",
                    "status": "NORMAL",
                    "impact": "Within healthy range",
                }
            )
        elif shimmer < 7:
            factors.append(
                {
                    "feature": "Shimmer",
                    "value": f"{shimmer:.2f}%",
                    "status": "BORDERLINE",
                    "impact": "Warrants monitoring",
                }
            )
        else:
            factors.append(
                {
                    "feature": "Shimmer",
                    "value": f"{shimmer:.2f}%",
                    "status": "ELEVATED",
                    "impact": "Significant risk factor",
                }
            )

        # HNR analysis
        if hnr < 10:
            factors.append(
                {
                    "feature": "HNR",
                    "value": f"{hnr:.1f} dB",
                    "status": "LOW",
                    "impact": "Reduced voice clarity",
                }
            )
        elif hnr > 20:
            factors.append(
                {
                    "feature": "HNR",
                    "value": f"{hnr:.1f} dB",
                    "status": "EXCELLENT",
                    "impact": "Good voice quality",
                }
            )

        return factors
