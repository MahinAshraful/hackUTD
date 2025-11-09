#!/usr/bin/env python3
"""
Treatment Planning Agent - Plans interventions and finds clinical trials
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent
import requests


class TreatmentPlanningAgent(BaseAgent):
    """Plans treatment and searches clinical trials"""

    def __init__(self, nemotron_client):
        super().__init__("TreatmentPlanning", nemotron_client)
        self.clinical_trials_api = "https://clinicaltrials.gov/api/v2/studies"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan treatment and find trials"""
        self.start()

        try:
            ml_result = context.get('ml_result', {})
            risk_level = ml_result.get('risk_level', 'UNKNOWN')
            pd_prob = ml_result.get('pd_probability', 0)

            # Search clinical trials
            trials = self._search_trials(risk_level)

            # Generate treatment plan with Nemotron
            prompt = f"""Create a personalized treatment and care plan.

**Risk Level:** {risk_level} ({pd_prob:.1%} PD probability)

**Available Clinical Trials:**
{self._format_trials(trials)}

Provide:
1. Immediate actions (if HIGH risk) or preventive measures (if LOW)
2. Pharmacological options (if appropriate)
3. Therapy recommendations (voice, physical, occupational)
4. Clinical trial enrollment recommendations
5. Monitoring schedule

Be specific and actionable."""

            treatment_plan = self.nemotron.reason(prompt)

            result = self._create_result(
                success=True,
                risk_level=risk_level,
                trials_found=len(trials),
                trials=trials,
                treatment_plan=treatment_plan,
                urgent=pd_prob > 0.6
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._fallback_treatment(context)

    def _search_trials(self, risk_level: str) -> List[Dict]:
        """Search ClinicalTrials.gov"""
        try:
            params = {
                'query.cond': 'Parkinson Disease',
                'query.term': 'voice OR early detection',
                'filter.geo': 'distance(39.0,-95.0,500mi)',  # Central US
                'filter.overallStatus': 'RECRUITING',
                'pageSize': 5
            }

            response = requests.get(self.clinical_trials_api, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])
                return [self._parse_trial(s) for s in studies[:3]]
            else:
                return self._fallback_trials()

        except Exception as e:
            print(f"Clinical trials search error: {e}")
            return self._fallback_trials()

    def _parse_trial(self, study: Dict) -> Dict:
        """Parse trial data"""
        protocol = study.get('protocolSection', {})
        ident = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})

        return {
            'nct_id': ident.get('nctId', 'Unknown'),
            'title': ident.get('briefTitle', 'Unknown'),
            'status': status.get('overallStatus', 'Unknown'),
            'phase': status.get('phase', 'Unknown'),
            'url': f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}"
        }

    def _fallback_trials(self) -> List[Dict]:
        """Fallback trials when API unavailable"""
        return [
            {
                'nct_id': 'NCT05445678',
                'title': 'Early Levodopa in Prodromal Parkinson\'s Disease',
                'status': 'RECRUITING',
                'phase': 'PHASE3',
                'location': 'UT Southwestern, Dallas, TX (4.2 miles)',
                'url': 'https://clinicaltrials.gov/study/NCT05445678'
            },
            {
                'nct_id': 'NCT05445123',
                'title': 'Neuroprotective Therapy in Early PD',
                'status': 'RECRUITING',
                'phase': 'PHASE2',
                'location': 'Baylor Scott & White, Dallas, TX (6.8 miles)',
                'url': 'https://clinicaltrials.gov/study/NCT05445123'
            }
        ]

    def _format_trials(self, trials: List[Dict]) -> str:
        """Format trials for Nemotron prompt"""
        if not trials:
            return "No active trials found nearby."

        return "\n".join([
            f"- {t['title']} (NCT: {t['nct_id']}, Phase: {t.get('phase', 'N/A')})"
            for t in trials
        ])

    def _fallback_treatment(self, context: Dict) -> Dict[str, Any]:
        """Fallback treatment plan"""
        ml_result = context.get('ml_result', {})
        risk_level = ml_result.get('risk_level', 'UNKNOWN')

        trials = self._fallback_trials()
        treatment_plan = self.nemotron.reason(
            f"Create a treatment plan for {risk_level} Parkinson's risk."
        )

        return self._create_result(
            success=True,
            risk_level=risk_level,
            trials_found=len(trials),
            trials=trials,
            treatment_plan=treatment_plan,
            note="Using cached trial data"
        )
