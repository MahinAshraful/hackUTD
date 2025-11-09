#!/usr/bin/env python3
"""
Monitoring Agent - Creates personalized follow-up schedules
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class MonitoringAgent(BaseAgent):
    """Plans personalized monitoring schedules"""

    def __init__(self, nemotron_client):
        super().__init__("Monitoring", nemotron_client)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring schedule"""
        self.start()

        try:
            ml_result = context.get('ml_result', {})
            risk_level = ml_result.get('risk_level', 'UNKNOWN')
            pd_prob = ml_result.get('pd_probability', 0)

            prompt = f"""Create a personalized monitoring and follow-up schedule.

**Risk Level:** {risk_level} ({pd_prob:.1%} PD probability)

Create a detailed monitoring plan with:
1. Follow-up timeline (specific dates/months)
2. What to monitor at each visit
3. Red flags to watch for
4. Digital/home monitoring recommendations
5. When to seek immediate help

Be specific with timelines. Adjust intensity based on risk level."""

            schedule = self.nemotron.reason(prompt)

            # Generate structured schedule
            structured = self._create_structured_schedule(risk_level, pd_prob)

            result = self._create_result(
                success=True,
                risk_level=risk_level,
                schedule_description=schedule,
                structured_schedule=structured
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._create_result(success=False, error=str(e))

    def _create_structured_schedule(self, risk_level: str, pd_prob: float) -> Dict:
        """Create structured monitoring schedule"""
        if pd_prob < 0.3:
            return {
                'frequency': 'Biannual',
                'next_visit': '6 months',
                'visits': [
                    {'month': 6, 'type': 'Voice reassessment'},
                    {'month': 12, 'type': 'Annual comprehensive'},
                    {'month': 24, 'type': 'Long-term follow-up'}
                ],
                'home_monitoring': 'Monthly voice samples',
                'alert_threshold': 'Marker change >20%'
            }
        elif pd_prob < 0.6:
            return {
                'frequency': 'Quarterly',
                'next_visit': '3 months',
                'visits': [
                    {'month': 3, 'type': 'Voice + clinical exam'},
                    {'month': 6, 'type': 'Comprehensive + imaging'},
                    {'month': 9, 'type': 'Treatment response'},
                    {'month': 12, 'type': 'Annual review'}
                ],
                'home_monitoring': 'Biweekly voice samples',
                'alert_threshold': 'Any new symptoms'
            }
        else:
            return {
                'frequency': 'Monthly',
                'next_visit': '2 weeks (urgent)',
                'visits': [
                    {'week': 2, 'type': 'Neurologist consultation'},
                    {'month': 1, 'type': 'Treatment initiation'},
                    {'month': 2, 'type': 'Response assessment'},
                    {'month': 3, 'type': 'Comprehensive review'}
                ],
                'home_monitoring': 'Weekly voice samples + symptom diary',
                'alert_threshold': 'Immediate for any worsening'
            }
