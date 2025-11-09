#!/usr/bin/env python3
"""
Orchestrator Agent - Plans and coordinates all other agents
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """Plans diagnostic workflow and coordinates agents"""

    def __init__(self, nemotron_client):
        super().__init__("Orchestrator", nemotron_client)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the diagnostic workflow based on ML results

        Args:
            context: Dict with 'ml_result' containing prediction data

        Returns:
            Orchestration plan with agent sequence
        """
        self.start()

        try:
            ml_result = context.get('ml_result', {})
            pd_prob = ml_result.get('pd_probability', 0)
            risk_level = ml_result.get('risk_level', 'UNKNOWN')
            clinical_features = ml_result.get('clinical_features', {})

            # Create prompt for Nemotron
            prompt = f"""You are orchestrating a clinical diagnostic workflow for Parkinson's disease.

ML Model Results:
- PD Probability: {pd_prob:.1%}
- Risk Level: {risk_level}
- Jitter: {clinical_features.get('jitter', 0):.2f}%
- Shimmer: {clinical_features.get('shimmer', 0):.2f}%
- HNR: {clinical_features.get('hnr', 0):.1f} dB

Based on this risk level, plan the optimal agent activation sequence.
Determine which specialized agents should be activated and in what order.
Consider: Research, Risk Assessment, Treatment Planning, Explainer, Monitoring, and Report Generation agents.

Provide a clear, actionable orchestration plan."""

            # Get Nemotron's orchestration plan
            response = self.nemotron.reason(prompt)

            # Determine agent sequence based on risk
            if pd_prob < 0.3:
                pathway = "light_monitoring"
                agents = ['research', 'risk', 'treatment', 'monitoring', 'explainer', 'report']
            elif pd_prob < 0.6:
                pathway = "moderate_intervention"
                agents = ['explainer', 'research', 'risk', 'treatment', 'monitoring', 'report']
            else:
                pathway = "urgent_intervention"
                agents = ['explainer', 'research', 'risk', 'treatment', 'monitoring', 'report']

            result = self._create_result(
                success=True,
                pathway=pathway,
                agent_sequence=agents,
                plan=response,
                reasoning=f"Risk level {risk_level} triggers {pathway} pathway"
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._create_result(success=False, error=str(e))
