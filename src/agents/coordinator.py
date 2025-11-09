#!/usr/bin/env python3
"""
Agent Coordinator - Runs multi-agent workflow
"""

from typing import Dict, Any, List, Callable
from src.nemotron_client import NemotronClient
from .orchestrator import OrchestratorAgent
from .research_agent import ResearchAgent
from .risk_agent import RiskAssessmentAgent
from .treatment_agent import TreatmentPlanningAgent
from .explainer_agent import ExplainerAgent
from .report_agent import ReportGeneratorAgent
from .monitoring_agent import MonitoringAgent


class AgentCoordinator:
    """Coordinates multi-agent workflow for Parkinson's diagnosis"""

    def __init__(self, nemotron_api_key: str = None):
        """
        Initialize coordinator with all agents

        Args:
            nemotron_api_key: Nvidia API key (optional)
        """
        # Initialize Nemotron client
        self.nemotron = NemotronClient(api_key=nemotron_api_key)

        # Initialize all agents
        self.orchestrator = OrchestratorAgent(self.nemotron)
        self.research = ResearchAgent(self.nemotron)
        self.risk = RiskAssessmentAgent(self.nemotron)
        self.treatment = TreatmentPlanningAgent(self.nemotron)
        self.explainer = ExplainerAgent(self.nemotron)
        self.report = ReportGeneratorAgent(self.nemotron)
        self.monitoring = MonitoringAgent(self.nemotron)

        # Agent registry
        self.agents = {
            'orchestrator': self.orchestrator,
            'research': self.research,
            'risk': self.risk,
            'treatment': self.treatment,
            'explainer': self.explainer,
            'report': self.report,
            'monitoring': self.monitoring
        }

    def run(self, ml_result: Dict[str, Any],
            progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Run complete multi-agent workflow

        Args:
            ml_result: ML model prediction result
            progress_callback: Optional callback for progress updates
                               Called with (agent_name, status, result)

        Returns:
            Complete analysis with all agent results
        """
        context = {
            'ml_result': ml_result
        }

        results = {
            'ml_result': ml_result,
            'agent_results': {},
            'timeline': []
        }

        try:
            # Step 1: Orchestrator plans the workflow
            self._run_agent('orchestrator', context, results, progress_callback)
            orchestration = results['agent_results']['orchestrator']

            # Get agent sequence from orchestrator
            agent_sequence = orchestration.get('agent_sequence', [
                'explainer', 'research', 'risk', 'treatment', 'monitoring', 'report'
            ])

            # Step 2: Run agents in sequence
            for agent_name in agent_sequence:
                if agent_name in self.agents and agent_name != 'orchestrator':
                    # Update context with previous results
                    self._update_context(context, results)

                    # Run agent
                    self._run_agent(agent_name, context, results, progress_callback)

            # Step 3: Generate final summary
            results['summary'] = self._generate_summary(results)
            results['success'] = True

            return results

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            return results

    def _run_agent(self, agent_name: str, context: Dict, results: Dict,
                   progress_callback: Callable = None):
        """Run a single agent and update results"""
        agent = self.agents[agent_name]

        # Notify start
        if progress_callback:
            progress_callback(agent_name, 'started', None)

        # Execute agent
        result = agent.execute(context)

        # Store result
        results['agent_results'][agent_name] = result
        results['timeline'].append({
            'agent': agent_name,
            'status': agent.status,
            'duration': result.get('duration_seconds', 0)
        })

        # Notify completion
        if progress_callback:
            progress_callback(agent_name, agent.status, result)

    def _update_context(self, context: Dict, results: Dict):
        """Update context with previous agent results"""
        agent_results = results.get('agent_results', {})

        # Add research findings
        if 'research' in agent_results:
            context['research_findings'] = agent_results['research']

        # Add risk assessment
        if 'risk' in agent_results:
            context['risk_assessment'] = agent_results['risk']

        # Add treatment plan
        if 'treatment' in agent_results:
            context['treatment_plan'] = agent_results['treatment']

        # Add explanation
        if 'explainer' in agent_results:
            context['explanation'] = agent_results['explainer']

    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate executive summary of all findings"""
        ml_result = results.get('ml_result', {})
        agent_results = results.get('agent_results', {})

        # Extract key points
        risk_level = ml_result.get('risk_level', 'UNKNOWN')
        pd_prob = ml_result.get('pd_probability', 0)

        # Count successful agents
        successful_agents = sum(
            1 for r in agent_results.values()
            if r.get('success', False)
        )

        # Extract key recommendations
        recommendations = []

        if 'treatment' in agent_results:
            treatment = agent_results['treatment']
            if treatment.get('urgent', False):
                recommendations.append('Immediate neurologist consultation recommended')

        if 'monitoring' in agent_results:
            monitoring = agent_results['monitoring']
            schedule = monitoring.get('structured_schedule', {})
            recommendations.append(f"Next follow-up: {schedule.get('next_visit', 'TBD')}")

        if 'research' in agent_results:
            research = agent_results['research']
            implications = research.get('clinical_implications', [])
            recommendations.extend(implications[:2])

        return {
            'risk_level': risk_level,
            'pd_probability': pd_prob,
            'agents_executed': len(agent_results),
            'agents_successful': successful_agents,
            'key_recommendations': recommendations[:5],
            'reports_generated': 'report' in agent_results,
            'pathway': agent_results.get('orchestrator', {}).get('pathway', 'unknown')
        }

    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get status of specific agent"""
        if agent_name in self.agents:
            return self.agents[agent_name].get_status()
        return {'error': f'Agent {agent_name} not found'}

    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
