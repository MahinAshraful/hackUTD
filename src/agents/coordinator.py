#!/usr/bin/env python3
"""
Agent Coordinator - Runs multi-agent workflow with parallel execution
"""

from typing import Dict, Any, List, Callable
from src.nemotron_client import NemotronClient
from .orchestrator import OrchestratorAgent
from .research_agent import ResearchAgent
from .risk_agent import RiskAssessmentAgent
from .explainer_agent import ExplainerAgent
from .report_agent import ReportGeneratorAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import RAG-enhanced agents
try:
    from .treatment_agent_rag import TreatmentPlanningAgentRAG
    from .monitoring_agent_rag import MonitoringAgentRAG
    RAG_AVAILABLE = True
except ImportError:
    from .treatment_agent import TreatmentPlanningAgent as TreatmentPlanningAgentRAG
    from .monitoring_agent import MonitoringAgent as MonitoringAgentRAG
    RAG_AVAILABLE = False


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

        # Initialize all agents (using RAG-enhanced versions when available)
        self.orchestrator = OrchestratorAgent(self.nemotron)
        self.research = ResearchAgent(self.nemotron)
        self.risk = RiskAssessmentAgent(self.nemotron)
        self.treatment = TreatmentPlanningAgentRAG(self.nemotron)  # RAG-enhanced
        self.explainer = ExplainerAgent(self.nemotron)
        self.report = ReportGeneratorAgent(self.nemotron)
        self.monitoring = MonitoringAgentRAG(self.nemotron)  # RAG-enhanced

        if RAG_AVAILABLE:
            print("✅ RAG-enhanced agents loaded (Treatment + Monitoring)")
        else:
            print("⚠️  Using standard agents (RAG not available)")

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
        Run complete multi-agent workflow with PARALLEL EXECUTION

        Args:
            ml_result: ML model prediction result
            progress_callback: Optional callback for progress updates
                               Called with (agent_name, status, result)

        Returns:
            Complete analysis with all agent results
        """
        start_time = time.time()

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

            # Step 2: PARALLEL EXECUTION - Run independent agents concurrently
            # Define dependency groups (agents that can run in parallel)
            parallel_groups = [
                # Group 1: All independent agents (don't need each other's results)
                ['explainer', 'research', 'risk'],
                # Group 2: Agents that need Group 1 results
                ['treatment', 'monitoring'],
                # Group 3: Report needs all previous results
                ['report']
            ]

            for group in parallel_groups:
                # Filter to only include agents in the sequence
                agents_to_run = [a for a in group if a in agent_sequence and a in self.agents]

                if not agents_to_run:
                    continue

                # Update context with previous results
                self._update_context(context, results)

                if len(agents_to_run) == 1:
                    # Single agent - run directly
                    self._run_agent(agents_to_run[0], context, results, progress_callback)
                else:
                    # Multiple agents - run in parallel
                    print(f"🚀 Running {len(agents_to_run)} agents in parallel: {', '.join(agents_to_run)}")
                    self._run_agents_parallel(agents_to_run, context, results, progress_callback)

            # Step 3: Generate final summary
            results['summary'] = self._generate_summary(results)
            results['success'] = True
            results['total_duration'] = time.time() - start_time

            print(f"✅ All agents completed in {results['total_duration']:.2f}s")

            return results

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            results['total_duration'] = time.time() - start_time
            return results

    def _run_agents_parallel(self, agent_names: List[str], context: Dict,
                            results: Dict, progress_callback: Callable = None):
        """
        Run multiple agents in parallel using ThreadPoolExecutor

        Args:
            agent_names: List of agent names to run
            context: Shared context (read-only for agents)
            results: Results dict to update
            progress_callback: Optional progress callback
        """
        with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
            # Submit all agents
            futures = {
                executor.submit(self._execute_agent, name, context): name
                for name in agent_names
            }

            # Collect results as they complete
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    agent_result = future.result()

                    # Store result
                    results['agent_results'][agent_name] = agent_result
                    results['timeline'].append({
                        'agent': agent_name,
                        'status': self.agents[agent_name].status,
                        'duration': agent_result.get('duration_seconds', 0)
                    })

                    # Notify completion
                    if progress_callback:
                        progress_callback(agent_name, self.agents[agent_name].status, agent_result)

                    print(f"  ✓ {agent_name.capitalize()} completed")

                except Exception as e:
                    print(f"  ✗ {agent_name.capitalize()} failed: {e}")
                    results['agent_results'][agent_name] = {
                        'success': False,
                        'error': str(e)
                    }

    def _execute_agent(self, agent_name: str, context: Dict) -> Dict[str, Any]:
        """
        Execute a single agent (used for parallel execution)

        Args:
            agent_name: Name of agent to execute
            context: Execution context

        Returns:
            Agent result dict
        """
        agent = self.agents[agent_name]
        return agent.execute(context)

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
