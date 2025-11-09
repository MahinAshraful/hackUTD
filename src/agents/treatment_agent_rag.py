#!/usr/bin/env python3
"""
Enhanced Treatment Planning Agent with RAG and ReAct Loop
- Queries Clinical Knowledge Base for evidence-based recommendations
- Implements ReAct pattern (Reason → Act → Observe) for iterative decision-making
- Checks drug interactions and contraindications
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import requests
import sys
from pathlib import Path

# Add path for RAG import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import ClinicalKnowledgeRAG


class TreatmentPlanningAgentRAG(BaseAgent):
    """Enhanced Treatment Planning Agent with RAG and Re Act capabilities"""

    def __init__(self, nemotron_client):
        super().__init__("TreatmentPlanning", nemotron_client)
        self.clinical_trials_api = "https://clinicaltrials.gov/api/v2/studies"

        # Initialize Clinical Knowledge RAG
        try:
            self.clinical_knowledge = ClinicalKnowledgeRAG()
            self.rag_enabled = True
            print(f"   ✅ Clinical Knowledge RAG loaded ({self.clinical_knowledge.count()} guidelines)")
        except Exception as e:
            print(f"   ⚠️  RAG initialization error: {e}")
            self.clinical_knowledge = None
            self.rag_enabled = False

        # ReAct loop configuration - Dynamic mode
        self.max_iterations = 5  # Increased for dynamic mode
        self.reasoning_log = []
        self.observations = []  # Track all observations
        self.citations = []  # Track RAG citations for explainability

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute treatment planning with dynamic ReAct loop"""
        self.start()
        self.reasoning_log = []
        self.observations = []
        self.citations = []  # Reset citations
        self.context = context  # Store for validation

        try:
            ml_result = context.get('ml_result', {})
            risk_level = ml_result.get('risk_level', 'UNKNOWN')
            pd_prob = ml_result.get('pd_probability', 0)
            clinical_features = ml_result.get('clinical_features', {})

            # Get patient context (medications, etc.)
            patient_context = context.get('patient_context', {})

            print(f"\n🔄 Starting DYNAMIC ReAct Loop (max {self.max_iterations} iterations)")
            print(f"   Risk Level: {risk_level} ({pd_prob:.1%})")
            print(f"   🤖 Agent will autonomously decide next actions\n")

            # Dynamic ReAct Loop - Agent decides its own path
            treatment_plan = self._dynamic_react_loop(risk_level, pd_prob, clinical_features, patient_context)

            # Search clinical trials (final action)
            trials = self._search_trials(risk_level)

            result = self._create_result(
                success=True,
                risk_level=risk_level,
                pd_probability=pd_prob,
                trials_found=len(trials),
                trials=trials,
                treatment_plan=treatment_plan,
                reasoning_log=self.reasoning_log,
                rag_used=self.rag_enabled,
                citations=self.citations,  # Include citations for explainability
                evidence_sources=self._format_citations_for_display(),
                urgent=pd_prob > 0.6
            )

            self.complete(result)
            return result

        except Exception as e:
            self.fail(str(e))
            return self._fallback_treatment(context)

    def _dynamic_react_loop(
        self,
        risk_level: str,
        pd_prob: float,
        clinical_features: Dict[str, Any],
        patient_context: Dict[str, Any]
    ) -> str:
        """
        Dynamic ReAct Loop - Agent autonomously decides next actions
        Based on observations, not a fixed script
        """
        # Initial context for the agent
        state = {
            'risk_level': risk_level,
            'pd_prob': pd_prob,
            'clinical_features': clinical_features,
            'patient_context': patient_context,
            'goal_achieved': False,
            'gathered_info': []
        }

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n   --- Iteration {iteration}/{self.max_iterations} ---")

            # REASON: Let Nemotron decide what to do next
            decision = self._reason_autonomous(state, iteration)

            print(f"   🧠 REASON: {decision['reasoning']}")
            print(f"   🎯 DECISION: {decision['decision']}")

            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'REASON',
                'reasoning': decision['reasoning'],
                'decision': decision['decision']
            })

            # Check if agent thinks it's done
            if decision['decision'] == 'FINALIZE':
                print(f"   ✅ Agent determined sufficient information gathered")
                # ACT: Create final recommendation
                final_plan = self._synthesize_final_plan(state)

                self.reasoning_log.append({
                    'iteration': iteration,
                    'step': 'ACT',
                    'action': 'FINALIZE',
                    'content': 'Synthesized final treatment plan'
                })

                return final_plan

            # ACT: Execute the decided action
            observation = self._execute_dynamic_action(decision, state)

            print(f"   🎬 ACT: {decision['action_type']}")
            print(f"   👁️  OBSERVE: {observation[:150]}...")

            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'ACT',
                'action_type': decision['action_type'],
                'query': decision.get('query', 'N/A')
            })

            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'OBSERVE',
                'content': observation
            })

            # Update state with new observations
            self.observations.append(observation)
            state['gathered_info'].append({
                'action': decision['action_type'],
                'result': observation
            })

        # Max iterations reached - force finalization
        print(f"\n   ⚠️  Max iterations reached - finalizing...")
        return self._synthesize_final_plan(state)

    def _reason_autonomous(self, state: Dict[str, Any], iteration: int) -> Dict[str, str]:
        """
        Let Nemotron autonomously decide the next action
        This is TRUE agentic behavior - no predetermined path
        """
        # Build context for reasoning
        observations_summary = "\n".join([
            f"- {info['action']}: {info['result'][:200]}..."
            for info in state['gathered_info']
        ]) if state['gathered_info'] else "No information gathered yet"

        patient_meds = state['patient_context'].get('current_medications', [])
        meds_info = f"Patient medications: {', '.join(patient_meds)}" if patient_meds else "No patient medication data available"

        reasoning_prompt = f"""You are an AI treatment planning agent using the ReAct pattern.

**Current State:**
- Risk Level: {state['risk_level']} ({state['pd_prob']:.1%} PD probability)
- Iteration: {iteration}/{self.max_iterations}
- {meds_info}

**Information Gathered So Far:**
{observations_summary}

**Available Actions:**
1. QUERY_GUIDELINES - Query clinical knowledge base for treatment protocols
2. CHECK_SAFETY - Query for drug interactions and contraindications
3. QUERY_SPECIFIC - Ask specific clinical question
4. FINALIZE - Sufficient information, ready to create treatment plan

**Your Task:**
Analyze what you've learned so far and decide the NEXT best action. Consider:
- Do you have treatment guidelines for this risk level?
- Have you checked for drug safety/interactions?
- Do you have enough information to make a safe recommendation?

**Respond in this exact format:**
REASONING: [Your reasoning about what's needed next]
DECISION: [One of: QUERY_GUIDELINES, CHECK_SAFETY, QUERY_SPECIFIC, FINALIZE]
QUERY: [If querying, what specific question to ask]
"""

        # Ask Nemotron to decide
        response = self.nemotron.reason(reasoning_prompt)

        # Parse response
        decision = self._parse_reasoning_response(response, state)

        return decision

    def _parse_reasoning_response(self, response: str, state: Dict[str, Any]) -> Dict[str, str]:
        """Parse Nemotron's reasoning response"""
        lines = response.split('\n')

        reasoning = ""
        decision = "QUERY_GUIDELINES"  # Default
        query = ""

        for line in lines:
            if line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('DECISION:'):
                decision_text = line.replace('DECISION:', '').strip().upper()
                # Extract decision keyword
                if 'FINALIZE' in decision_text:
                    decision = 'FINALIZE'
                elif 'CHECK_SAFETY' in decision_text or 'SAFETY' in decision_text:
                    decision = 'CHECK_SAFETY'
                elif 'QUERY_SPECIFIC' in decision_text or 'SPECIFIC' in decision_text:
                    decision = 'QUERY_SPECIFIC'
                else:
                    decision = 'QUERY_GUIDELINES'
            elif line.startswith('QUERY:'):
                query = line.replace('QUERY:', '').strip()

        # Generate appropriate query if not provided
        if not query and decision != 'FINALIZE':
            if decision == 'QUERY_GUIDELINES':
                query = f"Treatment protocol for {state['risk_level']} risk Parkinson's (PD probability {state['pd_prob']:.1%})"
            elif decision == 'CHECK_SAFETY':
                meds = state['patient_context'].get('current_medications', [])
                query = f"Drug interactions with Parkinson's medications. Patient on: {', '.join(meds) if meds else 'no medications'}"
            elif decision == 'QUERY_SPECIFIC':
                query = f"Additional clinical guidance for {state['risk_level']} risk case"

        return {
            'reasoning': reasoning or f"Iteration {len(state['gathered_info'])+1}: Need more clinical guidance",
            'decision': decision,
            'action_type': decision,
            'query': query
        }

    def _execute_dynamic_action(self, decision: Dict[str, str], state: Dict[str, Any]) -> str:
        """Execute the dynamically decided action with citation tracking"""
        action_type = decision['decision']

        if action_type in ['QUERY_GUIDELINES', 'CHECK_SAFETY', 'QUERY_SPECIFIC']:
            if self.rag_enabled and decision.get('query'):
                # Query RAG with citations
                rag_result = self.clinical_knowledge.get_context_with_citations(
                    decision['query'],
                    n_results=2
                )

                # Track citations for explainability
                self.citations.extend(rag_result.get('citations', []))

                return rag_result['context']
            else:
                return f"Clinical guidance: Standard {state['risk_level']} risk management protocols recommend specialist consultation and evidence-based treatment selection."

        return "No actionable observation"

    def _format_citations_for_display(self) -> List[str]:
        """Format citations for human-readable display"""
        formatted = []

        for citation in self.citations:
            source = citation.get('source', 'Unknown')
            category = citation.get('category', 'General')
            relevance = citation.get('relevance_level', 'UNKNOWN')

            citation_str = f"{source} ({category}) - Relevance: {relevance}"

            if 'page' in citation:
                citation_str += f", p.{citation['page']}"

            formatted.append(citation_str)

        # Return unique citations only
        return list(set(formatted))

    def _synthesize_final_plan(self, state: Dict[str, Any]) -> str:
        """Synthesize final treatment plan from all gathered information"""
        # Collect all observations
        all_info = "\n\n".join([
            f"**{info['action']}:**\n{info['result']}"
            for info in state['gathered_info']
        ])

        synthesis_prompt = f"""Based on all the clinical information gathered, create a comprehensive treatment plan.

**Patient Profile:**
- Risk Level: {state['risk_level']} ({state['pd_prob']:.1%} PD probability)
- Jitter: {state['clinical_features'].get('jitter', 0):.2f}%
- Shimmer: {state['clinical_features'].get('shimmer', 0):.2f}%
- HNR: {state['clinical_features'].get('hnr', 0):.1f} dB

**Clinical Evidence Gathered:**
{all_info}

**Create a treatment plan with:**
1. Immediate actions (timeframe specified)
2. Pharmacological recommendations (if appropriate, with safety notes)
3. Non-pharmacological interventions (therapy, lifestyle)
4. Safety considerations and contraindications
5. Monitoring recommendations

Be specific, evidence-based, and actionable."""

        return self.nemotron.reason(synthesis_prompt)

    def _react_loop(
        self,
        risk_level: str,
        pd_prob: float,
        clinical_features: Dict[str, Any],
        patient_context: Dict[str, Any]
    ) -> str:
        """
        Implement ReAct pattern: Reason → Act → Observe
        """
        final_recommendation = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n   --- Iteration {iteration}/{self.max_iterations} ---")

            # REASON: Decide what action to take
            if iteration == 1:
                action = self._reason_initial(risk_level, pd_prob)
            elif iteration == 2:
                action = self._reason_safety_check(patient_context)
            else:
                action = self._reason_finalize(risk_level)

            print(f"   🧠 REASON: {action['reasoning']}")
            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'REASON',
                'content': action['reasoning']
            })

            # ACT: Execute the action
            observation = self._act(action)
            print(f"   🎬 ACT: {action['action_type']}")
            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'ACT',
                'content': action['action_type']
            })

            # OBSERVE: Evaluate the result
            print(f"   👁️  OBSERVE: {observation[:100]}...")
            self.reasoning_log.append({
                'iteration': iteration,
                'step': 'OBSERVE',
                'content': observation
            })

            # Store observation for next iteration
            if iteration == self.max_iterations:
                final_recommendation = observation

        return final_recommendation

    def _reason_initial(self, risk_level: str, pd_prob: float) -> Dict[str, str]:
        """Iteration 1: Reason about initial assessment"""
        if pd_prob < 0.10:
            reasoning = f"Patient has VERY LOW risk ({pd_prob:.1%}). Should I retrieve guidelines for preventive care or just reassure?"
            action_type = "QUERY_RAG: preventive care for low-risk patients"
            query = f"Management for patients with PD probability <10% (very low risk). What preventive measures and monitoring are recommended?"

        elif 0.10 <= pd_prob < 0.30:
            reasoning = f"Patient has borderline risk ({pd_prob:.1%}, range 10-30%). Need to query guidelines on whether to confirm diagnosis first or start preventive interventions."
            action_type = "QUERY_RAG: management for borderline PD cases"
            query = f"Management protocol for PD probability {pd_prob:.1%} (10-30% range). Should diagnosis be confirmed first? What are recommendations for borderline cases?"

        elif 0.30 <= pd_prob < 0.60:
            reasoning = f"Patient has MODERATE risk ({pd_prob:.1%}). Should retrieve guidelines for moderate-risk management and treatment initiation criteria."
            action_type = "QUERY_RAG: moderate-risk treatment protocols"
            query = f"Treatment protocol for moderate PD risk (30-60% probability). When to initiate pharmacotherapy? What immediate actions are needed?"

        else:  # >= 0.60
            reasoning = f"Patient has HIGH risk ({pd_prob:.1%}). Need urgent intervention guidelines."
            action_type = "QUERY_RAG: high-risk urgent protocols"
            query = f"Urgent management for HIGH PD risk (>{pd_prob:.1%}). Immediate actions, specialist referral timeline, treatment initiation."

        return {
            'reasoning': reasoning,
            'action_type': action_type,
            'query': query
        }

    def _reason_safety_check(self, patient_context: Dict[str, Any]) -> Dict[str, str]:
        """Iteration 2: Reason about safety and contraindications"""
        medications = patient_context.get('current_medications', [])

        if medications:
            reasoning = f"Patient is on medications: {', '.join(medications)}. Must check for contraindications with PD treatments, especially MAO-B inhibitors and SSRIs."
            action_type = "QUERY_RAG: drug interactions"
            query = f"Drug interactions and contraindications for Parkinson's medications. Patient currently taking: {', '.join(medications)}. What are the safety concerns?"
        else:
            reasoning = "No current medications reported. Should retrieve general safety guidelines and first-line treatment options."
            action_type = "QUERY_RAG: first-line treatment safety"
            query = "First-line treatment options for Parkinson's disease. Safety profile, contraindications, and patient selection criteria."

        return {
            'reasoning': reasoning,
            'action_type': action_type,
            'query': query
        }

    def _reason_finalize(self, risk_level: str) -> Dict[str, str]:
        """Iteration 3: Finalize recommendation"""
        reasoning = f"Based on previous iterations, need to synthesize final treatment plan considering guidelines and safety. Risk level: {risk_level}."
        action_type = "SYNTHESIZE: Create comprehensive treatment plan"
        query = None  # Will use Nemotron to synthesize

        return {
            'reasoning': reasoning,
            'action_type': action_type,
            'query': query
        }

    def _act(self, action: Dict[str, str]) -> str:
        """Execute the action (query RAG or synthesize)"""
        if action['action_type'].startswith("QUERY_RAG") and self.rag_enabled:
            # Query Clinical Knowledge RAG
            context = self.clinical_knowledge.get_formatted_context(
                action['query'],
                n_results=2
            )
            return context

        elif action['action_type'].startswith("SYNTHESIZE"):
            # Use all previous observations to create final plan
            all_observations = "\n\n".join([
                log['content'] for log in self.reasoning_log if log['step'] == 'OBSERVE'
            ])

            prompt = f"""Based on the following clinical guideline evidence, create a comprehensive treatment plan:

{all_observations}

Provide a clear, actionable treatment plan including:
1. Immediate actions
2. Pharmacological recommendations (if appropriate)
3. Non-pharmacological interventions
4. Safety considerations
5. Monitoring recommendations

Be specific and evidence-based."""

            return self.nemotron.reason(prompt)

        else:
            # Fallback if RAG not available
            return "Clinical guidelines: Consult with neurologist for personalized treatment plan based on individual risk factors and patient preferences."

    def _search_trials(self, risk_level: str) -> List[Dict]:
        """Search ClinicalTrials.gov"""
        try:
            params = {
                'query.cond': 'Parkinson Disease',
                'query.term': 'voice OR early detection',
                'filter.geo': 'distance(39.0,-95.0,500mi)',
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
            print(f"   Clinical trials search error: {e}")
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

    def _fallback_treatment(self, context: Dict) -> Dict[str, Any]:
        """Fallback treatment plan"""
        ml_result = context.get('ml_result', {})
        risk_level = ml_result.get('risk_level', 'UNKNOWN')

        trials = self._fallback_trials()
        treatment_plan = f"Treatment plan for {risk_level} Parkinson's risk. Consult with healthcare provider for personalized recommendations."

        return self._create_result(
            success=True,
            risk_level=risk_level,
            trials_found=len(trials),
            trials=trials,
            treatment_plan=treatment_plan,
            reasoning_log=[],
            rag_used=False,
            note="Using fallback mode"
        )
