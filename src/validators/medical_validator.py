#!/usr/bin/env python3
"""
Medical Output Validator
Prevents hallucinations and validates Nemotron responses for safety
"""

import re
from typing import Dict, Tuple, List, Optional


class MedicalValidator:
    """Validates medical recommendations to prevent hallucinations"""

    def __init__(self):
        # Known safe PD medications with typical dose ranges (mg/day)
        self.safe_medications = {
            'levodopa': {'min': 50, 'max': 1500, 'typical': '300-600'},
            'carbidopa': {'min': 25, 'max': 200, 'typical': '75-150'},
            'rasagiline': {'min': 0.5, 'max': 1, 'typical': '1'},
            'selegiline': {'min': 5, 'max': 10, 'typical': '5-10'},
            'pramipexole': {'min': 0.375, 'max': 4.5, 'typical': '1.5-4.5'},
            'ropinirole': {'min': 0.25, 'max': 24, 'typical': '9-24'},
            'amantadine': {'min': 100, 'max': 400, 'typical': '200-300'}
        }

        # Contraindicated combinations
        self.contraindications = [
            {'drugs': ['rasagiline', 'ssri'], 'reason': 'Serotonin syndrome risk'},
            {'drugs': ['selegiline', 'ssri'], 'reason': 'Serotonin syndrome risk'},
            {'drugs': ['mao-b inhibitor', 'ssri'], 'reason': 'Serotonin syndrome risk'},
            {'drugs': ['levodopa', 'vitamin b6'], 'reason': 'Reduces levodopa effectiveness'}
        ]

        # Red flag terms that should never appear
        self.red_flags = [
            'guaranteed', 'cure', '100% effective', 'miracle',
            'no side effects', 'completely safe', 'risk-free'
        ]

        # Required safety phrases for certain scenarios
        self.required_disclaimers = [
            'consult', 'physician', 'doctor', 'healthcare provider', 'specialist'
        ]

    def validate_treatment_plan(self, plan: str, context: Dict) -> Tuple[bool, str, List[str]]:
        """
        Validate treatment plan for safety and accuracy

        Args:
            plan: Treatment plan text
            context: Context with risk_level, patient_context, etc.

        Returns:
            (is_valid, message, warnings)
        """
        warnings = []

        # 1. Check for red flag language
        is_valid, msg = self._check_red_flags(plan)
        if not is_valid:
            return False, msg, warnings

        # 2. Check for required disclaimers
        is_valid, msg = self._check_disclaimers(plan)
        if not is_valid:
            warnings.append(msg)

        # 3. Validate medication doses
        is_valid, msg, dose_warnings = self._validate_doses(plan)
        if not is_valid:
            return False, msg, warnings
        warnings.extend(dose_warnings)

        # 4. Check contraindications
        patient_meds = context.get('patient_context', {}).get('current_medications', [])
        is_valid, msg = self._check_contraindications(plan, patient_meds)
        if not is_valid:
            return False, msg, warnings

        # 5. Validate risk-appropriate recommendations
        is_valid, msg = self._validate_risk_alignment(plan, context.get('risk_level', 'UNKNOWN'))
        if not is_valid:
            warnings.append(msg)

        # All checks passed
        return True, "Treatment plan validated successfully", warnings

    def _check_red_flags(self, text: str) -> Tuple[bool, str]:
        """Check for dangerous overconfident language"""
        text_lower = text.lower()

        for flag in self.red_flags:
            if flag in text_lower:
                return False, f"RED FLAG: Contains prohibited term '{flag}' (medical overconfidence)"

        return True, ""

    def _check_disclaimers(self, text: str) -> Tuple[bool, str]:
        """Ensure appropriate medical disclaimers present"""
        text_lower = text.lower()

        has_disclaimer = any(term in text_lower for term in self.required_disclaimers)

        if not has_disclaimer:
            return False, "WARNING: Missing healthcare provider consultation disclaimer"

        return True, ""

    def _validate_doses(self, text: str) -> Tuple[bool, str, List[str]]:
        """Validate medication doses are within safe ranges"""
        warnings = []

        # Extract dose mentions: "500mg", "1.5 mg", "100 milligrams"
        dose_pattern = r'(\d+(?:\.\d+)?)\s*(mg|milligram|milligrams)'
        doses = re.findall(dose_pattern, text.lower())

        for dose_str, unit in doses:
            dose = float(dose_str)

            # Check if dose is suspiciously high (>1000mg usually red flag)
            if dose > 1000:
                # Check if it's a known medication
                text_window = text.lower()[max(0, text.lower().find(dose_str)-50):text.lower().find(dose_str)+50]

                is_known_med = False
                for med_name, limits in self.safe_medications.items():
                    if med_name in text_window:
                        is_known_med = True
                        if dose > limits['max']:
                            return False, f"UNSAFE DOSE: {med_name} {dose}mg exceeds maximum safe dose ({limits['max']}mg)", warnings
                        elif dose > limits['max'] * 0.8:
                            warnings.append(f"HIGH DOSE: {med_name} {dose}mg approaching maximum ({limits['max']}mg)")

                # Unknown medication with high dose
                if not is_known_med and dose > 500:
                    warnings.append(f"VERIFY: High dose detected ({dose}mg) - ensure medically appropriate")

        return True, "", warnings

    def _check_contraindications(self, plan: str, patient_meds: List[str]) -> Tuple[bool, str]:
        """Check for contraindicated drug combinations"""
        plan_lower = plan.lower()
        patient_meds_lower = [m.lower() for m in patient_meds]

        # Extract medications mentioned in plan
        recommended_meds = []
        for med in self.safe_medications.keys():
            if med in plan_lower:
                recommended_meds.append(med)

        # Check for contraindications
        for combo in self.contraindications:
            combo_drugs = combo['drugs']

            # Check if plan recommends a drug + patient is on contraindicated drug
            for rec_med in recommended_meds:
                for patient_med in patient_meds_lower:
                    # Check if combination is contraindicated
                    if (rec_med in combo_drugs or any(cd in rec_med for cd in combo_drugs)) and \
                       (patient_med in combo_drugs or any(cd in patient_med for cd in combo_drugs)) and \
                       rec_med != patient_med:
                        return False, f"CONTRAINDICATION: Recommending {rec_med} for patient on {patient_med} ({combo['reason']})"

        return True, ""

    def _validate_risk_alignment(self, plan: str, risk_level: str) -> Tuple[bool, str]:
        """Ensure recommendations match risk level"""
        plan_lower = plan.lower()

        # HIGH risk should mention urgency
        if risk_level == 'HIGH':
            urgent_terms = ['urgent', 'immediate', 'within 2 weeks', 'neurologist', 'specialist']
            has_urgency = any(term in plan_lower for term in urgent_terms)

            if not has_urgency:
                return False, "WARNING: HIGH risk case missing urgency indicators (immediate, neurologist, etc.)"

        # LOW risk should NOT recommend immediate pharmacotherapy
        if risk_level == 'LOW':
            pharm_terms = ['levodopa', 'dopamine agonist', 'mao-b inhibitor', 'medication']
            has_meds = any(term in plan_lower for term in pharm_terms[:3])  # Exclude generic "medication"

            if has_meds:
                return False, "WARNING: LOW risk case inappropriately recommending pharmacotherapy"

        return True, ""

    def validate_explanation(self, explanation: str, prediction: float) -> Tuple[bool, str]:
        """Validate ML explanation for consistency"""
        exp_lower = explanation.lower()

        # Check for contradictory language
        if prediction < 0.3:  # LOW risk
            if 'high risk' in exp_lower or 'severe' in exp_lower:
                return False, "INCONSISTENT: LOW probability but explanation mentions high risk"

        if prediction > 0.6:  # HIGH risk
            if 'low risk' in exp_lower or 'normal' in exp_lower:
                return False, "INCONSISTENT: HIGH probability but explanation mentions low risk"

        # Check for red flags
        is_valid, msg = self._check_red_flags(explanation)
        if not is_valid:
            return False, msg

        return True, "Explanation validated"

    def validate_risk_assessment(self, assessment: str, current_prob: float) -> Tuple[bool, str]:
        """Validate risk trajectory for logical consistency"""
        # Extract probability percentages
        prob_pattern = r'(\d+(?:\.\d+)?)\s*%'
        probs = [float(p) for p in re.findall(prob_pattern, assessment)]

        if len(probs) >= 2:
            # Check that trajectory is reasonable (shouldn't jump >50% in one year)
            for i in range(len(probs)-1):
                diff = abs(probs[i+1] - probs[i])
                if diff > 50:
                    return False, f"UNREALISTIC: Trajectory shows {diff:.0f}% change (too rapid)"

        return True, "Risk assessment validated"


# Global validator instance
validator = MedicalValidator()


def validate_agent_output(agent_name: str, output: Dict, context: Dict) -> Dict:
    """
    Validate agent output with context-specific checks

    Args:
        agent_name: Name of the agent
        output: Agent output dict
        context: Execution context

    Returns:
        Validation result dict with is_valid, message, warnings
    """
    if agent_name == 'TreatmentPlanning':
        plan = output.get('treatment_plan', '')
        is_valid, message, warnings = validator.validate_treatment_plan(plan, context)

        return {
            'is_valid': is_valid,
            'message': message,
            'warnings': warnings,
            'agent': agent_name
        }

    elif agent_name == 'Explainer':
        explanation = output.get('explanation', '')
        prediction = context.get('ml_result', {}).get('pd_probability', 0)
        is_valid, message = validator.validate_explanation(explanation, prediction)

        return {
            'is_valid': is_valid,
            'message': message,
            'warnings': [],
            'agent': agent_name
        }

    elif agent_name == 'RiskAssessment':
        assessment = output.get('trajectory_analysis', '')
        current_prob = context.get('ml_result', {}).get('pd_probability', 0)
        is_valid, message = validator.validate_risk_assessment(assessment, current_prob)

        return {
            'is_valid': is_valid,
            'message': message,
            'warnings': [],
            'agent': agent_name
        }

    else:
        # Generic validation for other agents
        return {
            'is_valid': True,
            'message': 'No specific validation rules',
            'warnings': [],
            'agent': agent_name
        }
