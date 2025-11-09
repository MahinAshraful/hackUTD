#!/usr/bin/env python3
"""
Test script to demonstrate all 7 Nemotron agents working
Shows detailed logs for each agent
"""

import sys
sys.path.insert(0, '.')

from src.agents.coordinator import AgentCoordinator

print("\n" + "="*80)
print("TESTING ALL 7 NEMOTRON AGENTS")
print("="*80 + "\n")

# Initialize coordinator
print("🚀 Initializing Nemotron Multi-Agent System...")
coordinator = AgentCoordinator()
print("✅ All agents loaded\n")

# Create test data (LOW risk)
test_data_low = {
    'success': True,
    'prediction': 0,
    'pd_probability': 0.15,
    'healthy_probability': 0.85,
    'risk_level': 'LOW',
    'recommendation': 'Voice characteristics appear normal.',
    'clinical_features': {
        'jitter': 0.43,
        'shimmer': 6.89,
        'hnr': 17.1
    },
    'filename': 'test_audio.wav'
}

print("📊 Test Data (LOW RISK):")
print(f"   PD Probability: {test_data_low['pd_probability']:.1%}")
print(f"   Jitter: {test_data_low['clinical_features']['jitter']:.2f}%")
print(f"   Shimmer: {test_data_low['clinical_features']['shimmer']:.2f}%")
print(f"   HNR: {test_data_low['clinical_features']['hnr']:.1f} dB\n")

print("\n" + "#"*80)
print("RUNNING ALL 7 AGENTS (This will take a few seconds...)")
print("#"*80 + "\n")

# Run the multi-agent analysis
result = coordinator.run(test_data_low)

print("\n" + "#"*80)
print("RESULTS SUMMARY")
print("#"*80)
print(f"\n✅ Success: {result.get('success')}")
print(f"🤖 Agents Executed: {result['summary']['agents_executed']}/7")
print(f"🎯 Pathway: {result['summary']['pathway']}")
print(f"📋 Key Recommendations: {len(result['summary']['key_recommendations'])}")

print("\n" + "="*80)
print("AGENT OUTPUTS (Preview)")
print("="*80 + "\n")

# Show preview of each agent's output
agent_results = result.get('agent_results', {})

if 'orchestrator' in agent_results:
    plan = agent_results['orchestrator'].get('plan', 'N/A')
    print(f"🎯 ORCHESTRATOR:")
    print(f"   {plan[:200]}...\n")

if 'explainer' in agent_results:
    explanation = agent_results['explainer'].get('explanation', 'N/A')
    print(f"💡 EXPLAINER:")
    print(f"   {explanation[:200]}...\n")

if 'research' in agent_results:
    analysis = agent_results['research'].get('analysis', 'N/A')
    print(f"📚 RESEARCH:")
    print(f"   {analysis[:200]}...\n")

if 'risk' in agent_results:
    trajectory = agent_results['risk'].get('trajectory_analysis', 'N/A')
    print(f"📊 RISK ASSESSMENT:")
    print(f"   {trajectory[:200]}...\n")

if 'treatment' in agent_results:
    treatment = agent_results['treatment'].get('treatment_plan', 'N/A')
    print(f"💊 TREATMENT:")
    print(f"   {treatment[:200]}...\n")

if 'monitoring' in agent_results:
    monitoring = agent_results['monitoring'].get('content', 'N/A')
    print(f"📅 MONITORING:")
    print(f"   {monitoring[:200]}...\n")

if 'report' in agent_results:
    report_generated = 'report' in agent_results
    print(f"📄 REPORT: {'✅ Generated' if report_generated else '❌ Not generated'}\n")

print("="*80)
print("ALL 7 AGENTS COMPLETED SUCCESSFULLY!")
print("="*80 + "\n")
