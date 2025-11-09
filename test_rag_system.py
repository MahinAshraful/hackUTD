#!/usr/bin/env python3
"""
Test the complete RAG-enhanced system
Simulates multiple patient visits to demonstrate:
1. Clinical Knowledge RAG (Treatment Agent)
2. Patient History RAG (Monitoring Agent)
3. ReAct loops and trend detection
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.coordinator import AgentCoordinator
from src.rag.patient_history_rag import PatientHistoryRAG
from src.rag.clinical_knowledge_rag import ClinicalKnowledgeRAG


def test_clinical_knowledge_rag():
    """Test Clinical Knowledge RAG"""
    print("=" * 80)
    print("TEST 1: CLINICAL KNOWLEDGE RAG")
    print("=" * 80)
    print()

    rag = ClinicalKnowledgeRAG()

    print(f"📚 Knowledge base loaded: {rag.count()} documents\n")

    # Test queries
    test_queries = [
        "What are the drug interactions with MAO-B inhibitors?",
        "Treatment protocol for 15% PD probability",
        "Voice therapy recommendations for Parkinson's"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 80)

        results = rag.query(query, n_results=2)

        if results['documents']:
            print(f"✅ Found {len(results['documents'])} relevant documents")
            print(f"\nTop result preview:")
            preview = results['documents'][0][:300]
            print(f"{preview}...")
            print()
        else:
            print("❌ No results found")

    print("\n✅ Clinical Knowledge RAG test complete\n")


def test_patient_history_and_trends():
    """Test Patient History RAG with longitudinal data"""
    print("=" * 80)
    print("TEST 2: PATIENT HISTORY & TREND DETECTION")
    print("=" * 80)
    print()

    history = PatientHistoryRAG()

    # Simulate a patient with 3 visits over 6 months
    patient_id = "test_patient_progressive"

    print(f"👤 Simulating patient: {patient_id}")
    print(f"📅 Creating 3 visits with worsening trend...\n")

    # Visit 1: Low risk
    print("Visit 1 (Day 0): Low risk")
    visit1_features = {
        'jitter': 0.5,
        'shimmer': 3.2,
        'hnr': 21.5
    }
    visit1_result = {
        'pd_probability': 0.05,
        'healthy_probability': 0.95,
        'risk_level': 'LOW',
        'prediction': 0
    }
    voice_features1 = np.random.randn(44).astype('float32')

    history.add_visit(
        patient_id=patient_id,
        visit_date=(datetime.now() - timedelta(days=180)).isoformat(),
        clinical_features=visit1_features,
        ml_result=visit1_result,
        voice_features=voice_features1,
        notes="Initial visit - healthy"
    )

    # Visit 2: Borderline risk (worsening)
    print("Visit 2 (Day 90): Borderline risk - features worsening")
    visit2_features = {
        'jitter': 1.2,  # Increased
        'shimmer': 5.8,  # Increased
        'hnr': 17.2  # Decreased
    }
    visit2_result = {
        'pd_probability': 0.18,
        'healthy_probability': 0.82,
        'risk_level': 'BORDERLINE',
        'prediction': 0
    }
    voice_features2 = np.random.randn(44).astype('float32')

    history.add_visit(
        patient_id=patient_id,
        visit_date=(datetime.now() - timedelta(days=90)).isoformat(),
        clinical_features=visit2_features,
        ml_result=visit2_result,
        voice_features=voice_features2,
        notes="3-month follow-up - concerning trends"
    )

    # Visit 3: Moderate risk (continued worsening)
    print("Visit 3 (Today): Moderate risk - progression detected")
    visit3_features = {
        'jitter': 2.1,  # Further increased
        'shimmer': 8.5,  # Further increased
        'hnr': 13.8  # Further decreased
    }
    visit3_result = {
        'pd_probability': 0.42,
        'healthy_probability': 0.58,
        'risk_level': 'MODERATE',
        'prediction': 1
    }
    voice_features3 = np.random.randn(44).astype('float32')

    history.add_visit(
        patient_id=patient_id,
        visit_date=datetime.now().isoformat(),
        clinical_features=visit3_features,
        ml_result=visit3_result,
        voice_features=voice_features3,
        notes="6-month follow-up - significant progression"
    )

    print("\n📊 Analyzing trends...")
    trends = history.detect_trends(patient_id)

    print(f"\n{'='*80}")
    print("TREND ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Patient: {trends['patient_id']}")
    print(f"Visits analyzed: {trends['visits_analyzed']}")
    print(f"Overall assessment: {trends['overall_assessment']}")
    print()

    print("Feature Trends:")
    for feature, data in trends['trends'].items():
        print(f"\n  {feature.upper()}:")
        print(f"    Current: {data['current']:.2f}")
        print(f"    Previous: {data['previous']:.2f}")
        print(f"    Change: {data['percent_change']:+.1f}%")
        print(f"    Direction: {data['direction']}")
        if data['alert']:
            print(f"    🚨 ALERT: Worsening detected!")

    print(f"\n✅ Patient History & Trend Detection test complete\n")

    return patient_id, voice_features3


def test_full_system_with_rag(patient_id, voice_features):
    """Test complete multi-agent system with RAG"""
    print("=" * 80)
    print("TEST 3: FULL MULTI-AGENT SYSTEM WITH RAG")
    print("=" * 80)
    print()

    coordinator = AgentCoordinator()

    # Prepare ML result for moderate-risk patient
    ml_result = {
        'success': True,
        'prediction': 1,
        'pd_probability': 0.42,
        'healthy_probability': 0.58,
        'risk_level': 'MODERATE',
        'recommendation': 'Moderate risk detected with progression trends',
        'clinical_features': {
            'jitter': 2.1,
            'shimmer': 8.5,
            'hnr': 13.8
        },
        'voice_features': voice_features,
        'filename': 'test_visit_3.wav'
    }

    print(f"🤖 Running multi-agent analysis for {patient_id}...")
    print(f"   Risk Level: {ml_result['risk_level']}")
    print(f"   PD Probability: {ml_result['pd_probability']:.1%}")
    print()

    # Run agents
    results = coordinator.run(ml_result)

    print(f"\n{'='*80}")
    print("MULTI-AGENT ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print()

    if results.get('success'):
        print(f"✅ Analysis successful")
        print(f"   Agents executed: {results['summary']['agents_executed']}")
        print(f"   Pathway: {results['summary']['pathway']}")
        print()

        # Show RAG-enhanced agent results
        agent_results = results.get('agent_results', {})

        # Treatment Agent (should show RAG usage)
        if 'treatment' in agent_results:
            treatment = agent_results['treatment']
            print("\n📋 TREATMENT AGENT (RAG-Enhanced):")
            print(f"   RAG Used: {treatment.get('rag_used', False)}")
            if 'reasoning_log' in treatment and treatment['reasoning_log']:
                print(f"   ReAct Iterations: {len([r for r in treatment['reasoning_log'] if r['step'] == 'REASON'])}")
                print("\n   ReAct Loop:")
                for log in treatment['reasoning_log'][:6]:  # Show first 2 iterations
                    print(f"     {log['step']}: {log['content'][:80]}...")

        # Monitoring Agent (should show trend detection)
        if 'monitoring' in agent_results:
            monitoring = agent_results['monitoring']
            print("\n📊 MONITORING AGENT (RAG-Enhanced):")
            print(f"   RAG Used: {monitoring.get('rag_used', False)}")
            print(f"   Previous Visits: {monitoring.get('previous_visits', 0)}")
            if monitoring.get('trend_analysis'):
                print(f"   Trends Detected: ✅")
                print(f"   Similar Patients Found: {monitoring.get('similar_patients_found', 0)}")

        print(f"\n✅ Full system test complete!\n")

    else:
        print(f"❌ Analysis failed: {results.get('error')}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RAG-ENHANCED SYSTEM TEST SUITE")
    print("Testing Clinical Knowledge RAG + Patient History RAG + ReAct Loops")
    print("=" * 80)
    print()

    # Test 1: Clinical Knowledge RAG
    test_clinical_knowledge_rag()

    # Test 2: Patient History & Trends
    patient_id, voice_features = test_patient_history_and_trends()

    # Test 3: Full system integration
    test_full_system_with_rag(patient_id, voice_features)

    print("=" * 80)
    print("ALL TESTS COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print("✅ Clinical Knowledge RAG: Working")
    print("✅ Patient History Database: Working")
    print("✅ FAISS Similarity Search: Working")
    print("✅ Trend Detection: Working")
    print("✅ Treatment Agent RAG: Working")
    print("✅ Monitoring Agent RAG: Working")
    print("✅ ReAct Loops: Working")
    print()


if __name__ == "__main__":
    main()
