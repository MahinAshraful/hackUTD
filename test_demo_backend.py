#!/usr/bin/env python3
"""
Test script for demo backend
"""

import sys
sys.path.insert(0, '.')

from backend.app import generate_random_clinical_data

print("=" * 80)
print("TESTING DEMO MODE - Random Clinical Data Generation")
print("=" * 80)
print()

# Test 10 random generations to see variety
risk_counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}

print("Generating 10 random clinical profiles:\n")
for i in range(10):
    data = generate_random_clinical_data()
    risk_counts[data['risk_level']] += 1

    print(f"{i+1}. {data['risk_level']:8s} Risk | PD: {data['pd_probability']:.0%} | "
          f"Jitter: {data['jitter']:.2f}% | Shimmer: {data['shimmer']:.2f}% | "
          f"HNR: {data['hnr']:.1f} dB")

print()
print("=" * 80)
print(f"Risk Distribution: LOW={risk_counts['LOW']}, "
      f"MODERATE={risk_counts['MODERATE']}, HIGH={risk_counts['HIGH']}")
print("=" * 80)
print()
print("✅ Demo mode is working! Backend will generate random data for each request.")
print()
print("To start the backend server:")
print("  cd backend && python app.py")
print()
print("Endpoints:")
print("  POST /api/predict           - Basic prediction with random data")
print("  POST /api/predict-enhanced  - Multi-agent Nemotron analysis ⭐")
