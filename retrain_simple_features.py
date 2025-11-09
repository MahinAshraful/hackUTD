#!/usr/bin/env python3
"""
Train Model with ONLY Simple Acoustic Features
Uses only: Jitter, Shimmer, HNR (13 features total)
These match the UCI dataset extraction method
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("="*80)
print("TRAINING WITH SIMPLE ACOUSTIC FEATURES ONLY")
print("="*80)

# Features that WORK (match UCI dataset extraction)
SIMPLE_FEATURES = [
    # Jitter (4)
    'Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ',
    # Shimmer (5)
    'Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shi_APQ11',
    # HNR (5)
    'HNR05', 'HNR15', 'HNR25', 'HNR35', 'HNR38'
]

print(f"\n✅ Using {len(SIMPLE_FEATURES)} reliable acoustic features:")
for f in SIMPLE_FEATURES:
    print(f"   • {f}")

print("\n❌ Excluding problematic features:")
print("   • RPDE, DFA, PPE, GNE (placeholders)")
print("   • MFCC0-12 (extraction method mismatch)")
print("   • Delta0-12 (derived from MFCCs)")

# Load data
print("\n📊 Loading data...")
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train = train_df[SIMPLE_FEATURES].values
y_train = train_df['Status'].values
X_test = test_df[SIMPLE_FEATURES].values
y_test = test_df['Status'].values

print(f"   • X_train shape: {X_train.shape}")
print(f"   • X_test shape: {X_test.shape}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature statistics
feature_stats = {}
for i, col in enumerate(SIMPLE_FEATURES):
    feature_stats[col] = {
        'mean': float(scaler.mean_[i]),
        'std': float(scaler.scale_[i])
    }

with open('data/processed/feature_stats_simple.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

print("   ✓ Saved feature stats")

# Try multiple models
models_to_test = {
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10, probability=True, random_state=42)
}

results = []

for name, model in models_to_test.items():
    print(f"\n🤖 Training {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'model': model
    })

    print(f"   Accuracy: {accuracy:.1%}, Recall: {recall:.1%}, ROC-AUC: {roc_auc:.1%}")

# Find best model
results.sort(key=lambda x: x['roc_auc'], reverse=True)
best_result = results[0]
best_model = best_result['model']
best_name = best_result['name']

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_name}")
print("="*80)
print(f"Accuracy:  {best_result['accuracy']:.1%}")
print(f"Precision: {best_result['precision']:.1%}")
print(f"Recall:    {best_result['recall']:.1%}")
print(f"F1-Score:  {best_result['f1']:.1%}")
print(f"ROC-AUC:   {best_result['roc_auc']:.1%}")
print("="*80)

# Confusion matrix
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              Healthy    PD")
print(f"Actual Healthy    {cm[0][0]:2d}      {cm[0][1]:2d}")
print(f"       PD         {cm[1][0]:2d}      {cm[1][1]:2d}")

# Feature importance
if hasattr(best_model, 'coef_'):
    print("\n📊 Feature Importance:")
    coefficients = best_model.coef_[0]
    feature_importance = [(name, abs(coef)) for name, coef in zip(SIMPLE_FEATURES, coefficients)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"   {i:2d}. {name:<15} {importance:.4f}")

# Save best model
model_path = 'models/saved_models/SimpleAcoustic_best.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

with open('data/processed/feature_list_simple.json', 'w') as f:
    json.dump(SIMPLE_FEATURES, f, indent=2)

print(f"\n💾 Saved model to: {model_path}")
print(f"💾 Saved feature list to: data/processed/feature_list_simple.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nNow test with:")
print("  python3 predict_simple.py tanvir.wav")
print("  python3 predict_simple.py mahin.wav")
print("="*80 + "\n")
