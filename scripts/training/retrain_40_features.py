#!/usr/bin/env python3
"""
Retrain Model with Only 40 Reliable Features
Removes the 4 problematic placeholder features: RPDE, DFA, PPE, GNE
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*80)
print("RETRAINING MODEL WITH 40 FEATURES (Removing 4 Placeholder Features)")
print("="*80)

# Features to REMOVE (placeholders that cause problems)
FEATURES_TO_REMOVE = ['RPDE', 'DFA', 'PPE', 'GNE']

print(f"\n❌ Removing {len(FEATURES_TO_REMOVE)} problematic features:")
for f in FEATURES_TO_REMOVE:
    print(f"   • {f}")

# Load data
print("\n📊 Loading data...")
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

print(f"   • Train: {len(train_df)} patients")
print(f"   • Test: {len(test_df)} patients")

# Get feature columns
all_feature_cols = [col for col in train_df.columns
                   if col not in ['ID', 'Status', 'Gender']]

print(f"   • Original features: {len(all_feature_cols)}")

# Remove problematic features
feature_cols = [col for col in all_feature_cols if col not in FEATURES_TO_REMOVE]

print(f"   • Reliable features: {len(feature_cols)}")

# Prepare data
print("\n🔧 Preparing datasets...")
X_train = train_df[feature_cols].values
y_train = train_df['Status'].values
X_test = test_df[feature_cols].values
y_test = test_df['Status'].values

print(f"   • X_train shape: {X_train.shape}")
print(f"   • X_test shape: {X_test.shape}")

# Normalize
print("\n📏 Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save new feature statistics
print("\n💾 Saving new feature statistics...")
feature_stats = {}
for i, col in enumerate(feature_cols):
    feature_stats[col] = {
        'mean': float(scaler.mean_[i]),
        'std': float(scaler.scale_[i])
    }

with open('data/processed/feature_stats_40.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

print("   ✓ Saved to: data/processed/feature_stats_40.json")

# Train Logistic Regression
print("\n🤖 Training Logistic Regression (L2)...")
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("   ✓ Model trained")

# Evaluate on test set
print("\n📊 Evaluating on test set...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("RESULTS (40 Features)")
print("="*80)
print(f"Accuracy:  {accuracy:.1%} ({int(accuracy*len(y_test))}/{len(y_test)} correct)")
print(f"Precision: {precision:.1%}")
print(f"Recall:    {recall:.1%} ({'catches all PD cases' if recall == 1.0 else 'misses some PD cases'})")
print(f"F1-Score:  {f1:.1%}")
print(f"ROC-AUC:   {roc_auc:.1%}")
print("="*80)

# Show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              Healthy    PD")
print(f"Actual Healthy    {cm[0][0]:2d}      {cm[0][1]:2d}")
print(f"       PD         {cm[1][0]:2d}      {cm[1][1]:2d}")

# Show top features
print("\n📊 Top 10 Most Important Features:")
coefficients = model.coef_[0]
feature_importance = [(name, abs(coef)) for name, coef in zip(feature_cols, coefficients)]
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (name, importance) in enumerate(feature_importance[:10], 1):
    print(f"   {i:2d}. {name:<15} {importance:.4f}")

# Save model
print("\n💾 Saving model...")
model_path = 'models/saved_models/LogisticRegression_L2_40feat.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"   ✓ Saved to: {model_path}")

# Save feature list
with open('data/processed/feature_list_40.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("   ✓ Saved feature list to: data/processed/feature_list_40.json")

# Save comparison
comparison = {
    "original_model": {
        "features": 44,
        "includes_placeholders": True,
        "problem": "RPDE, DFA, PPE, GNE are hardcoded"
    },
    "new_model": {
        "features": 40,
        "includes_placeholders": False,
        "removed": FEATURES_TO_REMOVE,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        }
    }
}

with open('models/results/40_vs_44_features.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("\n" + "="*80)
print("✅ RETRAINING COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Test with your voice:")
print("   python3 run.py predict my_voice.wav --model-40")
print("\n2. Or use debug mode:")
print("   python3 debug_predict.py my_voice.wav")
print("\n3. Compare with old model:")
print("   cat models/results/40_vs_44_features.json")
print("="*80 + "\n")
