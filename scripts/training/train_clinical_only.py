#!/usr/bin/env python3
"""
Train Clinical-Only Models (18 features: Jitter, Shimmer, HNR only)
Compare with full 44-feature models
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*80)
print("🏥 CLINICAL-ONLY MODEL TRAINING (18 Features)")
print("="*80)

# Load existing phone features
features_file = "data/phone_recordings_features.csv"
df = pd.read_csv(features_file)

print(f"\n📂 Loaded {len(df)} samples")

# Define clinical features only (no MFCCs, no Deltas)
clinical_features = [
    # Jitter (4)
    'Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ',
    # Shimmer (5)
    'Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shi_APQ11',
    # HNR (5)
    'HNR05', 'HNR15', 'HNR25', 'HNR35', 'HNR38',
    # Advanced (4)
    'RPDE', 'DFA', 'PPE', 'GNE'
]

print(f"\n🏥 Using {len(clinical_features)} clinical features:")
for i, feat in enumerate(clinical_features, 1):
    print(f"   {i:2d}. {feat}")

# Prepare data
X = df[clinical_features].values
y = df['label'].values

print(f"\n📊 Dataset:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(clinical_features)} (clinical only)")
print(f"   HC: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"   PD: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n🔀 Split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
Path("models/clinical_models").mkdir(parents=True, exist_ok=True)
with open("models/clinical_models/Clinical_scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n💾 Scaler saved: models/clinical_models/Clinical_scaler.pkl")

# Train models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'LogisticRegression': LogisticRegression(penalty='l2', C=1.0, max_iter=2000, random_state=42),
    'SVM_Linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}

print("\n" + "="*80)
print("TRAINING CLINICAL-ONLY MODELS")
print("="*80)

results = []

for model_name, model in models.items():
    print(f"\n🤖 Training: Clinical_{model_name}")
    print("-" * 60)

    # Train
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                 scoring='roc_auc')

    # Test predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"   CV ROC-AUC:  {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"   Test ROC-AUC: {roc_auc:.3f}")
    print(f"   Accuracy:     {accuracy:.3f}")
    print(f"   Recall:       {recall:.3f}")

    # Save model
    model_path = f"models/clinical_models/Clinical_{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Saved: {model_path}")

    results.append({
        'model': f'Clinical_{model_name}',
        'features': 'Clinical-Only (18)',
        'cv_roc_auc': cv_scores.mean(),
        'test_roc_auc': roc_auc,
        'accuracy': accuracy,
        'recall': recall
    })

# Compare with full feature models
print("\n" + "="*80)
print("COMPARISON: Clinical-Only vs Full Features")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Test on your recordings
print("\n" + "="*80)
print("TESTING ON YOUR RECORDINGS")
print("="*80)

from src.audio_feature_extractor import AudioFeatureExtractor

test_files = ['mahintest.wav', 'mahin.wav', 'tanvir.wav']
extractor = AudioFeatureExtractor(validate_quality=False)

for audio_file in test_files:
    print(f"\n{'='*80}")
    print(f"FILE: {audio_file}")
    print("="*80)

    # Extract features
    features_dict = extractor.extract_features(audio_file, return_dict=True)
    clinical_values = np.array([features_dict[f] for f in clinical_features])

    # Show clinical markers
    print(f"\n🏥 CLINICAL MARKERS:")
    print(f"   Jitter:  {features_dict['Jitter_rel']*100:.2f}% (normal: <1%)")
    print(f"   Shimmer: {features_dict['Shim_loc']*100:.2f}% (normal: <5%)")
    print(f"   HNR:     {features_dict['HNR05']:.1f} dB (phone: 5-25 dB)")

    # Normalize
    clinical_scaled = scaler.transform(clinical_values.reshape(1, -1))

    print(f"\n📊 PREDICTIONS:")
    print("-" * 60)

    for model_name, model in models.items():
        pred = model.predict(clinical_scaled)[0]
        proba = model.predict_proba(clinical_scaled)[0]
        pd_prob = proba[1]

        risk = 'LOW' if pd_prob < 0.3 else 'MODERATE' if pd_prob < 0.6 else 'HIGH' if pd_prob < 0.8 else 'VERY HIGH'

        print(f"   Clinical_{model_name:20s}: {pd_prob:5.1%} PD → {risk}")

print("\n" + "="*80)
print("✅ CLINICAL-ONLY MODELS COMPLETE")
print("="*80)
print("\n💾 Generated files:")
print("   • models/clinical_models/Clinical_*.pkl")
print("   • models/clinical_models/Clinical_scaler.pkl")
print("\n🎯 These models focus ONLY on medical markers!")
print("="*80 + "\n")
