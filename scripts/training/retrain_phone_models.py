#!/usr/bin/env python3
"""
Complete Model Retraining Pipeline
Extracts features and trains models on phone recording data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

from src.audio_feature_extractor import AudioFeatureExtractor

def extract_phone_features():
    """Extract features from phone recording dataset"""

    print("\n" + "="*80)
    print("EXTRACTING FEATURES FROM PHONE RECORDINGS")
    print("="*80)

    # Initialize feature extractor (no quality validation for batch processing)
    extractor = AudioFeatureExtractor(validate_quality=False)

    # Paths
    hc_dir = Path("data/new_data/HC_AH")
    pd_dir = Path("data/new_data/PD_AH")

    # Get all audio files
    hc_files = sorted(list(hc_dir.glob("*.wav")))
    pd_files = sorted(list(pd_dir.glob("*.wav")))

    print(f"\n📁 Found files:")
    print(f"   Healthy Control (HC): {len(hc_files)} files")
    print(f"   Parkinson's (PD): {len(pd_files)} files")

    # Extract features
    all_features = []
    all_labels = []
    all_filenames = []

    print(f"\n🎵 Extracting features...")

    # Process HC files
    for i, audio_file in enumerate(hc_files, 1):
        try:
            print(f"   [{i}/{len(hc_files)}] HC: {audio_file.name}", end='\r')
            features = extractor.extract_features(str(audio_file), return_dict=True)
            all_features.append([features.get(name, 0.0) for name in extractor.feature_names])
            all_labels.append(0)  # 0 = Healthy
            all_filenames.append(audio_file.name)
        except Exception as e:
            print(f"\n   ⚠️  Failed to process {audio_file.name}: {e}")

    print(f"\n   ✓ Processed {len(hc_files)} HC files")

    # Process PD files
    for i, audio_file in enumerate(pd_files, 1):
        try:
            print(f"   [{i}/{len(pd_files)}] PD: {audio_file.name}", end='\r')
            features = extractor.extract_features(str(audio_file), return_dict=True)
            all_features.append([features.get(name, 0.0) for name in extractor.feature_names])
            all_labels.append(1)  # 1 = Parkinson's
            all_filenames.append(audio_file.name)
        except Exception as e:
            print(f"\n   ⚠️  Failed to process {audio_file.name}: {e}")

    print(f"\n   ✓ Processed {len(pd_files)} PD files")

    # Create DataFrame
    feature_df = pd.DataFrame(all_features, columns=extractor.feature_names)
    feature_df['label'] = all_labels
    feature_df['filename'] = all_filenames

    # Save to CSV
    output_file = "data/phone_recordings_features.csv"
    feature_df.to_csv(output_file, index=False)
    print(f"\n✓ Features saved to: {output_file}")
    print(f"   Total samples: {len(feature_df)}")
    print(f"   HC: {np.sum(all_labels == 0)}, PD: {np.sum(all_labels == 1)}")

    return feature_df

def train_phone_models(feature_df):
    """Train multiple models on phone recording features"""

    print("\n" + "="*80)
    print("TRAINING MODELS ON PHONE RECORDINGS")
    print("="*80)

    # Prepare data
    feature_cols = [col for col in feature_df.columns if col not in ['label', 'filename']]
    X = feature_df[feature_cols].values
    y = feature_df['label'].values

    print(f"\n📊 Dataset:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   HC: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"   PD: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\n🔀 Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature stats
    Path("models/phone_models").mkdir(parents=True, exist_ok=True)

    with open("models/phone_models/Phone_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n💾 Scaler saved: models/phone_models/Phone_scaler.pkl")

    # Save feature stats
    feature_stats = {}
    for i, col in enumerate(feature_cols):
        feature_stats[col] = {
            'mean': float(scaler.mean_[i]),
            'std': float(scaler.scale_[i])
        }

    with open("data/processed/phone_feature_stats.json", 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"💾 Feature stats saved: data/processed/phone_feature_stats.json")

    # Define models
    models = {
        'LogisticRegression_L2': LogisticRegression(penalty='l2', C=1.0, max_iter=2000, random_state=42),
        'LogisticRegression_L1': LogisticRegression(penalty='l1', C=0.5, solver='liblinear', max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'SVM_Linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
        'NeuralNet': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)

    results = []

    for model_name, model in models.items():
        print(f"\n🤖 Training: {model_name}")
        print("-" * 60)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring='roc_auc')

        # Test set predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"   Cross-Val ROC-AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        print(f"   Test Accuracy:     {accuracy:.3f}")
        print(f"   Test ROC-AUC:      {roc_auc:.3f}")
        print(f"   Precision:         {precision:.3f}")
        print(f"   Recall:            {recall:.3f}")
        print(f"   F1-Score:          {f1:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}  TP={cm[1,1]}]]")

        # Save model
        model_path = f"models/phone_models/Phone_{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✓ Saved: {model_path}")

        results.append({
            'model': model_name,
            'cv_roc_auc': cv_scores.mean(),
            'test_roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_roc_auc', ascending=False)

    print("\n" + results_df.to_string(index=False))

    # Save results
    results_df.to_csv("models/phone_models/Phone_model_comparison.csv", index=False)
    print(f"\n💾 Results saved: models/phone_models/Phone_model_comparison.csv")

    # Find best model
    best_model_name = results_df.iloc[0]['model']
    print(f"\n🏆 BEST MODEL: Phone_{best_model_name}")
    print(f"   ROC-AUC: {results_df.iloc[0]['test_roc_auc']:.3f}")
    print(f"   Accuracy: {results_df.iloc[0]['accuracy']:.3f}")

    return results_df

def main():
    """Main retraining pipeline"""

    print("\n" + "="*80)
    print("🚀 PHONE RECORDINGS MODEL RETRAINING PIPELINE")
    print("="*80)

    # Check if features already exist
    features_file = Path("data/phone_recordings_features.csv")

    if features_file.exists():
        print(f"\n📂 Loading existing features from: {features_file}")
        feature_df = pd.read_csv(features_file)
        print(f"   ✓ Loaded {len(feature_df)} samples")
    else:
        print(f"\n📂 Extracting features (this may take a few minutes)...")
        feature_df = extract_phone_features()

    # Train models
    results = train_phone_models(feature_df)

    print("\n" + "="*80)
    print("✅ RETRAINING COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("   • data/phone_recordings_features.csv - Extracted features")
    print("   • data/processed/phone_feature_stats.json - Normalization stats")
    print("   • models/phone_models/Phone_*.pkl - Trained models")
    print("   • models/phone_models/Phone_scaler.pkl - Feature scaler")
    print("   • models/phone_models/Phone_model_comparison.csv - Results")
    print("\n🎯 Use these models for phone/web recordings!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
