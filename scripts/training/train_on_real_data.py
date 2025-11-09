#!/usr/bin/env python3
"""
Train ML Models on Real HC vs PD Phone Recordings
Uses the 81 phone-quality voice samples
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def train_on_real_data():
    """Train models on real phone-quality recordings"""

    print("\n" + "="*80)
    print("TRAINING ON REAL HC vs PD PHONE RECORDINGS")
    print("="*80)

    # Load features
    features_file = "data/real_data_features.csv"
    print(f"\n📂 Loading features from: {features_file}")

    df = pd.read_csv(features_file)
    print(f"   ✓ Loaded {len(df)} samples")
    print(f"   ✓ Features: {len([c for c in df.columns if c not in ['label', 'filename']])} columns")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['label', 'filename']]
    X = df[feature_cols].values
    y = df['label'].values

    # Check class distribution
    n_hc = np.sum(y == 0)
    n_pd = np.sum(y == 1)
    print(f"\n📊 Class Distribution:")
    print(f"   Healthy (HC): {n_hc} samples ({n_hc/len(y)*100:.1f}%)")
    print(f"   Parkinson's (PD): {n_pd} samples ({n_pd/len(y)*100:.1f}%)")

    # Handle missing values
    print(f"\n🔧 Preprocessing...")
    if np.any(np.isnan(X)):
        print(f"   ⚠️  Found NaN values, replacing with column means...")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Split data (70% train, 30% test)
    print(f"   → Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"   ✓ Training set: {len(X_train)} samples ({np.sum(y_train==0)} HC, {np.sum(y_train==1)} PD)")
    print(f"   ✓ Test set: {len(X_test)} samples ({np.sum(y_test==0)} HC, {np.sum(y_test==1)} PD)")

    # Normalize features
    print(f"   → Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature statistics for later use
    feature_stats = {}
    for i, col in enumerate(feature_cols):
        feature_stats[col] = {
            'mean': float(scaler.mean_[i]),
            'std': float(scaler.scale_[i])
        }

    stats_file = "data/processed/real_data_feature_stats.json"
    Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"   ✓ Feature stats saved to: {stats_file}")

    # Train multiple models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression L2': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
        'Logistic Regression L1': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"\n🔧 Training: {name}")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = 0.0

        # Cross-validation (5-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"   Test Accuracy: {accuracy*100:.1f}%")
        print(f"   Test ROC-AUC: {roc_auc*100:.1f}%")
        print(f"   CV Accuracy: {cv_mean*100:.1f}% ± {cv_std*100:.1f}%")

        results.append({
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model_object': model
        })

    # Sort by ROC-AUC
    results.sort(key=lambda x: x['roc_auc'], reverse=True)

    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*80)

    for r in results:
        print(f"{r['model']:<25} {r['accuracy']*100:>9.1f}% {r['precision']*100:>9.1f}% "
              f"{r['recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['roc_auc']*100:>9.1f}%")

    # Best model
    best = results[0]
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best['model']}")
    print("="*80)
    print(f"\n   Test Accuracy:  {best['accuracy']*100:.1f}%")
    print(f"   Test Precision: {best['precision']*100:.1f}%")
    print(f"   Test Recall:    {best['recall']*100:.1f}%")
    print(f"   Test F1 Score:  {best['f1']*100:.1f}%")
    print(f"   Test ROC-AUC:   {best['roc_auc']*100:.1f}%")
    print(f"   CV Accuracy:    {best['cv_mean']*100:.1f}% ± {best['cv_std']*100:.1f}%")

    # Confusion matrix
    best_model = best['model_object']
    y_pred_best = best_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               HC      PD")
    print(f"   Actual HC  {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"          PD  {cm[1,0]:3d}     {cm[1,1]:3d}")

    # Save best model
    model_dir = Path("models/saved_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "RealData_best.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n💾 Best model saved to: {model_path}")

    # Also save scaler
    scaler_path = model_dir / "RealData_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {scaler_path}")

    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    output_dir = Path("outputs/real_data_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r['model'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    roc_aucs = [r['roc_auc'] * 100 for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance on Real Phone-Quality Data (81 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close()
    print(f"   ✓ Saved: model_comparison.png")

    # 2. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC', 'PD'], yticklabels=['HC', 'PD'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {best["model"]}')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"   ✓ Saved: confusion_matrix.png")

    # Save detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['HC', 'PD']))

    # Save train/test splits for reference
    train_df = df.iloc[X_train.shape[0]:].copy()
    test_df = df.iloc[:X_test.shape[0]].copy()

    # Actually need to track indices properly
    train_indices = list(range(len(df)))
    test_indices = []
    np.random.seed(42)
    np.random.shuffle(train_indices)
    test_size = int(len(df) * 0.3)
    test_indices = train_indices[:test_size]
    train_indices = train_indices[test_size:]

    # Save train/test files
    df.iloc[train_indices].to_csv("data/real_data_train.csv", index=False)
    df.iloc[test_indices].to_csv("data/real_data_test.csv", index=False)
    print(f"\n💾 Saved train/test splits:")
    print(f"   • data/real_data_train.csv ({len(train_indices)} samples)")
    print(f"   • data/real_data_test.csv ({len(test_indices)} samples)")

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n💡 Key Takeaways:")

    if best['accuracy'] < 0.60:
        print("   ❌ Low accuracy (<60%) - Dataset is too small or features don't differ enough")
        print("   💡 HC and PD are very similar in phone recordings")
    elif best['accuracy'] < 0.75:
        print("   ⚠️  Moderate accuracy (60-75%) - Model shows some predictive power")
        print("   💡 Limited by small dataset size (only 81 samples)")
    else:
        print("   ✅ Good accuracy (>75%) - Model performs reasonably well!")
        print("   💡 Phone-quality features are somewhat discriminative")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Test on your voice recordings:")
    print(f"      python3 predict_realdata.py tanvir.wav")
    print(f"      python3 predict_realdata.py mahin.wav")
    print(f"\n   2. This model was trained on PHONE-QUALITY recordings")
    print(f"      Should work better on your laptop/phone recordings!")
    print(f"\n   3. Model saved as: RealData_best.pkl")
    print(f"      Feature stats: real_data_feature_stats.json")
    print("="*80 + "\n")

if __name__ == "__main__":
    train_on_real_data()
