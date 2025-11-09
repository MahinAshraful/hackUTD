#!/usr/bin/env python3
"""
Analyze Real HC vs PD Voice Data
Step-by-step feature extraction and comparison
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.audio_feature_extractor import AudioFeatureExtractor

def preprocess_audio(audio, sr, target_sr=22050):
    """Preprocess audio for consistent feature extraction"""

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize volume
    audio = librosa.util.normalize(audio)

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Pre-emphasis filter (boost high frequencies)
    audio = librosa.effects.preemphasis(audio)

    return audio, sr

def extract_features_from_file(audio_file, extractor, preprocess=True):
    """Extract features from single audio file"""

    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)

        if preprocess:
            # Save preprocessed version
            audio, sr = preprocess_audio(audio, sr)
            preprocessed_dir = Path("data/preprocessed") / Path(audio_file).parent.name
            preprocessed_dir.mkdir(parents=True, exist_ok=True)

            preprocessed_file = preprocessed_dir / Path(audio_file).name
            sf.write(preprocessed_file, audio, sr)

            # Extract from preprocessed
            features_dict = extractor.extract_features(str(preprocessed_file), return_dict=True)
        else:
            # Extract from raw
            features_dict = extractor.extract_features(audio_file, return_dict=True)

        return features_dict, None

    except Exception as e:
        return None, str(e)

def analyze_all_files():
    """Main analysis pipeline"""

    print("\n" + "="*80)
    print("ANALYZING REAL PARKINSON'S VOICE DATA")
    print("="*80)

    # Paths
    hc_dir = Path("data/new_data/HC_AH")
    pd_dir = Path("data/new_data/PD_AH")

    hc_files = sorted(list(hc_dir.glob("*.wav")))
    pd_files = sorted(list(pd_dir.glob("*.wav")))

    print(f"\n📊 Dataset:")
    print(f"   Healthy Controls (HC): {len(hc_files)} files")
    print(f"   Parkinson's (PD):      {len(pd_files)} files")
    print(f"   Total:                 {len(hc_files) + len(pd_files)} files")

    # Initialize extractor (skip quality validation for batch processing)
    print("\n🔧 Initializing feature extractor...")
    extractor = AudioFeatureExtractor(validate_quality=False)

    # Extract features
    print("\n⚙️  Extracting features...")
    print("   (This will take ~2-3 minutes)")

    all_features = []
    failed = []

    # Process HC files
    print(f"\n   Processing {len(hc_files)} HC files...")
    for i, file in enumerate(hc_files, 1):
        features, error = extract_features_from_file(file, extractor, preprocess=True)

        if features:
            features['label'] = 0  # Healthy
            features['filename'] = file.name
            all_features.append(features)
            if i % 10 == 0:
                print(f"      {i}/{len(hc_files)} done")
        else:
            failed.append((file.name, error))

    # Process PD files
    print(f"\n   Processing {len(pd_files)} PD files...")
    for i, file in enumerate(pd_files, 1):
        features, error = extract_features_from_file(file, extractor, preprocess=True)

        if features:
            features['label'] = 1  # Parkinson's
            features['filename'] = file.name
            all_features.append(features)
            if i % 10 == 0:
                print(f"      {i}/{len(pd_files)} done")
        else:
            failed.append((file.name, error))

    print(f"\n   ✓ Processed: {len(all_features)} files")
    if failed:
        print(f"   ✗ Failed: {len(failed)} files")
        for fname, error in failed[:5]:
            print(f"      - {fname}: {error}")

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Save to CSV
    output_file = "data/real_data_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved features to: {output_file}")

    # Analyze differences
    print("\n" + "="*80)
    print("FEATURE ANALYSIS: HC vs PD")
    print("="*80)

    feature_cols = [col for col in df.columns if col not in ['label', 'filename']]

    # Calculate statistics
    hc_data = df[df['label'] == 0][feature_cols]
    pd_data = df[df['label'] == 1][feature_cols]

    print(f"\n📊 Statistical Comparison:")
    print("-"*80)
    print(f"{'Feature':<15} {'HC Mean':>10} {'PD Mean':>10} {'Difference':>12} {'p-value':>10} {'Significant':>12}")
    print("-"*80)

    significant_features = []

    for col in feature_cols:
        hc_values = hc_data[col].dropna()
        pd_values = pd_data[col].dropna()

        if len(hc_values) > 0 and len(pd_values) > 0:
            hc_mean = hc_values.mean()
            pd_mean = pd_values.mean()
            diff = pd_mean - hc_mean

            # t-test
            t_stat, p_value = stats.ttest_ind(hc_values, pd_values)

            is_sig = "✓✓✓" if p_value < 0.001 else ("✓✓" if p_value < 0.01 else ("✓" if p_value < 0.05 else ""))

            print(f"{col:<15} {hc_mean:>10.3f} {pd_mean:>10.3f} {diff:>12.3f} {p_value:>10.4f} {is_sig:>12}")

            if p_value < 0.05:
                significant_features.append((col, abs(diff), p_value))

    # Sort by difference
    significant_features.sort(key=lambda x: x[2])  # Sort by p-value

    print("\n" + "="*80)
    print(f"🎯 TOP 10 DISCRIMINATIVE FEATURES (p < 0.05)")
    print("="*80)

    for i, (feature, diff, p_val) in enumerate(significant_features[:10], 1):
        print(f"   {i:2d}. {feature:<15} (p = {p_val:.4f})")

    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(df, feature_cols, significant_features[:10])

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"   • data/real_data_features.csv - All extracted features")
    print(f"   • data/preprocessed/ - Cleaned audio files")
    print(f"   • outputs/analysis/ - Visualization plots")
    print("\n💡 Next step: Train model on this data")
    print("   Run: python3 train_on_real_data.py")
    print("="*80 + "\n")

def create_visualizations(df, feature_cols, top_features):
    """Create comparison visualizations"""

    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution plots for top features
    if top_features:
        n_features = min(6, len(top_features))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (feature, _, _) in enumerate(top_features[:n_features]):
            hc_values = df[df['label'] == 0][feature].dropna()
            pd_values = df[df['label'] == 1][feature].dropna()

            axes[i].hist(hc_values, bins=20, alpha=0.5, label='HC', color='blue')
            axes[i].hist(pd_values, bins=20, alpha=0.5, label='PD', color='red')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].set_title(f'{feature} Distribution')

        plt.tight_layout()
        plt.savefig(output_dir / "top_features_distribution.png", dpi=300)
        plt.close()
        print(f"   ✓ Saved: top_features_distribution.png")

    # 2. Box plots for comparison
    top_feature_names = [f[0] for f in top_features[:10]]
    if top_feature_names:
        df_melted = df[top_feature_names + ['label']].melt(id_vars=['label'])

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_melted, x='variable', y='value', hue='label')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title('HC vs PD Feature Comparison (Top 10)')
        plt.legend(title='Label', labels=['HC', 'PD'])
        plt.tight_layout()
        plt.savefig(output_dir / "feature_boxplots.png", dpi=300)
        plt.close()
        print(f"   ✓ Saved: feature_boxplots.png")

    # 3. Correlation heatmap
    if len(feature_cols) > 5:
        correlation = df[feature_cols[:20]].corr()  # Top 20 features

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix (Top 20)')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()
        print(f"   ✓ Saved: correlation_heatmap.png")

if __name__ == "__main__":
    analyze_all_files()
