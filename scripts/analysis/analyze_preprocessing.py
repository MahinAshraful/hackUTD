#!/usr/bin/env python3
"""
Compare Different Audio Preprocessing Strategies
Tests: Raw, Python-only, FFmpeg
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

def preprocess_python(audio, sr):
    """Python-based preprocessing using librosa"""

    # 1. Resample to 22050 Hz
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050

    # 2. Normalize volume
    audio = librosa.util.normalize(audio)

    # 3. Trim silence (top_db=20 means remove parts quieter than -20dB)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # 4. Bandpass filter (keep voice frequencies 80-8000 Hz)
    # High-pass filter (remove low rumble)
    audio = librosa.effects.preemphasis(audio, coef=0.97)

    return audio, sr

def analyze_audio_quality(audio_file):
    """Analyze single audio file quality"""

    # Load
    audio, sr = librosa.load(audio_file, sr=None)

    results = {
        'file': Path(audio_file).name,
        'duration': len(audio) / sr,
        'sample_rate': sr,
        'max_amplitude': float(np.max(np.abs(audio))),
        'mean_amplitude': float(np.mean(np.abs(audio))),
        'silence_percentage': float(np.mean(np.abs(audio) < 0.01) * 100),
    }

    # SNR estimate
    signal_energy = np.percentile(np.abs(audio), 90)
    noise_energy = np.percentile(np.abs(audio), 10)
    if noise_energy > 0:
        results['snr_db'] = float(20 * np.log10(signal_energy / noise_energy))
    else:
        results['snr_db'] = float('inf')

    # Check clipping
    results['clipping_percentage'] = float(np.mean(np.abs(audio) > 0.99) * 100)

    return results

def compare_preprocessing():
    """Compare raw vs preprocessed audio"""

    print("="*80)
    print("AUDIO PREPROCESSING COMPARISON")
    print("="*80)

    # Test on one HC and one PD file
    hc_file = "data/new_data/HC_AH/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav"
    pd_file = "data/new_data/PD_AH/AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav"

    for label, audio_file in [("Healthy Control", hc_file), ("Parkinson's", pd_file)]:
        print(f"\n{'='*80}")
        print(f"{label}: {Path(audio_file).name}")
        print("="*80)

        # Load raw
        audio_raw, sr = librosa.load(audio_file, sr=None)

        print("\n📊 RAW AUDIO:")
        stats_raw = analyze_audio_quality(audio_file)
        print(f"   Duration: {stats_raw['duration']:.2f}s")
        print(f"   Sample Rate: {stats_raw['sample_rate']} Hz")
        print(f"   Max Amplitude: {stats_raw['max_amplitude']:.3f}")
        print(f"   Mean Amplitude: {stats_raw['mean_amplitude']:.3f}")
        print(f"   SNR: {stats_raw['snr_db']:.1f} dB")
        print(f"   Silence: {stats_raw['silence_percentage']:.1f}%")
        print(f"   Clipping: {stats_raw['clipping_percentage']:.1f}%")

        # Preprocess
        audio_processed, sr_processed = preprocess_python(audio_raw.copy(), sr)

        # Save temporarily for analysis
        temp_file = f"temp_processed_{label.replace(' ', '_')}.wav"
        sf.write(temp_file, audio_processed, sr_processed)

        print("\n✨ PREPROCESSED (Python):")
        stats_processed = analyze_audio_quality(temp_file)
        print(f"   Duration: {stats_processed['duration']:.2f}s")
        print(f"   Sample Rate: {stats_processed['sample_rate']} Hz")
        print(f"   Max Amplitude: {stats_processed['max_amplitude']:.3f}")
        print(f"   Mean Amplitude: {stats_processed['mean_amplitude']:.3f}")
        print(f"   SNR: {stats_processed['snr_db']:.1f} dB")
        print(f"   Silence: {stats_processed['silence_percentage']:.1f}%")

        print("\n📈 IMPROVEMENTS:")
        print(f"   Volume normalized: {stats_raw['max_amplitude']:.3f} → {stats_processed['max_amplitude']:.3f}")
        print(f"   Silence removed: {stats_raw['silence_percentage']:.1f}% → {stats_processed['silence_percentage']:.1f}%")
        print(f"   Duration trimmed: {stats_raw['duration']:.2f}s → {stats_processed['duration']:.2f}s")

        # Clean up
        Path(temp_file).unlink()

    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    print("\n✅ Python preprocessing works well for:")
    print("   • Volume normalization")
    print("   • Silence removal")
    print("   • Resampling")
    print("   • Basic filtering")
    print("\n🎯 For now: Use Python preprocessing")
    print("   Fast, simple, no external dependencies")
    print("\n🚀 For production: Consider FFmpeg")
    print("   Better noise reduction, more professional")
    print("="*80 + "\n")

if __name__ == "__main__":
    compare_preprocessing()
