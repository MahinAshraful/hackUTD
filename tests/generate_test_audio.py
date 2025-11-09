"""
Generate Test Audio File for Parkinson's Predictor
Creates a synthetic 5-second "Ahhhhh" sound
"""

import numpy as np
import soundfile as sf
from scipy import signal

def generate_ahh_sound(duration=5.0, sample_rate=22050):
    """
    Generate synthetic "Ahhhhh" vowel sound

    Args:
        duration: Length in seconds
        sample_rate: Sample rate in Hz

    Returns:
        numpy.ndarray: Audio signal
    """
    print(f"🎵 Generating {duration}s test audio...")

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Fundamental frequency (F0) - typical adult voice ~120-250 Hz
    # Adding slight vibrato (natural voice variation)
    f0 = 150  # Base frequency
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato, ±5 Hz depth
    pitch = f0 + vibrato

    # Phase accumulation for frequency modulation
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate

    # Generate harmonics (vowel "ahh" formants)
    # Formants for /a/ vowel: F1~700Hz, F2~1200Hz, F3~2500Hz
    harmonics = []

    # Fundamental and harmonics
    harmonics.append(1.0 * np.sin(phase))  # F0
    harmonics.append(0.5 * np.sin(2 * phase))  # 2*F0
    harmonics.append(0.3 * np.sin(3 * phase))  # 3*F0
    harmonics.append(0.4 * np.sin(4 * phase))  # 4*F0 (near F1)
    harmonics.append(0.35 * np.sin(5 * phase))  # 5*F0
    harmonics.append(0.25 * np.sin(8 * phase))  # 8*F0 (near F2)
    harmonics.append(0.15 * np.sin(16 * phase))  # 16*F0 (near F3)

    # Combine harmonics
    audio = np.sum(harmonics, axis=0)

    # Add formant filtering to shape vowel sound
    # Simple resonance peaks
    sos1 = signal.butter(4, [650, 750], btype='band', fs=sample_rate, output='sos')
    sos2 = signal.butter(4, [1100, 1300], btype='band', fs=sample_rate, output='sos')
    sos3 = signal.butter(4, [2400, 2600], btype='band', fs=sample_rate, output='sos')

    formant1 = signal.sosfilt(sos1, audio) * 1.5
    formant2 = signal.sosfilt(sos2, audio) * 1.0
    formant3 = signal.sosfilt(sos3, audio) * 0.5

    audio = audio + formant1 + formant2 + formant3

    # Add slight breath noise (makes it more realistic)
    noise = np.random.normal(0, 0.02, len(t))
    audio = audio + noise

    # Apply amplitude envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * sample_rate)  # 0.1s fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    audio = audio * envelope

    # Normalize to [-0.8, 0.8] to avoid clipping
    audio = audio / np.max(np.abs(audio)) * 0.8

    print(f"   ✓ Generated {len(audio)} samples")
    print(f"   ✓ Duration: {duration}s")
    print(f"   ✓ Sample rate: {sample_rate} Hz")
    print(f"   ✓ Frequency: {f0} Hz (with vibrato)")

    return audio, sample_rate

def save_test_audio(filename='test.wav'):
    """
    Generate and save test audio file

    Args:
        filename: Output filename
    """
    print("=" * 60)
    print("TEST AUDIO GENERATOR")
    print("=" * 60)

    # Generate audio
    audio, sr = generate_ahh_sound(duration=5.0, sample_rate=22050)

    # Save as WAV
    print(f"\n💾 Saving to {filename}...")
    sf.write(filename, audio, sr)

    # Verify file
    data, samplerate = sf.read(filename)
    print(f"   ✓ File saved: {filename}")
    print(f"   ✓ File size: {len(data)} samples")
    print(f"   ✓ Duration: {len(data)/samplerate:.2f}s")

    print("\n" + "=" * 60)
    print("✅ Test audio ready!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Listen to test.wav to verify it sounds OK")
    print("   2. Run: python3 test_prediction.py")
    print("   3. See the prediction results!")
    print()

    return filename

if __name__ == "__main__":
    save_test_audio('test.wav')
