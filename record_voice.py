#!/usr/bin/env python3
"""
Voice Recorder for Parkinson's Detection
Records 5 seconds of "Ahhhhh" from your microphone
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import sys
from pathlib import Path

def record_voice(duration=5, sample_rate=48000, filename=None):
    """
    Record audio from microphone

    Args:
        duration: Recording length in seconds
        sample_rate: Sample rate in Hz
        filename: Output filename (auto-generated if None)
    """

    print("\n" + "="*70)
    print("🎙️  VOICE RECORDER - Parkinson's Detection")
    print("="*70)

    # Auto-generate filename if not provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}.wav"

    # Ensure .wav extension
    if not filename.endswith('.wav'):
        filename = filename + '.wav'

    print(f"\n📝 Recording Settings:")
    print(f"   • Duration: {duration} seconds")
    print(f"   • Sample Rate: {sample_rate} Hz")
    print(f"   • Output: {filename}")

    print("\n" + "-"*70)
    print("INSTRUCTIONS:")
    print("-"*70)
    print("1. Find a QUIET room (no TV, music, or background noise)")
    print("2. Position yourself 6-12 inches from the microphone")
    print("3. Take a deep breath")
    print("4. When countdown finishes, say 'Ahhhhh' steadily")
    print("5. Keep the same volume and pitch throughout")
    print("6. Don't whisper - use normal speaking voice")
    print("-"*70)

    # Countdown
    print("\n⏱️  Get ready...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)

    print("\n🔴 RECORDING NOW! Say 'Ahhhhh' steadily...")
    print(f"   [", end="", flush=True)

    # Record with progress indicator
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )

    # Show progress
    for i in range(duration):
        time.sleep(1)
        print("▓", end="", flush=True)

    sd.wait()  # Wait until recording is finished
    print("] ✓")

    print("\n✅ Recording complete!")

    # Save to file
    print(f"\n💾 Saving to {filename}...")
    sf.write(filename, audio, sample_rate)

    # Check file size
    file_size = Path(filename).stat().st_size / 1024  # KB
    print(f"   ✓ Saved ({file_size:.1f} KB)")

    # Quick quality check
    print("\n🔍 Quick Quality Check:")

    # Check if audio is too quiet
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < 0.01:
        print("   ⚠️  WARNING: Recording is very quiet!")
        print("      Try speaking louder or moving closer to microphone")
    elif max_amplitude > 0.99:
        print("   ⚠️  WARNING: Recording is clipping (too loud)!")
        print("      Try speaking softer or moving away from microphone")
    else:
        print(f"   ✓ Volume looks good (peak: {max_amplitude:.2f})")

    # Check for silence
    silence_threshold = 0.01
    silence_percentage = np.mean(np.abs(audio) < silence_threshold) * 100
    if silence_percentage > 20:
        print(f"   ⚠️  WARNING: {silence_percentage:.1f}% silence detected")
        print("      Make sure you said 'Ahhhhh' throughout the recording")
    else:
        print(f"   ✓ Good voice activity ({100-silence_percentage:.1f}% active)")

    # Calculate SNR estimate
    signal_energy = np.percentile(np.abs(audio), 90)
    noise_energy = np.percentile(np.abs(audio), 10)
    if noise_energy > 0:
        snr_estimate = 20 * np.log10(signal_energy / noise_energy)
        if snr_estimate < 10:
            print(f"   ⚠️  WARNING: Low signal-to-noise ratio ({snr_estimate:.1f} dB)")
            print("      Recording may be too noisy")
        else:
            print(f"   ✓ Good signal-to-noise ratio ({snr_estimate:.1f} dB)")

    print("\n" + "="*70)
    print("✅ RECORDING SAVED SUCCESSFULLY!")
    print("="*70)

    return filename

def main():
    """Main function with menu"""

    print("\n" + "="*70)
    print("🎙️  VOICE RECORDER FOR PARKINSON'S DETECTION")
    print("="*70)

    # Ask for filename
    print("\n📝 Enter your name (or press Enter for auto-generated filename):")
    name = input("   Name: ").strip()

    if name:
        filename = f"{name}.wav"
    else:
        filename = None

    # Ask for duration
    print("\n⏱️  Recording duration:")
    print("   1. Short (5 seconds) - Recommended")
    print("   2. Medium (7 seconds)")
    print("   3. Long (10 seconds)")

    duration_choice = input("\n   Choose (1/2/3) [default: 1]: ").strip()

    if duration_choice == "2":
        duration = 7
    elif duration_choice == "3":
        duration = 10
    else:
        duration = 5

    # Record
    try:
        recorded_file = record_voice(duration=duration, filename=filename)
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        print("\nPossible issues:")
        print("  • No microphone detected")
        print("  • Microphone permissions not granted")
        print("  • sounddevice library not installed")
        print("\nTry: pip3 install sounddevice soundfile --user")
        sys.exit(1)

    # Ask if user wants to test immediately
    print("\n🧪 Would you like to test this recording now?")
    print("   1. Yes - Test with simple model (recommended)")
    print("   2. Yes - Test with detailed debug")
    print("   3. No - I'll test later")

    test_choice = input("\n   Choose (1/2/3): ").strip()

    if test_choice == "1":
        print(f"\n🔬 Testing {recorded_file}...")
        import subprocess
        subprocess.run(['python3', 'predict_simple.py', recorded_file])
    elif test_choice == "2":
        print(f"\n🔍 Debug analysis of {recorded_file}...")
        import subprocess
        subprocess.run(['python3', 'predict_simple.py', recorded_file, '--debug'])
    else:
        print(f"\n💡 To test later, run:")
        print(f"   python3 predict_simple.py {recorded_file}")

    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
