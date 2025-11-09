# Voice Recording Guide

## 🎤 Quick Start

Just run:
```bash
python3 record_voice.py
```

That's it! The script will guide you through everything.

---

## 📋 What It Does

1. **Asks for your name** (optional - creates filename like "john.wav")
2. **Asks recording length** (5, 7, or 10 seconds - 5 is recommended)
3. **Counts down** (3, 2, 1...)
4. **Records your voice** (shows progress bar)
5. **Checks quality** (volume, noise, silence)
6. **Saves the file** (as WAV format)
7. **Offers to test immediately** (runs prediction)

---

## 🎯 Recording Tips

### Before Recording:
- ✅ Find a **quiet room** (no TV, music, fans)
- ✅ Close windows (reduce street noise)
- ✅ Turn off notifications
- ✅ **Clear your throat** first
- ✅ Position 6-12 inches from mic

### During Recording:
- ✅ Say **"Ahhhhh"** steadily
- ✅ Use **normal speaking voice** (not whisper, not shout)
- ✅ Keep **same volume** throughout
- ✅ Keep **same pitch** (don't go up/down)
- ✅ Breathe before, not during

### Common Mistakes:
- ❌ Whispering (too quiet)
- ❌ Shouting (causes clipping)
- ❌ Starting/stopping multiple times
- ❌ Changing pitch mid-recording
- ❌ Recording in noisy environment

---

## 📊 Quality Checks

The script automatically checks:

### Volume Level:
```
✓ Volume looks good (peak: 0.45)     ← GOOD
⚠️  Recording is very quiet!          ← Speak louder
⚠️  Recording is clipping!            ← Speak softer
```

### Voice Activity:
```
✓ Good voice activity (95% active)   ← GOOD
⚠️  45% silence detected              ← Didn't say "Ahhhhh" enough
```

### Signal-to-Noise Ratio:
```
✓ Good SNR (25.3 dB)                 ← GOOD
⚠️  Low SNR (8.1 dB)                  ← Too much background noise
```

---

## 🧪 Testing Your Recording

The script offers 3 options after recording:

### Option 1: Test with Simple Model (Recommended)
```
Choose: 1
```
- Quick prediction
- Shows risk level
- Uses 14-feature model
- Best for quick testing

### Option 2: Test with Debug Mode
```
Choose: 2
```
- Detailed analysis
- Shows all feature values
- Shows what's unusual
- Best for understanding results

### Option 3: Test Later
```
Choose: 3
```
- Just saves the file
- Test manually later:
  ```bash
  python3 predict_simple.py yourname.wav
  ```

---

## 🔧 Troubleshooting

### Error: "No module named 'sounddevice'"
```bash
pip3 install sounddevice soundfile --user
```

### Error: "No microphone detected"
- Check if microphone is plugged in
- On Mac: System Preferences → Security & Privacy → Microphone → Allow Terminal
- Try restarting Terminal

### Error: "Permission denied"
- Grant microphone permissions to Terminal/Python
- On Mac: System Preferences → Security & Privacy → Microphone

### Recording is too quiet
- Move closer to microphone (6-12 inches)
- Speak louder (normal conversation volume)
- Check microphone isn't muted
- Increase system microphone volume

### Recording has lots of noise
- Find quieter room
- Turn off fans/AC
- Close windows
- Use better microphone (phone > laptop usually)

---

## 📁 Output Files

Recordings are saved as:
- **With name**: `yourname.wav`
- **Without name**: `voice_20250108_143025.wav` (timestamp)

All recordings are WAV format, 48kHz sample rate, mono.

---

## 🎯 Example Session

```bash
$ python3 record_voice.py

======================================================================
🎙️  VOICE RECORDER FOR PARKINSON'S DETECTION
======================================================================

📝 Enter your name (or press Enter for auto-generated filename):
   Name: john

⏱️  Recording duration:
   1. Short (5 seconds) - Recommended
   2. Medium (7 seconds)
   3. Long (10 seconds)

   Choose (1/2/3) [default: 1]: 1

======================================================================
🎙️  VOICE RECORDER - Parkinson's Detection
======================================================================

📝 Recording Settings:
   • Duration: 5 seconds
   • Sample Rate: 48000 Hz
   • Output: john.wav

----------------------------------------------------------------------
INSTRUCTIONS:
----------------------------------------------------------------------
1. Find a QUIET room (no TV, music, or background noise)
2. Position yourself 6-12 inches from the microphone
3. Take a deep breath
4. When countdown finishes, say 'Ahhhhh' steadily
5. Keep the same volume and pitch throughout
6. Don't whisper - use normal speaking voice
----------------------------------------------------------------------

⏱️  Get ready...
   3...
   2...
   1...

🔴 RECORDING NOW! Say 'Ahhhhh' steadily...
   [▓▓▓▓▓] ✓

✅ Recording complete!

💾 Saving to john.wav...
   ✓ Saved (480.2 KB)

🔍 Quick Quality Check:
   ✓ Volume looks good (peak: 0.52)
   ✓ Good voice activity (92.3% active)
   ✓ Good signal-to-noise ratio (23.4 dB)

======================================================================
✅ RECORDING SAVED SUCCESSFULLY!
======================================================================

🧪 Would you like to test this recording now?
   1. Yes - Test with simple model (recommended)
   2. Yes - Test with detailed debug
   3. No - I'll test later

   Choose (1/2/3): 1

🔬 Testing john.wav...
[... prediction results ...]
```

---

## 🎤 Multiple Recordings

Record multiple people:
```bash
python3 record_voice.py   # Enter: john
python3 record_voice.py   # Enter: sarah
python3 record_voice.py   # Enter: mike
```

Then test all:
```bash
python3 predict_simple.py john.wav
python3 predict_simple.py sarah.wav
python3 predict_simple.py mike.wav
```

---

## ✅ Ready to Record!

```bash
python3 record_voice.py
```

**Remember:**
- Quiet room
- Clear throat first
- Normal speaking voice
- Steady "Ahhhhh" for 5 seconds
- Don't whisper or shout

Good luck! 🎙️
