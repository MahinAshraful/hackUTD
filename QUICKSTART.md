# Quick Start Guide

## 🚀 Get Started in 30 Seconds

```bash
# 1. Run test to verify everything works
python3 run.py test

# 2. Check system info
python3 run.py info

# 3. Predict from your own voice recording
python3 run.py predict my_voice.wav
```

## 📋 What You Can Do

### Option 1: Test with Synthetic Audio
```bash
python3 run.py test
```
Generates synthetic "Ahhhhh" sound and runs full prediction pipeline.

**Output:**
- `outputs/audio/test.wav` - Synthetic audio file
- `outputs/reports/test_prediction_report.json` - Detailed results
- Console output with risk assessment

### Option 2: Predict from Real Audio
```bash
# Record 5 seconds of "Ahhhhh" and save as voice.wav
python3 run.py predict voice.wav

# Save detailed report
python3 run.py predict voice.wav --output my_report.json
```

### Option 3: Python API
```python
from src.parkinson_predictor import ParkinsonPredictor

# Initialize
predictor = ParkinsonPredictor()

# Predict
result = predictor.predict('voice.wav')

# Check risk
print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['pd_probability'] * 100:.1f}%")
print(f"Recommendation: {result['recommendation']}")
```

## 🎙️ Recording Your Voice

### On Mac:
1. Open QuickTime Player
2. File → New Audio Recording
3. Click record
4. Say "Ahhhhh" for 5 seconds
5. Save as WAV format

### On Windows:
1. Open Voice Recorder app
2. Click record
3. Say "Ahhhhh" for 5 seconds
4. Save and export as WAV

### On Phone:
1. Use any voice recorder app
2. Record "Ahhhhh" for 5 seconds
3. Transfer file to computer
4. Convert to WAV if needed

### Tips for Best Results:
- ✅ Quiet environment (minimal background noise)
- ✅ Consistent volume (not too loud, not too quiet)
- ✅ Sustained "Ahhhhh" at comfortable pitch
- ✅ At least 5 seconds duration
- ✅ WAV format (22kHz or higher)

## 📊 Understanding Results

### Risk Levels
| Output | Meaning | Next Step |
|--------|---------|-----------|
| **LOW** (< 30%) | Normal voice characteristics | Routine checkup |
| **MODERATE** (30-60%) | Some indicators present | Monitor, retest in 3-6 months |
| **HIGH** (60-80%) | Significant indicators | Consult neurologist |
| **VERY HIGH** (> 80%) | Strong indicators | Urgent neurologist referral |

### Example Output
```
============================================================
RESULTS
============================================================

⚠️  RISK LEVEL: MODERATE

📊 Parkinson's Probability: 54.2%
   Healthy Probability: 45.8%

💡 Recommendation:
   Some indicators present. Monitor and retest in 3-6 months.

============================================================
```

## 🔧 Troubleshooting

### Error: Audio file not found
```bash
# Make sure file path is correct
ls -l my_voice.wav

# Use absolute path if needed
python3 run.py predict /full/path/to/voice.wav
```

### Error: Audio quality check failed
**Issue**: Audio too short, too noisy, or clipped

**Solution:**
- Re-record in quiet environment
- Ensure at least 3-5 seconds duration
- Don't speak too loud (causes clipping)
- Check microphone is working properly

### Error: Module not found
```bash
# Install dependencies
pip3 install -r requirements.txt --user

# Verify installation
python3 -c "import librosa; import parselmouth; print('OK')"
```

### Error: Import errors
```bash
# Make sure you're in project root directory
cd /path/to/hackUTD

# Run from project root
python3 run.py test
```

## 📁 Project Structure Quick Reference

```
hackUTD/
├── run.py              ← Main entry point (use this!)
├── README.md           ← Full documentation
├── QUICKSTART.md       ← This file
│
├── src/                ← Source code
│   ├── parkinson_predictor.py
│   ├── audio_feature_extractor.py
│   └── train_models.py
│
├── data/               ← Data files
│   ├── raw/           ← Original UCI dataset
│   └── processed/     ← Processed train/test splits
│
├── models/            ← Trained ML models
│   ├── saved_models/  ← .pkl model files
│   └── results/       ← Training visualizations
│
├── outputs/           ← Your results go here
│   ├── audio/        ← Generated test audio
│   └── reports/      ← Prediction reports (JSON)
│
├── tests/            ← Test files
└── docs/             ← Documentation
```

## 💡 Next Steps

1. **Test with your own voice** - Record yourself and see the prediction
2. **Try multiple recordings** - Compare results over time
3. **Read the docs** - See [README.md](README.md) for full details
4. **Explore the code** - Check out `src/` to understand how it works
5. **Train your own model** - Run `python3 src/train_models.py`

## ⚠️ Important Disclaimer

This is a **screening tool**, not a medical diagnostic device. Always consult qualified medical professionals for clinical diagnosis and treatment of Parkinson's disease.

## 🆘 Need Help?

- **Documentation**: See [README.md](README.md)
- **Model details**: See [docs/MODEL_SELECTION_PITCH.md](docs/MODEL_SELECTION_PITCH.md)
- **Feature extraction**: See [docs/PHASE2_COMPLETE.md](docs/PHASE2_COMPLETE.md)
- **Issues**: Open a GitHub issue

---

**Ready?** Run `python3 run.py test` to get started!
