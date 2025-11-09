# Real Data Analysis Workflow

## 🎯 Goal
Train a model on YOUR real voice data (HC vs PD recordings) so predictions actually work!

---

## 📋 **The Workflow**

### **Step 1: Compare Preprocessing Methods** (Optional but recommended)
See how preprocessing improves audio quality

```bash
python3 analyze_preprocessing.py
```

**What it shows:**
- Raw audio stats (volume, SNR, silence)
- Preprocessed audio stats
- Improvements from preprocessing

**Time:** 30 seconds

---

### **Step 2: Analyze Real Data** ⭐ Main Step
Extract features from all HC and PD files, compare them

```bash
python3 analyze_real_data.py
```

**What it does:**
1. ✅ Preprocesses all 81 audio files (normalize, resample, trim)
2. ✅ Extracts 40+ features from each
3. ✅ Compares HC vs PD distributions
4. ✅ Identifies which features actually differ
5. ✅ Creates visualizations
6. ✅ Saves everything to CSV

**Time:** 2-3 minutes

**Output:**
```
data/real_data_features.csv          # All features
data/preprocessed/HC_AH/*.wav        # Cleaned HC audio
data/preprocessed/PD_AH/*.wav        # Cleaned PD audio
outputs/analysis/top_features_distribution.png
outputs/analysis/feature_boxplots.png
outputs/analysis/correlation_heatmap.png
```

---

### **Step 3: Train Model on Real Data**
Build classifier using the extracted features

```bash
python3 train_on_real_data.py
```

**What it does:**
1. ✅ Loads real_data_features.csv
2. ✅ Splits 70% train / 30% test
3. ✅ Trains multiple models
4. ✅ Evaluates performance
5. ✅ Saves best model

**Time:** 1 minute

**Output:**
```
models/saved_models/RealData_best.pkl  # New model
data/real_data_train.csv              # Training set
data/real_data_test.csv               # Test set
```

---

### **Step 4: Test with Your Voice**
See if it works on your recordings!

```bash
python3 predict_realdata.py tanvir.wav
python3 predict_realdata.py mahin.wav
```

**Expected:** Different results (not always 100%!)

---

## 🔬 **What's Different?**

### Old Approach (Broken):
```
UCI Dataset (44 features)
    ↓
Train Model
    ↓
Extract Features (different method!)
    ↓
Predict ❌ (always 100% PD)
```

### New Approach (Fixed):
```
Real Audio (HC + PD)
    ↓
Preprocess (normalize, clean)
    ↓
Extract Features (OUR method)
    ↓
Train Model
    ↓
Extract Features from Your Voice (SAME method)
    ↓
Predict ✅ (actually works!)
```

---

## 📊 **Expected Results**

### Step 2 Will Show You:
```
📊 Statistical Comparison:
Feature          HC Mean    PD Mean   Difference    p-value   Significant
--------------------------------------------------------------------------------
HNR35            75.234     45.123      -30.111      0.0001   ✓✓✓
Jitter_rel        0.324      0.892        0.568      0.0003   ✓✓✓
Shimmer_dB        0.245      0.567        0.322      0.0015   ✓✓
MFCC2            12.345     15.678        3.333      0.0234   ✓
...
```

**This tells you:**
- Which features ACTUALLY differ between HC and PD
- How much they differ
- Statistical significance

### Step 3 Will Show:
```
🏆 BEST MODEL: Random Forest
Accuracy:  85%
Precision: 83%
Recall:    87%
ROC-AUC:   91%
```

Much better than current 70%!

---

## 🎤 **Audio Preprocessing Details**

### What We Do:
```python
1. Resample → 22050 Hz (consistent)
2. Normalize → Volume to -1 to +1
3. Trim → Remove silence from ends
4. Filter → Pre-emphasis (boost high freq)
```

### Why This Helps:
- ✅ **Resampling:** Consistent sample rate (8kHz → 22kHz)
- ✅ **Normalization:** Same volume for all files
- ✅ **Trimming:** No silence affecting features
- ✅ **Filtering:** Better voice characteristics

### Visual:
```
Before Preprocessing:
[silence]▁▁▃▅█▅▃▁▁[silence]  ← Low volume, lots of silence

After Preprocessing:
▃▅███████████▅▃              ← Normalized, trimmed
```

---

## 📈 **Expected Feature Differences**

Based on Parkinson's research, we expect:

### Higher in PD:
- **Jitter** (voice frequency instability)
- **Shimmer** (amplitude variation)
- **MFCC variations** (spectral changes)

### Lower in PD:
- **HNR** (harmonic-to-noise ratio)
- **Pitch stability**

### Maybe No Difference:
- Some MFCCs
- Some deltas
- Placeholder features (RPDE, DFA, PPE, GNE)

**Step 2 will tell us which ones ACTUALLY matter!**

---

## 🐛 **Troubleshooting**

### "ModuleNotFoundError: scipy"
```bash
pip3 install scipy --user
```

### "No files found"
Check your data folder:
```bash
ls data/new_data/HC_AH/*.wav | wc -l  # Should be 41
ls data/new_data/PD_AH/*.wav | wc -l  # Should be 40
```

### "Feature extraction failed"
Some audio files might be corrupted. The script will skip them and report at the end.

### Takes too long
Normal! Processing 81 files with full feature extraction takes 2-3 minutes.

Progress is shown:
```
Processing 41 HC files...
   10/41 done
   20/41 done
   ...
```

---

## 🎯 **Quick Start Commands**

```bash
# Step 1: Check preprocessing (optional)
python3 analyze_preprocessing.py

# Step 2: Analyze real data (required)
python3 analyze_real_data.py

# Step 3: Train model (after step 2)
python3 train_on_real_data.py

# Step 4: Test with your voice
python3 predict_realdata.py tanvir.wav
```

---

## ✅ **Success Criteria**

You'll know it worked if:

1. **Step 2 shows significant features**
   ```
   ✓ HNR35: p=0.0001 (highly significant)
   ✓ Jitter: p=0.0003 (highly significant)
   ```

2. **Step 3 shows good accuracy**
   ```
   ✓ Accuracy > 75%
   ✓ Recall > 80%
   ✓ ROC-AUC > 85%
   ```

3. **Step 4 gives varied predictions**
   ```
   ✓ tanvir.wav: 45% PD (not 100%!)
   ✓ mahin.wav: 32% PD (not 100%!)
   ✓ Different voices → different scores
   ```

---

## 💡 **After This Works**

Once you have a working model:

1. **Add FFmpeg preprocessing** for production
2. **Collect more voice samples** to improve model
3. **Integrate with Nemotron** for intelligent reasoning
4. **Build web app** for easy use
5. **Deploy as API** for remote access

---

Ready to start?

```bash
python3 analyze_real_data.py
```

This is the key step that will solve the "always 100% PD" problem!
