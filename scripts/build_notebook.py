"""
Generates and executes notebook/shl_grammar_scoring.ipynb covering:
- Part A: Thoughtful, human-centric explanations (What are we doing? Why are we doing it? What did we learn?)
- Phase 0: Data audit & EDA
- Phase 1: Stratified 5-fold validation protocol & fold usage explanations
- Phase 2: Acoustic baselines (Mean, Duration, Ridge, Random Forest, LightGBM)
- Phase 3: Whisper ASR transcription, transcript quality checks, linguistic feature engineering,
           text-only models, Phase 3 visualizations, and combined acoustic+text validation.
- Phase 4: Text representations (Word/Char TF-IDF baselines, frozen all-MiniLM-L6-v2 embeddings,
           Ridge/ElasticNet regression, feature combinations, preventing high-dimensional overfitting,
           tier-based error analysis, residual correlations, and simple model blending).
- Phase 5: Robustness checks (blend optimism disclosure, multi-seed validation, non-optimized blends),
           submission format verification (216 vs 204 rows resolved), pretrained speech embeddings (wav2vec2-base),
           feature combinations, residual complementarity, score tier error analysis, and final model decision.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

# ==============================================================================
# HEADER
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# SHL Hiring Assessment 2026: Spoken English Grammar Scoring Engine

**Author / Candidate**: Nandini Khandelwal  
**Task**: Build a strong, explainable machine learning pipeline that takes a 45-60 second spoken `.wav` response and predicts a continuous **Grammar Score in [0.0, 5.0]**.  
**Evaluation Metrics**: **RMSE** and **Pearson Correlation ($r$)**  
**Core Deliverables**:
* Reproducible notebook executable from top to bottom
* Explicit **Training RMSE** reporting for all models (Compulsory requirement)
* Thorough validation methodology, feature engineering, robustness checks, and error analysis

---

## Technical Philosophy & Development Roadmap
* **Start simple, then build up**: Never jump to unnecessary complexity. Establish clean reference baselines first.
* **Leakage prevention**: Strict 5-fold stratified cross-validation where all scalers, transformations, and feature selections fit strictly within training folds.
* **Explainability**: Every modeling choice should be easy to explain and defend in an interview.
* **Robustness & Stability**: Multi-seed cross-validation and non-optimized blending checks ensure reported improvements are genuine and not artifacts of lucky splits.
"""))

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Imports, Configuration & Reproducibility Settings

### What are we doing?
We are importing the necessary data science, audio, and NLP libraries, configuring system paths, and fixing the random seed.

### Why are we doing it?
Some machine learning procedures (such as tree subsampling, feature selection, and data splitting) involve randomness. We fix the random seed so that the same experiment produces the exact same split and metrics every time, allowing us to compare models fairly. The specific number 42 is completely arbitrary; consistency across runs is what matters.
"""))

cells.append(nbf.v4.new_code_cell(r"""import os
import sys
import glob
import json
import random
import warnings
from itertools import groupby
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Suppress non-critical warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Aesthetic visualization styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Path Configurations
BASE_DIR = os.path.abspath("..") if os.path.exists("../data") else os.path.abspath(".")
DATA_DIR = os.path.join(BASE_DIR, "data", "Dataset_Final")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("System Initialized.")
"""))

# ==============================================================================
# SECTION 2: DATA LOADING & INTEGRITY
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Loading & Integrity Verification

### What are we doing?
We load the training labels, match audio files to their labels, and verify that there are no corrupt audio files, missing labels, or duplicate path issues.

### Why are we doing it?
Audio files from real-world collection can often suffer from missing headers, zero-length files, or mismatched IDs. Doing a formal audit before building models ensures we never train on corrupt data or leak test files.

> **Critical Gotcha Handled**: 212 audio file names exist in both the `train/` and `test/` folders (e.g. `audio_128.wav`). However, their MD5 checksums prove they are distinct recordings. We therefore always track files by their full directory path, never by filename alone.
"""))

cells.append(nbf.v4.new_code_cell(r"""train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")
train_audio_dir = os.path.join(DATA_DIR, "train")
test_audio_dir = os.path.join(DATA_DIR, "test")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Clean whitespace in column names if any
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Construct unambiguous full paths
train_df['audio_path'] = train_df['filename'].apply(lambda f: os.path.join(train_audio_dir, f))
test_df['audio_path'] = test_df['filename'].apply(lambda f: os.path.join(test_audio_dir, f))

# Verify file existence
train_exists = train_df['audio_path'].apply(os.path.exists).all()
test_exists = test_df['audio_path'].apply(os.path.exists).all()

print(f"Train samples: {len(train_df):,} | All audio files exist: {train_exists}")
print(f"Test samples:  {len(test_df):,}  | All audio files exist: {test_exists}")
print(f"Missing labels in train: {train_df['label'].isnull().sum()}")
"""))

# ==============================================================================
# SECTION 2.1: SUBMISSION FORMAT VERIFICATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 2.1 Submission Format Verification & Discrepancy Resolution

#### What are we doing?
We audit the discrepancy between `test.csv` (216 rows) and `sample_submission.csv` (204 rows) to establish the verified Kaggle submission contract.

#### Why are we doing it?
Submitting the wrong row count or file IDs causes an immediate submission rejection. We verified:
1. **Competition Description Specification**: The official Kaggle competition overview explicitly states:
   - *"Testing (Evaluation): The testing dataset consists of 216 samples."*
   - *"test.csv - This contains the names of the test audio files along with random labels."*
   - *"sample_submission.csv - This contains the sample submission format for a valid submission."*
2. **Audio File Count**: The `Dataset_Final/test/` directory contains exactly **216 `.wav` files** (`audio_0.wav` to `audio_215.wav`), matching `test.csv` 1-to-1.
3. **Discrepancy Root Cause**: `sample_submission.csv` (204 rows) contains audio IDs like `audio_804.wav` and `audio_1028.wav` from an earlier data packaging split; only 25 filenames overlap with the actual test audio folder. It was provided as a structural schema example (`filename, label`) rather than the active evaluation index.
4. **Verified Required Submission Format**:
   - Exactly **216 rows** matching `test.csv`'s filenames.
   - Header: `filename,label`
   - Target values: Continuous numeric scores in $[0.0, 5.0]$.
"""))

cells.append(nbf.v4.new_code_cell(r"""sub_template = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_meta = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_wavs = glob.glob(os.path.join(DATA_DIR, "test", "*.wav"))

print("=== Submission Format Audit ===")
print(f"Rows in sample_submission.csv: {len(sub_template)} (Template Schema: {list(sub_template.columns)})")
print(f"Rows in test.csv:              {len(test_meta)}")
print(f"Actual .wav files in test/:    {len(test_wavs)}")
print(f"Match between test.csv & test/: {set(test_meta['filename']) == set(os.path.basename(f) for f in test_wavs)}")

def format_submission(predictions, test_df, output_path=None):
    # Generates a submission dataframe conforming to competition requirements
    sub = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.clip(predictions, 0.0, 5.0)
    })
    if output_path:
        sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path} ({len(sub)} rows)")
    return sub
"""))

# ==============================================================================
# SECTION 3: EDA & TARGET DISTRIBUTION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis: Target Distribution & Audio Durations

### What are we doing?
We analyze the distribution of the target grammar score and inspect the audio duration profiles for both train and test sets.

### Why are we doing it?
Understanding target skew and class imbalances is critical for choosing the right validation strategy. If rare score values exist, standard random K-Fold will distribute them unevenly across folds, corrupting validation reliability.
"""))

cells.append(nbf.v4.new_code_cell(r"""target = train_df['label']
target_summary = pd.Series({
    'Sample Count': len(target),
    'Distinct Scores': target.nunique(),
    'Min Score': target.min(),
    'Max Score': target.max(),
    'Mean Score': target.mean(),
    'Std Dev': target.std(),
    'Median Score': target.median(),
    '25th Percentile': target.quantile(0.25),
    '75th Percentile': target.quantile(0.75)
})
print("=== Grammar Score Target Summary ===")
print(target_summary.round(4))

freq_df = train_df['label'].value_counts().sort_index().reset_index()
freq_df.columns = ['Rubric Score', 'Count']
freq_df['Percentage (%)'] = (freq_df['Count'] / len(train_df) * 100).round(2)
display(freq_df)
"""))

cells.append(nbf.v4.new_code_cell(r"""# Visualization 1: Target Label Distribution
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
colors = ['#e74c3c' if x <= 1.5 else '#3498db' if x < 4.0 else '#2ecc71' for x in freq_df['Rubric Score']]
bars = ax.bar(freq_df['Rubric Score'].astype(str), freq_df['Count'], color=colors, edgecolor='#2c3e50', width=0.6, alpha=0.9)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title("Grammar Score Label Distribution (Train Set)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Grammar Score Rubric (0.0 to 5.0)", fontsize=11, labelpad=8)
ax.set_ylabel("Number of Samples", fontsize=11, labelpad=8)
ax.set_ylim(0, max(freq_df['Count']) + 25)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(r"""# Audio Duration Analysis
train_durations = [sf.info(p).duration for p in train_df['audio_path']]
test_durations = [sf.info(p).duration for p in test_df['audio_path']]

train_df['duration'] = train_durations
test_df['duration'] = test_durations

print(f"Train Duration (sec): Min={min(train_durations):.2f}, Max={max(train_durations):.2f}, Mean={np.mean(train_durations):.2f}, Median={np.median(train_durations):.2f}")
print(f"Test Duration (sec):  Min={min(test_durations):.2f}, Max={max(test_durations):.2f}, Mean={np.mean(test_durations):.2f}, Median={np.median(test_durations):.2f}")

# Visualization 2: Audio Duration Distribution
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
sns.kdeplot(train_durations, ax=ax, label=f"Train (Mean: {np.mean(train_durations):.1f}s)", fill=True, color='#2980b9', alpha=0.35, linewidth=2)
sns.kdeplot(test_durations, ax=ax, label=f"Test (Mean: {np.mean(test_durations):.1f}s)", fill=True, color='#e67e22', alpha=0.35, linewidth=2)
ax.set_title("Audio Sample Duration Distribution: Train vs Test", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Duration (seconds)", fontsize=11, labelpad=8)
ax.set_ylabel("Density", fontsize=11, labelpad=8)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### What did we learn?
* **Severe target imbalance**: The rubric has 10 distinct values in $[0.0, 5.0]$, but scores `1.0` (1 sample) and `1.5` (3 samples) are extraordinarily rare. The distribution is skewed toward moderate-to-high scores ($3.0$ is the mode with 174 samples).
* **Audio lengths**: Training responses average $55.8$ seconds (most are ~45-60 seconds, typical for recorded prompts). Test responses average $48.8$ seconds with a slightly wider range ($6.5$s to $61.0$s).
"""))

# ==============================================================================
# SECTION 4: ACOUSTIC SIGNAL VISUALIZATIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Acoustic Signal & Spectrogram Exploration

### What are we doing?
We plot the speech waveform, short-time energy, detected pause regions, and the log-mel spectrogram for a representative sample response.

### Why are we doing it?
A waveform shows how signal amplitude changes over time, while a Mel spectrogram shows how audio energy is distributed across time and frequency. This gives us an intuitive way to inspect pauses, speech activity, and changes in acoustic structure.

> **Important Note on Speech Interpretation**: A spectrogram can show acoustic regions, speech formants, and pauses, but identifying specific words like "um" and "uh" is better handled using speech-to-text. We use acoustics to study speech rhythm and pauses, reserving lexical filler detection for transcript analysis.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Load a representative sample for acoustic inspection
sample_audio_path = train_df.iloc[0]['audio_path']
sample_label = train_df.iloc[0]['label']
y_sig, sr = sf.read(sample_audio_path)
if y_sig.ndim > 1:
    y_sig = np.mean(y_sig, axis=1)

time_axis = np.linspace(0, len(y_sig) / sr, len(y_sig))

# Frame-level RMS energy for pause detection
hop_vad = int(0.01 * sr)   # 10ms hop
win_vad = int(0.025 * sr)  # 25ms window
num_frames = (len(y_sig) - win_vad) // hop_vad + 1
strided = np.lib.stride_tricks.as_strided(
    y_sig, shape=(num_frames, win_vad), strides=(y_sig.strides[0]*hop_vad, y_sig.strides[0])
)
rms_frames = np.sqrt(np.mean(strided**2, axis=1))
time_frames = np.linspace(0, len(y_sig) / sr, len(rms_frames))

# Pause detection threshold: 10% of 90th percentile RMS
p90 = np.percentile(rms_frames, 90)
threshold = 0.10 * p90
is_silence = rms_frames < threshold

# Visualization 3: Waveform with Pause Detection
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, dpi=100)

ax1.plot(time_axis, y_sig, color='#2c3e50', alpha=0.65, linewidth=0.5)
ax1.set_title(f"Speech Waveform: {os.path.basename(sample_audio_path)} (Score: {sample_label})", fontsize=11, fontweight='bold')
ax1.set_ylabel("Amplitude", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.4)

ax2.plot(time_frames, rms_frames, color='#2980b9', linewidth=1.2, label='Short-Time RMS Energy')
ax2.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=1.2, label=f'Silence Threshold ({threshold:.3f})')
ax2.fill_between(time_frames, 0, max(rms_frames), where=is_silence, color='#e74c3c', alpha=0.25, label='Detected Silence / Pause')
ax2.set_title("Short-Time Energy & Detected Pauses", fontsize=11, fontweight='bold')
ax2.set_xlabel("Time (seconds)", fontsize=10)
ax2.set_ylabel("RMS Energy", fontsize=10)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(r"""# Visualization 4: Log-Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y_sig, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

fig, ax = plt.subplots(figsize=(11, 4), dpi=100)
img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Log Power (dB)')
ax.set_title("Log-Mel Spectrogram (0 - 8 kHz)", fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel("Time (seconds)", fontsize=10)
ax.set_ylabel("Frequency (Hz)", fontsize=10)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### What did we learn?
* Energy drops during pauses correspond directly to silence intervals in speech.
* Candidates with lower scores often exhibit longer, more frequent pauses and erratic energy envelopes, reflecting hesitation.
* Spectral energy concentrates around speech formants ($200 - 3000$ Hz), confirming that Mel filterbank representations and MFCCs will capture meaningful timbre and vocal tract shape.
"""))

# ==============================================================================
# SECTION 5: PHASE 1 - VALIDATION FRAMEWORK
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Phase 1 - Validation Framework

### What are we doing?
We construct a stratified 5-fold cross-validation split based on target score bins, and implement a leak-free evaluation framework.

### Why Stratified K-Fold?
Our target is a regression score, but the training labels come from a small set of rubric values and some low scores are extremely rare.
We grouped the rare low-score labels to make the overall target distribution more consistent across folds. Because there are only four samples at 1.0 or 1.5, those individual labels cannot appear in every fold. We therefore group them with 2.0 to stabilize fold splits.

### Why 5 folds?
We only have 769 labelled examples, so using 5 folds lets us train on most of the data (80%) while still evaluating on a meaningful validation portion each time.

### How do we use the 5 folds?
* **We do not choose one fold as the "best" fold.** Each fold is used as validation once, while the other four folds are used for training. This gives an out-of-fold (OOF) prediction for every single training sample.
* **Model selection is based on overall cross-validation / OOF results**, rather than the score of one particular fold.
* **After choosing the final modeling approach**, the model is retrained on all available training data before generating test predictions.

### Training vs Validation Metrics
The training RMSE is required by the competition, but we use OOF/CV performance for model comparison because training error can be overly optimistic.

### Why track both RMSE and Pearson Correlation?
* **RMSE** tells us how far our predicted grammar scores are from the actual scores, with larger mistakes receiving more penalty.
* **Pearson Correlation ($r$)** tells us whether the model correctly tracks the relative ordering and trend of grammar quality across candidates, even when predictions are not exactly equal to true scores.
* Both are official competition metrics, so both must be tracked.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Target Stratification Bins
def create_stratified_bins(y):
    # Map continuous rubric values into stratification bins,
    # grouping rare 1.0 and 1.5 with 2.0 so low-score examples are balanced.
    bins = np.zeros(len(y), dtype=int)
    for idx, val in enumerate(y):
        if val <= 0.2:
            bins[idx] = 0
        elif val <= 2.2:
            bins[idx] = 1 # Combines rare 1.0 (1 sample), 1.5 (3 samples), and 2.0
        elif val <= 2.7:
            bins[idx] = 2
        elif val <= 3.2:
            bins[idx] = 3
        elif val <= 3.7:
            bins[idx] = 4
        elif val <= 4.2:
            bins[idx] = 5
        elif val <= 4.7:
            bins[idx] = 6
        else:
            bins[idx] = 7
    return bins

strat_bins = create_stratified_bins(train_df['label'])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_df['fold'] = -1
for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, strat_bins)):
    train_df.loc[val_idx, 'fold'] = fold_idx

# Verify fold distribution
fold_dist = pd.crosstab(train_df['fold'], strat_bins, rownames=['Fold'], colnames=['Strat Bin'])
print("Stratification Bins Distribution across 5 Folds:")
display(fold_dist)
"""))

cells.append(nbf.v4.new_code_cell(r"""# Leak-Free CV Evaluator
def safe_pearsonr(y_true, y_pred):
    # Calculates Pearson r safely, handling constant prediction edge cases.
    if np.all(y_pred == y_pred[0]) or np.std(y_pred) < 1e-12:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(0.0 if np.isnan(r) else r)

def evaluate_cv(X, y, folds, model_factory, clip_range=(0.0, 5.0)):
    # Runs 5-fold cross validation with strict fold isolation.
    n_splits = len(np.unique(folds))
    oof_preds = np.zeros(len(y), dtype=float)
    fold_train_rmses = []
    fold_val_rmses = []
    fold_val_pearsons = []
    fitted_models = []
    
    for fold in range(n_splits):
        train_mask = (folds != fold)
        val_mask = (folds == fold)
        
        if isinstance(X, pd.DataFrame):
            X_tr, y_tr = X.iloc[train_mask], y[train_mask]
            X_va, y_va = X.iloc[val_mask], y[val_mask]
        else:
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_va, y_va = X[val_mask], y[val_mask]
            
        model = model_factory()
        model.fit(X_tr, y_tr)
        
        train_pred = np.clip(model.predict(X_tr), *clip_range)
        val_pred = np.clip(model.predict(X_va), *clip_range)
        
        oof_preds[val_mask] = val_pred
        fitted_models.append(model)
        
        fold_train_rmses.append(root_mean_squared_error(y_tr, train_pred))
        fold_val_rmses.append(root_mean_squared_error(y_va, val_pred))
        fold_val_pearsons.append(safe_pearsonr(y_va, val_pred))
        
    full_model = model_factory()
    full_model.fit(X, y)
    full_train_pred = np.clip(full_model.predict(X), *clip_range)
    full_train_rmse = float(root_mean_squared_error(y, full_train_pred))
    
    return {
        "cv_rmse_mean": float(np.mean(fold_val_rmses)),
        "cv_rmse_std": float(np.std(fold_val_rmses)),
        "cv_pearson_mean": float(np.mean(fold_val_pearsons)),
        "cv_pearson_std": float(np.std(fold_val_pearsons)),
        "oof_rmse": float(root_mean_squared_error(y, oof_preds)),
        "oof_pearson": float(safe_pearsonr(y, oof_preds)),
        "train_rmse": full_train_rmse,
        "oof_preds": oof_preds,
        "fitted_models": fitted_models
    }
"""))

cells.append(nbf.v4.new_markdown_cell("""### What did we learn?
Stratifying by target bins ensures each of the 5 validation folds contains a representative spread of scores. All feature transformations (like scalers) will be fit strictly within training folds, ensuring zero data leakage into validation folds.
"""))

# ==============================================================================
# SECTION 6: PHASE 2 - ACOUSTIC BASELINES
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Phase 2 - Baseline Models & Handcrafted Acoustic Features

### What are we doing?
We construct our benchmark progression:
1. **Mean Baseline**: Predicts the training mean.
2. **Duration-Only Baseline**: Uses recording length only.
3. **Handcrafted Acoustic Baseline**: Extracts 75 features and evaluates Ridge, Random Forest, and LightGBM.

### Why handcrafted acoustic features?
Grammar is the final target, but spoken communication also has acoustic patterns such as pauses, speaking rhythm, and energy changes. We start with interpretable acoustic features to see how much useful information is available before using larger pretrained models:
* **RMS Energy**: Measures the strength and amplitude variance of the audio signal over time.
* **Zero-Crossing Rate (ZCR)**: Measures how rapidly the waveform changes sign, helping describe noisy vs tonal speech segments.
* **Spectral Centroid & Rolloff**: Indicates where spectral energy is concentrated in frequency, reflecting vocal brightness and clarity.
* **MFCCs**: Compactly describe the spectral envelope and vocal tract resonances.
* **Pause / Silence Features**: Measure how much time the speaker spends in low-energy regions and how frequently long pauses occur.
* **Pitch / F0**: Describes fundamental vocal frequency and pitch variability.

> *Note*: None of these are direct measures of grammar, but they reflect fluency and delivery.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Experiment Tracker
experiment_log_path = os.path.join(ARTIFACTS_DIR, "experiment_log.csv")

class ExperimentTracker:
    def __init__(self, path):
        self.path = path
        self.rows = []
        if os.path.exists(path):
            self.df = pd.read_csv(path)
            self.rows = self.df.to_dict('records')
        else:
            self.df = pd.DataFrame()
            
    def record(self, exp_id, features, model, res, notes=""):
        entry = {
            "experiment_id": exp_id,
            "features": features,
            "model": model,
            "cv_rmse_mean": round(res["cv_rmse_mean"], 4),
            "cv_rmse_std": round(res["cv_rmse_std"], 4),
            "cv_pearson_mean": round(res["cv_pearson_mean"], 4),
            "cv_pearson_std": round(res["cv_pearson_std"], 4),
            "oof_rmse": round(res["oof_rmse"], 4),
            "oof_pearson": round(res["oof_pearson"], 4),
            "train_rmse": round(res["train_rmse"], 4),
            "notes": notes
        }
        self.rows = [r for r in self.rows if r['experiment_id'] != exp_id] + [entry]
        self.df = pd.DataFrame(self.rows)
        self.df.to_csv(self.path, index=False)
        return self.df

tracker = ExperimentTracker(experiment_log_path)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Baseline 1: Mean Predictor

#### What are we doing?
Predicting the average training score for every example.

#### Why are we doing it?
Before building ML models, we need a simple benchmark. Predicting the average score gives us a minimum reference floor. If a model cannot beat this, it is not learning useful signal.
"""))

cells.append(nbf.v4.new_code_cell(r"""class MeanPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)

y_true = train_df['label'].values
folds = train_df['fold'].values

res_mean = evaluate_cv(np.zeros((len(y_true), 1)), y_true, folds, MeanPredictor)
tracker.record("EXP-01-MEAN", "None", "MeanPredictor", res_mean, "Reference floor baseline")

print(f"=== Baseline 1: Mean Predictor ===")
print(f"Training RMSE: {res_mean['train_rmse']:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {res_mean['oof_rmse']:.4f}")
print(f"OOF Pearson:   {res_mean['oof_pearson']:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Baseline 2: Audio Duration-Only Regressor

#### What are we doing?
Predicting grammar score using only the length of the recording.

#### Why are we doing it?
If test files systematically differ in length or if candidates who speak longer receive higher scores simply by talking more, duration alone could be a confounding feature. We test this directly.
"""))

cells.append(nbf.v4.new_code_cell(r"""X_dur = train_df[['duration']].values

def get_dur_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0, random_state=SEED))
    ])

res_dur = evaluate_cv(X_dur, y_true, folds, get_dur_model)
tracker.record("EXP-02-DUR", "Duration", "Ridge(alpha=1.0)", res_dur, "Diagnostic audio duration baseline")

print(f"=== Baseline 2: Audio Duration Regressor ===")
print(f"Training RMSE: {res_dur['train_rmse']:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {res_dur['oof_rmse']:.4f}")
print(f"OOF Pearson:   {res_dur['oof_pearson']:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### What did we learn?
Duration alone achieves an OOF Pearson of only **0.0786** and does not meaningfully improve RMSE over the mean baseline ($1.2344$ vs $1.2383$). This confirms that length of speaking time by itself does not predict grammar quality; candidates must be judged by acoustic and linguistic content.
"""))

# Acoustic Feature Models
cells.append(nbf.v4.new_markdown_cell("""### Handcrafted Acoustic Feature Models (75 Features)

#### What are we doing?
We load our 75 acoustic features (RMS energy, spectral envelope, MFCCs, silence/pause metrics, and F0 pitch stats) and evaluate three models under strict 5-fold CV:
1. **Ridge Regression**: Linear baseline with L2 regularization.
2. **Random Forest**: Non-linear bagging tree ensemble.
3. **LightGBM**: Gradient-boosted decision trees.

#### Why are we doing it?
Comparing linear vs non-linear algorithms shows whether the relationship between speech delivery and grammar ratings is predominantly linear or involves complex non-linear combinations (such as interactions between pauses and vocal pitch).
"""))

cells.append(nbf.v4.new_code_cell(r"""# Load pre-extracted acoustic features
audio_train_path = os.path.join(ARTIFACTS_DIR, "audio_features_train.parquet")
audio_test_path = os.path.join(ARTIFACTS_DIR, "audio_features_test.parquet")

df_audio_train = pd.read_parquet(audio_train_path)
df_audio_test = pd.read_parquet(audio_test_path)

feature_cols = [c for c in df_audio_train.columns if c not in ['filename', 'label', 'audio_path']]
X_audio = df_audio_train[feature_cols].copy()

print(f"Loaded {len(feature_cols)} acoustic features for {len(X_audio)} training samples.")

# Model 2C: Acoustic Ridge Regression
def get_audio_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=50.0, random_state=SEED))
    ])

res_ridge = evaluate_cv(X_audio, y_true, folds, get_audio_ridge)
tracker.record("EXP-03-AUDIO-RIDGE", "Handcrafted Acoustic (75)", "Ridge(alpha=50)", res_ridge, "Linear L2 regularized acoustic model")

# Model 2D: Acoustic Random Forest
def get_audio_rf():
    return RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )

res_rf = evaluate_cv(X_audio, y_true, folds, get_audio_rf)
tracker.record("EXP-04-AUDIO-RF", "Handcrafted Acoustic (75)", "RandomForest(depth=6)", res_rf, "Non-linear tree ensemble on acoustics")

# Model 2E: Acoustic LightGBM
def get_audio_lgbm():
    return LGBMRegressor(
        n_estimators=60, learning_rate=0.03, max_depth=3, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1, n_jobs=-1
    )

res_lgbm = evaluate_cv(X_audio, y_true, folds, get_audio_lgbm)
tracker.record("EXP-05-AUDIO-LGBM", "Handcrafted Acoustic (75)", "LightGBM", res_lgbm, "Gradient boosted trees on acoustics")

print(f"Acoustic Ridge:         OOF RMSE = {res_ridge['oof_rmse']:.4f}, OOF Pearson = {res_ridge['oof_pearson']:.4f}, Train RMSE = {res_ridge['train_rmse']:.4f}")
print(f"Acoustic Random Forest: OOF RMSE = {res_rf['oof_rmse']:.4f}, OOF Pearson = {res_rf['oof_pearson']:.4f}, Train RMSE = {res_rf['train_rmse']:.4f}")
print(f"Acoustic LightGBM:      OOF RMSE = {res_lgbm['oof_rmse']:.4f}, OOF Pearson = {res_lgbm['oof_pearson']:.4f}, Train RMSE = {res_lgbm['train_rmse']:.4f}")
"""))

cells.append(nbf.v4.new_code_cell(r"""# Visualizations 5 & 6: Predictions, Residuals & Feature Importances
best_oof = res_rf['oof_preds']
residuals = best_oof - y_true

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

# True vs Predicted
sns.regplot(x=y_true, y=best_oof, ax=axes[0],
            scatter_kws={'alpha': 0.5, 'color': '#2980b9'},
            line_kws={'color': '#e74c3c', 'linewidth': 2})
axes[0].plot([0, 5], [0, 5], '--', color='#7f8c8d', linewidth=1.5, label='Ideal 1:1')
axes[0].set_title(f"Acoustic RF: True vs Predicted Score (r = {res_rf['oof_pearson']:.4f})", fontsize=11, fontweight='bold')
axes[0].set_xlabel("True Grammar Score", fontsize=10)
axes[0].set_ylabel("Out-of-Fold Predicted Score", fontsize=10)
axes[0].set_xlim(-0.2, 5.2)
axes[0].set_ylim(-0.2, 5.2)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Residual Distribution
sns.histplot(residuals, kde=True, ax=axes[1], color='#8e44ad', bins=25, alpha=0.6)
axes[1].axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5)
axes[1].set_title(f"Residual Error Distribution (RMSE = {res_rf['oof_rmse']:.4f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Residual Error (Predicted - True)", fontsize=10)
axes[1].set_ylabel("Count", fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Top Feature Importances & Directionality
full_rf = get_audio_rf()
full_rf.fit(X_audio, y_true)
importances = pd.Series(full_rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
correlations = pd.Series([stats.pearsonr(X_audio[feat], y_true)[0] for feat in importances.index], index=importances.index)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=100)

# Feature Importance
colors = ['#16a085' if any(w in feat for w in ['pause', 'silence', 'speech']) else '#2980b9' for feat in importances.index]
importances.sort_values().plot(kind='barh', ax=axes[0], color=colors, edgecolor='#2c3e50', alpha=0.85)
axes[0].set_title("Top 12 Informative Acoustic Features (Random Forest)", fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel("Feature Importance (Gini Reduction)", fontsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# Directional Correlation (Pearson r)
corr_colors = ['#27ae60' if r > 0 else '#c0392b' for r in correlations[importances.sort_values().index]]
correlations[importances.sort_values().index].plot(kind='barh', ax=axes[1], color=corr_colors, edgecolor='#2c3e50', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title("Feature Directionality: Pearson Correlation with Score", fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel("Pearson Correlation (r)", fontsize=10)
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Feature Importance Interpretation & What We Learned
* **Feature importance tells us which variables contributed to prediction, but not whether higher values increase or decrease the predicted grammar score.** We therefore computed directional Pearson correlations separately:
  - `silence_ratio` ($r = -0.42$) and `num_pauses` ($r = -0.38$) exhibit strong negative correlations with grammar score: candidates who pause frequently receive lower scores.
  - `active_speech_duration` ($r = +0.47$) exhibits a positive correlation: sustained speech delivery without excessive hesitation tracks higher scores.
* **Random Forest produced the strongest acoustic baseline** ($\text{OOF Pearson} = 0.7591$, $\text{OOF RMSE} = 0.8078$), beating Ridge ($r = 0.7491$) and LightGBM ($r = 0.7297$).
* **Motivation for Phase 3**: Acoustic delivery correlates with proficiency, but cannot directly inspect syntax or sentence construction.
"""))

# ==============================================================================
# SECTION 7: PHASE 3 - SPEECH-TO-TEXT & LINGUISTIC FEATURES
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Phase 3 - Speech-to-Text & Linguistic Grammar Features

```text
Audio (.wav)
    ↓
Whisper ASR
    ↓
Transcript Text
    ↓
Linguistic / Grammar Features
    ↓
Regression Models (Ridge, RF, LightGBM)
    ↓
Predicted Grammar Score
```

### 7.1 Whisper Model Selection & Local ASR
#### What are we doing?
We use OpenAI's Whisper ASR to convert all spoken responses into text transcripts, caching them locally in `artifacts/transcripts/`.

#### Why Whisper base.en?
* `whisper-base.en` (139 MB) transcribes a 55-second audio sample in ~1.0-1.5 seconds on CPU with high accuracy for conversational English.
* `whisper-small.en` is 3.3× larger and takes ~4× longer per file, but yields almost identical transcripts on standard telephony speech.
* We select `whisper-base.en` because it balances high transcription quality with low computational cost.
"""))

cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# PIPELINE STEP: Speech-to-Text Transcription via Whisper ASR
# ==============================================================================
# NOTE: This transcription process was executed across all 769 train and 216 test
# audio files using 4 parallel Whisper workers (~1.0s per file), and results are
# saved in `artifacts/transcripts/`.
# ==============================================================================

# Load cached Whisper transcripts
train_transcripts_path = os.path.join(ARTIFACTS_DIR, "transcripts", "train_transcripts.json")
test_transcripts_path = os.path.join(ARTIFACTS_DIR, "transcripts", "test_transcripts.json")

with open(train_transcripts_path, 'r') as f:
    train_transcripts = json.load(f)
with open(test_transcripts_path, 'r') as f:
    test_transcripts = json.load(f)

print(f"Loaded {len(train_transcripts)} train transcripts and {len(test_transcripts)} test transcripts.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.2 Transcript Quality Checks

#### What are we doing?
We inspect transcription completeness, word counts, empty transcripts, and ASR hallucination loops.

#### Why are we doing it?
Speech recognition systems can occasionally produce empty outputs on faint audio or get stuck in repetition loops on background noise. Verifying transcript quality ensures clean NLP inputs.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Transcript Audit
train_word_counts = []
train_char_counts = []
empty_count = 0
short_count = 0

for fn, data in train_transcripts.items():
    text = data.get('text', '')
    words = text.split()
    train_word_counts.append(len(words))
    train_char_counts.append(len(text))
    if len(words) == 0:
        empty_count += 1
    elif len(words) < 5:
        short_count += 1

print(f"=== Transcript Quality Audit ===")
print(f"Total transcripts audited: {len(train_transcripts)}")
print(f"Empty transcripts:         {empty_count} ({empty_count/len(train_transcripts)*100:.1f}%)")
print(f"Very short (< 5 words):    {short_count} ({short_count/len(train_transcripts)*100:.1f}%)")
print(f"Word Count: Min={min(train_word_counts)}, Max={max(train_word_counts)}, Mean={np.mean(train_word_counts):.1f}, Median={np.median(train_word_counts):.1f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.3 Linguistic Feature Engineering & Models

#### What are we doing?
We extract 23 handcrafted linguistic features from the transcripts:
1. **Fluency & Volume**: Total words, character count, words per second, words per minute.
2. **Lexical Richness**: Unique word count, Type-Token Ratio (TTR), average word length.
3. **Disfluencies**: Filler words count ('um', 'uh', 'like'), repeated word count.
4. **Syntactic Complexity**: Sentence count, average sentence length, sentence length variance, fragments, commas, question marks.
5. **Part-of-Speech Proportions**: Noun, verb, adjective, adverb, pronoun, and preposition ratios.
6. **Grammar Error Proxies**: Heuristic subject-verb disagreement count.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Load pre-extracted linguistic features
text_train_path = os.path.join(ARTIFACTS_DIR, "text_features_train.parquet")
text_test_path = os.path.join(ARTIFACTS_DIR, "text_features_test.parquet")

df_text_train = pd.read_parquet(text_train_path)
df_text_test = pd.read_parquet(text_test_path)

text_feature_cols = [c for c in df_text_train.columns if c not in ['filename', 'label', 'audio_path']]
X_text = df_text_train[text_feature_cols].copy()

print(f"Loaded {len(text_feature_cols)} linguistic features for {len(X_text)} training samples.")

# Model 3D: Text Ridge
def get_text_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0, random_state=SEED))
    ])

res_text_ridge = evaluate_cv(X_text, y_true, folds, get_text_ridge)
tracker.record("EXP-06-TEXT-RIDGE", "Linguistic / Transcript Features", "Ridge(alpha=10)", res_text_ridge, "Linear linguistic model")

# Model 3E: Text Random Forest
def get_text_rf():
    return RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )

res_text_rf = evaluate_cv(X_text, y_true, folds, get_text_rf)
tracker.record("EXP-07-TEXT-RF", "Linguistic / Transcript Features", "RandomForest(depth=6)", res_text_rf, "Non-linear tree ensemble on text")

# Model 3F: Text LightGBM
def get_text_lgbm():
    return LGBMRegressor(
        n_estimators=60, learning_rate=0.03, max_depth=3, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1, n_jobs=-1
    )

res_text_lgbm = evaluate_cv(X_text, y_true, folds, get_text_lgbm)
tracker.record("EXP-08-TEXT-LGBM", "Linguistic / Transcript Features", "LightGBM", res_text_lgbm, "Gradient boosted trees on text")

print(f"Text Ridge:         OOF RMSE = {res_text_ridge['oof_rmse']:.4f}, OOF Pearson = {res_text_ridge['oof_pearson']:.4f}, Train RMSE = {res_text_ridge['train_rmse']:.4f}")
print(f"Text Random Forest: OOF RMSE = {res_text_rf['oof_rmse']:.4f}, OOF Pearson = {res_text_rf['oof_pearson']:.4f}, Train RMSE = {res_text_rf['train_rmse']:.4f}")
print(f"Text LightGBM:      OOF RMSE = {res_text_lgbm['oof_rmse']:.4f}, OOF Pearson = {res_text_lgbm['oof_pearson']:.4f}, Train RMSE = {res_text_lgbm['train_rmse']:.4f}")
"""))

# ==============================================================================
# SECTION 8: COMBINED ACOUSTIC + TEXT EXPERIMENT
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Combining Handcrafted Acoustic + Linguistic Features (Phase 3 Benchmark)

### What are we doing?
We concatenate our 75 acoustic features with our 23 linguistic features and evaluate a combined Random Forest regressor on the exact same 5-fold cross-validation split.

### Why are we doing it?
We want to establish whether combining acoustic and text features improves over acoustic alone:
* Acoustic-only RF: $\text{OOF Pearson} = 0.7591$, $\text{OOF RMSE} = 0.8078$
* Text-only RF: $\text{OOF Pearson} = 0.5778$, $\text{OOF RMSE} = 1.0108$
* Combined Acoustic + Text RF: Evaluated below.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Combined Feature Matrix (75 Acoustic + 23 Linguistic = 98 Features)
X_combined = pd.concat([X_audio, X_text], axis=1)

def get_comb_rf():
    return RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )

res_comb_rf = evaluate_cv(X_combined, y_true, folds, get_comb_rf)
tracker.record("EXP-09-COMBINED-RF", "Acoustic (75) + Linguistic (23)", "RandomForest(depth=6)", res_comb_rf, "Multi-modal acoustic + text ensemble")

print(f"=== Model 3G: Combined Acoustic + Text Random Forest ===")
print(f"Training RMSE: {res_comb_rf['train_rmse']:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {res_comb_rf['oof_rmse']:.4f}")
print(f"OOF Pearson:   {res_comb_rf['oof_pearson']:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Phase 3 Benchmark to Beat
```text
Acoustic + Linguistic Features + Random Forest
OOF RMSE:      0.7651
OOF Pearson:   0.7866
Training RMSE: 0.5397
```
Combining transcript features with acoustic features yielded an immediate improvement of $+0.0275$ in Pearson correlation and a reduction of $0.0427$ in RMSE. This proves that acoustics and transcripts capture complementary signals.
"""))

# ==============================================================================
# SECTION 9: PHASE 4 - TEXT REPRESENTATIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Phase 4 - Text Representations: TF-IDF & Pretrained Sentence Embeddings

> “Our linguistic features describe specific measurable properties of the transcript. We now test pretrained text representations to see whether a model can capture broader linguistic patterns without manually engineering every feature.”
"""))

cells.append(nbf.v4.new_code_cell(r"""# Extract raw transcripts in order of train_df
raw_train_texts = np.array([train_transcripts.get(fn, {}).get('text', '') for fn in train_df['filename']])

# Word & Character TF-IDF Vectorizers
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=300, min_df=3, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=300, min_df=5, sublinear_tf=True)
tfidf_union = FeatureUnion([('word', word_vec), ('char', char_vec)])

# Leak-Free 5-Fold Cross Validation for TF-IDF Ridge
oof_tfidf = np.zeros(len(y_true), dtype=float)
fold_train_rmses_tfidf = []
fold_val_rmses_tfidf = []
fold_val_pearsons_tfidf = []

for fold in range(5):
    tr_mask = (folds != fold)
    va_mask = (folds == fold)
    X_tr_tfidf = tfidf_union.fit_transform(raw_train_texts[tr_mask])
    X_va_tfidf = tfidf_union.transform(raw_train_texts[va_mask])
    clf = Ridge(alpha=10.0, random_state=SEED).fit(X_tr_tfidf, y_true[tr_mask])
    oof_tfidf[va_mask] = np.clip(clf.predict(X_va_tfidf), 0.0, 5.0)
    fold_train_rmses_tfidf.append(root_mean_squared_error(y_true[tr_mask], clf.predict(X_tr_tfidf)))
    fold_val_rmses_tfidf.append(root_mean_squared_error(y_true[va_mask], oof_tfidf[va_mask]))
    fold_val_pearsons_tfidf.append(safe_pearsonr(y_true[va_mask], oof_tfidf[va_mask]))

X_all_tfidf = tfidf_union.fit_transform(raw_train_texts)
clf_full_tfidf = Ridge(alpha=10.0, random_state=SEED).fit(X_all_tfidf, y_true)
train_rmse_tfidf = float(root_mean_squared_error(y_true, clf_full_tfidf.predict(X_all_tfidf)))

res_tfidf = {
    "cv_rmse_mean": float(np.mean(fold_val_rmses_tfidf)),
    "cv_rmse_std": float(np.std(fold_val_rmses_tfidf)),
    "cv_pearson_mean": float(np.mean(fold_val_pearsons_tfidf)),
    "cv_pearson_std": float(np.std(fold_val_pearsons_tfidf)),
    "oof_rmse": float(root_mean_squared_error(y_true, oof_tfidf)),
    "oof_pearson": float(safe_pearsonr(y_true, oof_tfidf)),
    "train_rmse": train_rmse_tfidf,
    "oof_preds": oof_tfidf
}
tracker.record("EXP-10-TFIDF-COMB", "Word + Char TF-IDF (600)", "Ridge(alpha=10)", res_tfidf, "TF-IDF n-gram text baseline")

# Load cached pretrained MiniLM embeddings
emb_train_path = os.path.join(ARTIFACTS_DIR, "text_embeddings", "train_embeddings_minilm.parquet")
df_emb_train = pd.read_parquet(emb_train_path)
emb_cols = [c for c in df_emb_train.columns if c.startswith('emb_')]
X_emb = df_emb_train[emb_cols].copy()

def get_emb_ridge():
    return Ridge(alpha=1.0, random_state=SEED)

res_emb_ridge = evaluate_cv(X_emb, y_true, folds, get_emb_ridge)
tracker.record("EXP-11-EMB-RIDGE", "Dense MiniLM Embeddings (384)", "Ridge(alpha=1.0)", res_emb_ridge, "Frozen sentence transformer embeddings")

print(f"TF-IDF Ridge:   OOF RMSE = {res_tfidf['oof_rmse']:.4f}, Pearson = {res_tfidf['oof_pearson']:.4f}, Train RMSE = {res_tfidf['train_rmse']:.4f}")
print(f"MiniLM Ridge:   OOF RMSE = {res_emb_ridge['oof_rmse']:.4f}, Pearson = {res_emb_ridge['oof_pearson']:.4f}, Train RMSE = {res_emb_ridge['train_rmse']:.4f}")
"""))

# ==============================================================================
# SECTION 10: PHASE 5 - ROBUSTNESS CHECKS & PRETRAINED SPEECH EMBEDDINGS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 10. Phase 5 - Robustness Checks & Pretrained Speech Embeddings

### 10.1 Methodological Check: Blend Optimism Disclosure
> “The blend weights were optimized using OOF predictions. Because the same OOF predictions were then used to evaluate the blend, this estimate can be slightly optimistic. We therefore perform additional robustness checks before treating the blend as our final benchmark.”

### 10.2 Multi-Seed Robustness Check (Seeds: 42, 123, 2026)
#### What are we doing?
We repeat the 5-fold stratified cross-validation for our two shortlisted models across three independent random seeds: `42`, `123`, and `2026`.

#### Why are we doing it?
> “Our dataset is small, so one particular train/validation split could make a model look slightly better or worse by chance. Testing a few random seeds helps us see whether the improvement is stable.”
"""))

cells.append(nbf.v4.new_code_cell(r"""# Multi-Seed Cross-Validation Check
seeds = [42, 123, 2026]
results_rf_seeds = []
results_ridge_seeds = []
results_blend_seeds = []

X_comb_al = pd.concat([X_audio, X_text], axis=1)

for s in seeds:
    skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    folds_s = np.zeros(len(y_true), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf_s.split(train_df, strat_bins)):
        folds_s[val_idx] = fold_idx
        
    res_rf_s = evaluate_cv(X_comb_al, y_true, folds_s, get_comb_rf)
    res_rg_s = evaluate_cv(X_emb, y_true, folds_s, get_emb_ridge)
    
    # Simple fixed 75/25 blend on this seed's OOF predictions
    b_pred_s = 0.75 * res_rf_s['oof_preds'] + 0.25 * res_rg_s['oof_preds']
    b_rmse_s = float(root_mean_squared_error(y_true, b_pred_s))
    b_r_s = float(safe_pearsonr(y_true, b_pred_s))
    
    results_rf_seeds.append({'Seed': s, 'OOF RMSE': res_rf_s['oof_rmse'], 'OOF Pearson': res_rf_s['oof_pearson']})
    results_ridge_seeds.append({'Seed': s, 'OOF RMSE': res_rg_s['oof_rmse'], 'OOF Pearson': res_rg_s['oof_pearson']})
    results_blend_seeds.append({'Seed': s, 'OOF RMSE': b_rmse_s, 'OOF Pearson': b_r_s})

df_rf_seeds = pd.DataFrame(results_rf_seeds)
df_rg_seeds = pd.DataFrame(results_ridge_seeds)
df_bl_seeds = pd.DataFrame(results_blend_seeds)

print("=== Multi-Seed Robustness Summary ===")
print(f"Acoustic+Ling RF:  Mean RMSE = {df_rf_seeds['OOF RMSE'].mean():.4f} +/- {df_rf_seeds['OOF RMSE'].std():.4f} | Mean Pearson = {df_rf_seeds['OOF Pearson'].mean():.4f} +/- {df_rf_seeds['OOF Pearson'].std():.4f}")
print(f"MiniLM Ridge:      Mean RMSE = {df_rg_seeds['OOF RMSE'].mean():.4f} +/- {df_rg_seeds['OOF RMSE'].std():.4f} | Mean Pearson = {df_rg_seeds['OOF Pearson'].mean():.4f} +/- {df_rg_seeds['OOF Pearson'].std():.4f}")
print(f"Fixed 75/25 Blend: Mean RMSE = {df_bl_seeds['OOF RMSE'].mean():.4f} +/- {df_bl_seeds['OOF RMSE'].std():.4f} | Mean Pearson = {df_bl_seeds['OOF Pearson'].mean():.4f} +/- {df_bl_seeds['OOF Pearson'].std():.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.3 Simple Non-Optimized Blending Check

#### What are we doing?
We evaluate fixed, predetermined blending ratios (100/0, 75/25, 70/30, 50/50) without optimizing any weights on OOF predictions.

#### Why are we doing it?
We want an interview-defensible result:
> “The two models were complementary, so a simple 75/25 blend improved validation performance.”
That is easier to defend than:
> “We optimized 137 possible weights and selected 0.737.”
"""))

cells.append(nbf.v4.new_code_cell(r"""# Predetermined Blends on Seed 42
oof_rf_42 = res_comb_rf['oof_preds']
oof_rg_42 = res_emb_ridge['oof_preds']

predet_results = []
for w_rf, w_rg in [(1.0, 0.0), (0.75, 0.25), (0.70, 0.30), (0.50, 0.50)]:
    b_pred = w_rf * oof_rf_42 + w_rg * oof_rg_42
    b_rmse = root_mean_squared_error(y_true, b_pred)
    b_r = safe_pearsonr(y_true, b_pred)
    full_train = np.clip(w_rf * res_comb_rf['fitted_models'][0].predict(X_comb_al) + w_rg * res_emb_ridge['fitted_models'][0].predict(X_emb), 0.0, 5.0)
    tr_rmse = root_mean_squared_error(y_true, full_train)
    predet_results.append({
        'Blend Specification': f"{int(w_rf*100)}% RF + {int(w_rg*100)}% MiniLM",
        'OOF RMSE': round(b_rmse, 4),
        'OOF Pearson': round(b_r, 4),
        'Train RMSE': round(tr_rmse, 4)
    })

df_predet = pd.DataFrame(predet_results)
display(df_predet)

tracker.record("EXP-17-SIMPLE-BLEND-75", "75% RF + 25% MiniLM", "Fixed Predetermined Blend", {
    "cv_rmse_mean": df_predet.iloc[1]['OOF RMSE'], "cv_rmse_std": 0.0,
    "cv_pearson_mean": df_predet.iloc[1]['OOF Pearson'], "cv_pearson_std": 0.0,
    "oof_rmse": df_predet.iloc[1]['OOF RMSE'], "oof_pearson": df_predet.iloc[1]['OOF Pearson'],
    "train_rmse": df_predet.iloc[1]['Train RMSE'], "oof_preds": 0.75 * oof_rf_42 + 0.25 * oof_rg_42
}, "Robust non-optimized 3:1 blend")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.4 Pretrained Speech Embeddings (`facebook/wav2vec2-base`)

#### What are we doing?
We extract dense speech representations using the frozen speech transformer `facebook/wav2vec2-base` (94.4M parameters), pooling the frame representations across time via mean and standard deviation into a 1536-dimensional embedding, and train regularized linear models (Ridge and ElasticNet).

#### Why are we doing it?
> “Our handcrafted acoustic features summarize things such as pauses, energy, pitch and spectral characteristics. A pretrained speech encoder may capture richer patterns in speech that are difficult to describe manually, so we test a frozen speech representation as another source of information.”

#### Computational Trade-off & Practicality
* **Model Size**: 94.4M parameters (~360 MB float32 weights).
* **Extraction Time**: Processed on Apple Silicon GPU (`mps`) at ~0.76s per 50-second audio sample.
* **Dimensionality**: Frame representation (768) pooled with mean (768) and standard deviation (768) $\to$ 1536 features per audio recording.
* **Regularization Choice**:
  > “The information in dense embeddings is distributed across many dimensions, so a regularized linear model is a natural first choice. It also reduces the risk of fitting noise in a small dataset.”
"""))

cells.append(nbf.v4.new_code_cell(r"""# Load pre-extracted wav2vec2-base speech embeddings
speech_train_path = os.path.join(ARTIFACTS_DIR, "audio_embeddings", "train_speech_embeddings.parquet")
speech_test_path = os.path.join(ARTIFACTS_DIR, "audio_embeddings", "test_speech_embeddings.parquet")

df_speech_train = pd.read_parquet(speech_train_path)
df_speech_test = pd.read_parquet(speech_test_path)

speech_cols = [c for c in df_speech_train.columns if c.startswith('emb_speech_')]
X_speech = df_speech_train[speech_cols].copy()

print(f"Speech Embeddings Loaded: Shape = {X_speech.shape} (1536 dimensions per recording)")

# Model 5A: Speech Embedding Ridge (alpha=100.0)
def get_speech_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=100.0, random_state=SEED))
    ])

res_speech_ridge = evaluate_cv(X_speech, y_true, folds, get_speech_ridge)
tracker.record("EXP-18-SPEECH-RIDGE", "Dense Speech Embeddings (1536)", "Ridge(alpha=100)", res_speech_ridge, "Frozen wav2vec2 speech embeddings")

# Model 5B: Speech Embedding ElasticNet
def get_speech_elasticnet():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(alpha=0.05, l1_ratio=0.1, random_state=SEED, max_iter=2000))
    ])

res_speech_elasticnet = evaluate_cv(X_speech, y_true, folds, get_speech_elasticnet)
tracker.record("EXP-19-SPEECH-ELASTICNET", "Dense Speech Embeddings (1536)", "ElasticNet(alpha=0.05, l1=0.1)", res_speech_elasticnet, "Sparse regularized speech regression")

print(f"=== Model 5A: Speech Embedding Ridge ===")
print(f"Training RMSE: {res_speech_ridge['train_rmse']:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {res_speech_ridge['oof_rmse']:.4f}")
print(f"OOF Pearson:   {res_speech_ridge['oof_pearson']:.4f}")
print()
print(f"=== Model 5B: Speech Embedding ElasticNet ===")
print(f"Training RMSE: {res_speech_elasticnet['train_rmse']:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {res_speech_elasticnet['oof_rmse']:.4f}")
print(f"OOF Pearson:   {res_speech_elasticnet['oof_pearson']:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.5 Feature Combinations with Pretrained Speech Embeddings

#### What are we doing?
We evaluate regularized combinations:
1. **Acoustic + Speech**: Handcrafted acoustic features (75) + Speech embeddings (1536).
2. **Linguistic + Speech**: Handcrafted linguistic features (23) + Speech embeddings (1536).
3. **Acoustic + Linguistic + Speech**: Handcrafted (98) + Speech embeddings (1536).
4. **All Modalities**: Acoustic (75) + Linguistic (23) + Text MiniLM (384) + Speech wav2vec2 (1536).
"""))

cells.append(nbf.v4.new_code_cell(r"""# Feature Matrices
X_aud_sp = np.hstack([X_audio.values, X_speech.values])
X_ling_sp = np.hstack([X_text.values, X_speech.values])
X_aud_ling_sp = np.hstack([X_audio.values, X_text.values, X_speech.values])
X_all_modalities = np.hstack([X_audio.values, X_text.values, X_emb.values, X_speech.values])

def get_combo_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=100.0, random_state=SEED))
    ])

res_aud_sp = evaluate_cv(X_aud_sp, y_true, folds, get_combo_ridge)
tracker.record("EXP-20-AUDIO-SPEECH", "Acoustic (75) + Speech (1536)", "Ridge(alpha=100)", res_aud_sp, "Handcrafted + deep speech representations")

res_ling_sp = evaluate_cv(X_ling_sp, y_true, folds, get_combo_ridge)
tracker.record("EXP-21-LING-SPEECH", "Linguistic (23) + Speech (1536)", "Ridge(alpha=100)", res_ling_sp, "Handcrafted text + deep speech representations")

res_aud_ling_sp = evaluate_cv(X_aud_ling_sp, y_true, folds, get_combo_ridge)
tracker.record("EXP-22-AUD-LING-SPEECH", "Acoustic (75) + Ling (23) + Speech (1536)", "Ridge(alpha=100)", res_aud_ling_sp, "Multi-modal acoustic, text & speech representations")

res_all_mod = evaluate_cv(X_all_modalities, y_true, folds, get_combo_ridge)
tracker.record("EXP-23-ALL-MODALITIES", "Acoustic + Ling + TextEmb + SpeechEmb (2018)", "Ridge(alpha=100)", res_all_mod, "Complete unified multi-modal feature set")

print(f"Acoustic + Speech Ridge:          OOF RMSE = {res_aud_sp['oof_rmse']:.4f}, Pearson = {res_aud_sp['oof_pearson']:.4f}, Train RMSE = {res_aud_sp['train_rmse']:.4f}")
print(f"Linguistic + Speech Ridge:        OOF RMSE = {res_ling_sp['oof_rmse']:.4f}, Pearson = {res_ling_sp['oof_pearson']:.4f}, Train RMSE = {res_ling_sp['train_rmse']:.4f}")
print(f"Acoustic + Ling + Speech Ridge:   OOF RMSE = {res_aud_ling_sp['oof_rmse']:.4f}, Pearson = {res_aud_ling_sp['oof_pearson']:.4f}, Train RMSE = {res_aud_ling_sp['train_rmse']:.4f}")
print(f"All Modalities Ridge (Unified):   OOF RMSE = {res_all_mod['oof_rmse']:.4f}, Pearson = {res_all_mod['oof_pearson']:.4f}, Train RMSE = {res_all_mod['train_rmse']:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.6 Model Complementarity & Multi-Modal Blending

#### What are we doing?
We inspect prediction and residual correlations between:
1. `Acoustic + Linguistic RF` (EXP-09)
2. `MiniLM Text Ridge` (EXP-11)
3. `wav2vec2 Speech Ridge` (EXP-18)
and evaluate simple, predetermined blends.

#### Why are we doing it?
> “If two models make different mistakes, combining them may improve generalization.”
"""))

cells.append(nbf.v4.new_code_cell(r"""oof_rf_m = res_comb_rf['oof_preds']
oof_txt_m = res_emb_ridge['oof_preds']
oof_sp_m = res_speech_ridge['oof_preds']

# Residuals
res_rf_m = oof_rf_m - y_true
res_txt_m = oof_txt_m - y_true
res_sp_m = oof_sp_m - y_true

print("=== Model Independence Analysis ===")
print(f"Pred corr (RF, Speech Ridge):     {stats.pearsonr(oof_rf_m, oof_sp_m)[0]:.4f}")
print(f"Pred corr (MiniLM, Speech Ridge): {stats.pearsonr(oof_txt_m, oof_sp_m)[0]:.4f}")
print(f"Residual corr (RF, Speech Ridge): {stats.pearsonr(res_rf_m, res_sp_m)[0]:.4f}")
print(f"Residual corr (MiniLM, Speech):   {stats.pearsonr(res_txt_m, res_sp_m)[0]:.4f}")

# Simple Predetermined Tri-Modal Blend (50% Speech + 30% RF + 20% MiniLM)
oof_tri_blend = 0.50 * oof_sp_m + 0.30 * oof_rf_m + 0.20 * oof_txt_m
tri_rmse = float(root_mean_squared_error(y_true, oof_tri_blend))
tri_r = float(safe_pearsonr(y_true, oof_tri_blend))

# Full training fit for compulsory train RMSE
full_train_tri = np.clip(
    0.50 * res_speech_ridge['fitted_models'][0].predict(X_speech) +
    0.30 * res_comb_rf['fitted_models'][0].predict(X_comb_al) +
    0.20 * res_emb_ridge['fitted_models'][0].predict(X_emb),
    0.0, 5.0
)
tri_train_rmse = float(root_mean_squared_error(y_true, full_train_tri))

tracker.record("EXP-24-TRI-MODAL-BLEND", "50% Speech + 30% RF + 20% MiniLM", "Predetermined Tri-Modal Blend", {
    "cv_rmse_mean": tri_rmse, "cv_rmse_std": 0.0,
    "cv_pearson_mean": tri_r, "cv_pearson_std": 0.0,
    "oof_rmse": tri_rmse, "oof_pearson": tri_r,
    "train_rmse": tri_train_rmse, "oof_preds": oof_tri_blend
}, "Complementary tri-modal ensemble")

print()
print(f"=== Model 5C: Predetermined Tri-Modal Blend (50% Speech + 30% RF + 20% MiniLM) ===")
print(f"Training RMSE: {tri_train_rmse:.4f}  (COMPULSORY)")
print(f"OOF RMSE:      {tri_rmse:.4f}")
print(f"OOF Pearson:   {tri_r:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.7 Error Analysis Across Score Tiers

#### What are we doing?
We compare performance across score tiers:
* **Low**: $y \le 2.5$ ($N = 222$)
* **Mid**: $2.5 < y \le 3.5$ ($N = 238$)
* **High**: $y > 3.5$ ($N = 309$)

> *Note*: Low scores are sparse in this dataset (only 4 samples $\le 1.5$), so metrics on tiny subgroups should be interpreted as diagnostic rather than conclusive.
"""))

cells.append(nbf.v4.new_code_cell(r"""tier_masks = {
    'Low (<= 2.5)': y_true <= 2.5,
    'Mid (2.5 < y <= 3.5)': (y_true > 2.5) & (y_true <= 3.5),
    'High (> 3.5)': y_true > 3.5
}

tier_models = [
    ('Acoustic+Ling RF', oof_rf_m),
    ('Speech Ridge', oof_sp_m),
    ('All-Modalities Ridge', res_all_mod['oof_preds']),
    ('Tri-Modal Blend', oof_tri_blend)
]

tier_data = []
for tier_name, mask in tier_masks.items():
    n_tier = int(np.sum(mask))
    for m_name, preds in tier_models:
        t_rmse = root_mean_squared_error(y_true[mask], preds[mask])
        t_r = safe_pearsonr(y_true[mask], preds[mask])
        tier_data.append({'Score Tier': tier_name, 'Count': n_tier, 'Model': m_name, 'Tier RMSE': round(t_rmse, 4), 'Tier Pearson': round(t_r, 4)})

df_tiers = pd.DataFrame(tier_data)
display(df_tiers.pivot(index='Score Tier', columns='Model', values=['Tier RMSE', 'Tier Pearson']))
"""))

cells.append(nbf.v4.new_code_cell(r"""# Visualizations 10, 11 & 12: Residual Distributions, True vs Predicted & Tier RMSE
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=100)

# 1. Residual KDE Comparison
sns.kdeplot(res_rf_m, ax=axes[0], label=f"Acoustic+Ling RF (RMSE: {res_comb_rf['oof_rmse']:.3f})", color='#2980b9', linewidth=1.8)
sns.kdeplot(res_sp_m, ax=axes[0], label=f"Speech Ridge (RMSE: {res_speech_ridge['oof_rmse']:.3f})", color='#e67e22', linewidth=1.8)
sns.kdeplot(oof_tri_blend - y_true, ax=axes[0], label=f"Tri-Modal Blend (RMSE: {tri_rmse:.3f})", color='#27ae60', linewidth=2.0)
axes[0].axvline(0, color='#e74c3c', linestyle='--', linewidth=1.2)
axes[0].set_title("Residual Distributions: RF vs Speech vs Blend", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Residual Error (Predicted - True)", fontsize=10)
axes[0].set_ylabel("Density", fontsize=10)
axes[0].legend(fontsize=8, frameon=True)
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. True vs Predicted Scatter for Tri-Modal Blend
sns.regplot(x=y_true, y=oof_tri_blend, ax=axes[1],
            scatter_kws={'alpha': 0.45, 'color': '#27ae60'},
            line_kws={'color': '#e74c3c', 'linewidth': 2})
axes[1].plot([0, 5], [0, 5], '--', color='#7f8c8d', linewidth=1.5, label='Ideal 1:1')
axes[1].set_title(f"Tri-Modal Blend: True vs Predicted (r = {tri_r:.4f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("True Grammar Score", fontsize=10)
axes[1].set_ylabel("Out-of-Fold Predicted Score", fontsize=10)
axes[1].set_xlim(-0.2, 5.2)
axes[1].set_ylim(-0.2, 5.2)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

# 3. Tier-wise RMSE Comparison Bar Chart
pivot_rmse = df_tiers.pivot(index='Score Tier', columns='Model', values='Tier RMSE')[['Acoustic+Ling RF', 'Speech Ridge', 'All-Modalities Ridge', 'Tri-Modal Blend']]
pivot_rmse.plot(kind='bar', ax=axes[2], color=['#2980b9', '#e67e22', '#8e44ad', '#27ae60'], edgecolor='#2c3e50', alpha=0.85)
axes[2].set_title("Tier RMSE Comparison", fontsize=11, fontweight='bold')
axes[2].set_ylabel("RMSE (Lower is Better)", fontsize=10)
axes[2].set_xlabel("")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=15, ha='right')
axes[2].legend(fontsize=8, frameon=True)
axes[2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10.8 Final Model Decision & Interview Defense

#### Decision Hierarchy:
1. **Robust CV / OOF Performance**:
   - **Tri-Modal Blend**: $\text{OOF Pearson} = \mathbf{0.8786}$, $\text{OOF RMSE} = \mathbf{0.6304}$.
   - **All-Modalities Ridge**: $\text{OOF Pearson} = \mathbf{0.8671}$, $\text{OOF RMSE} = \mathbf{0.6170}$.
2. **Stability Across Random Seeds**:
   - Both models demonstrated remarkable stability across seeds `42`, `123`, `2026`:
     - All-Modalities Ridge: $0.8675 \pm 0.0007$
     - Tri-Modal Blend: $0.8780 \pm 0.0010$
3. **Simplicity & Interpretability**:
   - The Tri-Modal Blend ($50\%$ Speech Ridge + $30\%$ Acoustic-Linguistic RF + $20\%$ Text MiniLM Ridge) is intuitive: it blends three complementary perspectives of spoken language:
     - **Non-linear acoustic pauses and cadence** (Random Forest)
     - **Semantic and syntactical transcript quality** (MiniLM Ridge)
     - **Continuous acoustic timbre, phonetics, and pronunciation** (wav2vec2 Ridge)
4. **Computational Practicality**:
   - All pretrained encoders (`all-MiniLM-L6-v2`, `wav2vec2-base`) remain completely frozen.
   - Downstream models are linear Ridge regressors and shallow Random Forests that fit in seconds without GPU clusters or end-to-end backpropagation risk.
"""))

# ==============================================================================
# SECTION 11: COMPLETE EXPERIMENT TABLE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 11. Final Comprehensive Experiment Comparison Table

*Note*: The Phase 4 optimized blend is marked with an asterisk (`*`) to denote potential optimistic bias from weight search on OOF predictions, resolved by our Phase 5 multi-seed and fixed-weight checks.
"""))

cells.append(nbf.v4.new_code_cell(r"""summary_table = tracker.df[['experiment_id', 'features', 'model', 'oof_rmse', 'oof_pearson', 'train_rmse', 'notes']].copy()
# Format asterisk on optimized blend
summary_table['oof_pearson'] = summary_table.apply(
    lambda r: f"{r['oof_pearson']:.4f}*" if 'EXP-17-BLEND-RF-EMB' in str(r['experiment_id']) else f"{r['oof_pearson']:.4f}", axis=1
)
summary_table = summary_table.sort_values(by='oof_rmse', ascending=True)
display(summary_table)
"""))

# ==============================================================================
# SECTION 12: SUBMISSION FILE GENERATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 12. Submission Generation

We generate final test predictions for all 216 test audio files using our validated models and format the CSV strictly according to competition requirements.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Retrain full components on all 769 training samples
full_rf = get_comb_rf().fit(X_comb_al, y_true)
full_txt = get_emb_ridge().fit(X_emb, y_true)
full_sp = get_speech_ridge().fit(X_speech, y_true)

# Load test features
df_text_test_raw = pd.read_parquet(text_test_path)
df_emb_test_raw = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "text_embeddings", "test_embeddings_minilm.parquet"))
df_speech_test_raw = pd.read_parquet(speech_test_path)

X_audio_test = df_audio_test[feature_cols].copy()
X_text_test = df_text_test_raw[text_feature_cols].copy()
X_comb_test = pd.concat([X_audio_test, X_text_test], axis=1)

X_emb_test = df_emb_test_raw[emb_cols].copy()
X_speech_test = df_speech_test_raw[speech_cols].copy()

# Generate component test predictions
test_preds_rf = np.clip(full_rf.predict(X_comb_test), 0.0, 5.0)
test_preds_txt = np.clip(full_txt.predict(X_emb_test), 0.0, 5.0)
test_preds_sp = np.clip(full_sp.predict(X_speech_test), 0.0, 5.0)

# Final Tri-Modal Blend Test Predictions
final_test_preds = 0.50 * test_preds_sp + 0.30 * test_preds_rf + 0.20 * test_preds_txt

# Format and save submission
submission_path = os.path.join(ARTIFACTS_DIR, "submission_final_trimodal.csv")
sub_df = format_submission(final_test_preds, test_df, output_path=submission_path)
display(sub_df.head(10))
print(f"Final submission verified: {len(sub_df)} rows, mean predicted score = {sub_df['label'].mean():.2f}")
"""))

nb['cells'] = cells

notebook_path = "/Users/nandinikhandelwal/Desktop/Codes/shl/notebook/shl_grammar_scoring.ipynb"
with open(notebook_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}. Executing all cells from top to bottom...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open(notebook_path, "w") as f:
    nbf.write(nb, f)

print("Notebook successfully executed and saved with all cell outputs!")
