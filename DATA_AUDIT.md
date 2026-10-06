# DATA AUDIT — SHL Hiring Assessment 2026 (Grammar Scoring Engine)

**Date**: 2026-10-06  
**Auditor**: Antigravity Pair-Programming Agent  
**Dataset Path**: `data/Dataset_Final/`  
**Machine-Readable Summary**: [`artifacts/data_audit.json`](file:///Users/nandinikhandelwal/Desktop/Codes/shl/artifacts/data_audit.json)

---

## 1. Directory Structure & Files

The extracted dataset directory `data/Dataset_Final/` contains:
```text
Dataset_Final/
├── train.csv                (769 rows × 2 cols)
├── test.csv                 (216 rows × 2 cols)
├── sample_submission.csv    (204 rows × 2 cols)
├── train/                   (769 .wav audio files)
└── test/                    (216 .wav audio files)
```

Total files: **988 items** (985 `.wav` files across train/test + 3 `.csv` files).

---

## 2. Dataset Sizes & CSV Schemas

### `train.csv`
* **Rows**: 769 | **Columns**: 2 (`filename`, `label`)
* **Null Values**: 0
* **Format**:
  * `filename`: string (e.g. `audio_192.wav`)
  * `label`: float64 continuous/rubric grammar rating

### `test.csv`
* **Rows**: 216 | **Columns**: 2 (`filename`, `label`)
* **Null Values**: 0
* `label` column is filled with dummy value `-1`.
* Filenames strictly range from `audio_0.wav` to `audio_215.wav` matching exactly the 216 `.wav` files in `data/Dataset_Final/test/`.

### `sample_submission.csv` Discrepancy (CRITICAL FINDING)
* **Rows**: 204 (Columns: `filename`, `label`)
* Filenames in `sample_submission.csv` range up to `audio_1329.wav` (e.g., `audio_804.wav`, `audio_1028.wav`) and do **not** match the 216 evaluation files in `test.csv` / `test/`.
* **Action Required**: The actual evaluation target is the 216 audio files in `test.csv`. When generating predictions, predictions must be generated for all 216 test files in `test.csv`, and if Kaggle expects `test.csv` ordering, we maintain `test.csv` as the primary submission target while keeping a format compatible with `sample_submission.csv` columns.

---

## 3. Training Labels Analysis

* **Target Column**: `label`
* **Target Range**: Min = `0.0`, Max = `5.0`
* **Central Tendency & Dispersion**:
  * **Mean**: 3.3134
  * **Std**: 1.2390
  * **Median**: 3.0
  * **IQR**: 25th percentile = 2.5, 75th percentile = 4.0
* **Data Type / Discreteness**:
  * 10 distinct values: `[0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`
  * Labels follow a 0.5-step rubric. However, competition metric evaluates **Pearson Correlation** and **RMSE** with a continuous prediction expectation in `[0.0, 5.0]`.
* **Class Distribution & Severe Imbalance**:
  * `0.0`: 37 (4.81%)
  * `1.0`: 1 (0.13%) — *Extreme minority*
  * `1.5`: 3 (0.39%) — *Extreme minority*
  * `2.0`: 102 (13.26%)
  * `2.5`: 79 (10.27%)
  * `3.0`: 174 (22.63%) — *Mode*
  * `3.5`: 64 (8.32%)
  * `4.0`: 124 (16.12%)
  * `4.5`: 52 (6.76%)
  * `5.0`: 133 (17.29%)
  * The range `[1.0, 1.5]` has only 4 samples total in the training set! Scores are skewed toward moderate-to-high scores (3.0 – 5.0 accounts for >70% of data).

---

## 4. Audio Signal Properties

| Property | Training Set (769 files) | Test Set (216 files) |
| :--- | :--- | :--- |
| **Sampling Rate** | 16,000 Hz (uniform) | 16,000 Hz (uniform) |
| **Channels** | 1 (Mono) | 1 (Mono) |
| **Encoding / Subtype** | PCM 16-bit (`PCM_16`) | PCM 16-bit (`PCM_16`) |
| **Min Duration** | 20.06 s | 6.48 s |
| **Max Duration** | 61.04 s | 61.03 s |
| **Mean Duration** | 55.78 s | 48.76 s |
| **Median Duration** | 60.07 s | 45.06 s |
| **5th - 95th Percentile** | 43.21 s – 60.17 s | 41.41 s – 60.08 s |
| **File Sizes** | 642 KB – 1.95 MB (mean: 1.78 MB) | 207 KB – 1.95 MB (mean: 1.56 MB) |
| **Audio Quality Check** | High SNR, standard telephony/mic speech, minimal clipping (<3% samples reach -0.01 dBFS), 0 dead-silent files. | High SNR, slightly wider duration variance (min 6.48s). |

---

## 5. Metadata, Speaker Information & Leakage Checks

### A. Filename Collisions (CRITICAL GOTCHA)
* **212 filenames** appear in **both** `data/Dataset_Final/train/` and `data/Dataset_Final/test/` (e.g. `audio_128.wav`).
* **MD5 Verification**: Every colliding pair has **completely different MD5 hashes and different signal lengths**.
* **Implication**: Filenames are locally indexed inside each folder. Any code doing `os.path.basename` without carrying the folder path will cause severe data corruption. All file paths must be stored with full paths or explicit folder prefixes (`train/audio_X.wav` vs `test/audio_X.wav`).

### B. Duplicate / Train-Test Overlap
* Exact duplicate audio files in Train: **0**
* Exact duplicate audio files in Test: **0**
* Exact overlap between Train and Test: **0**

### C. Speaker & Session Metadata
* RIFF header inspection confirms files were encoded using FFmpeg (`Lavf58.29.100`) without embedded speaker, age, gender, or session tags.
* Filenames are synthetic numeric indices (`audio_N.wav`).
* Because speaker IDs are not directly available, GroupKFold cannot be constructed from metadata alone.

---

## 6. Recommended Validation Strategy

Given the small dataset size (769 samples), discrete rubric targets, and extreme class imbalance in lower scores (only 4 samples in `[1.0, 1.5]`):

1. **Stratified Continuous K-Fold (5 Folds)**:
   * Bin the target score into discrete stratification bins:
     * Bin 0: `[0.0]`
     * Bin 1: `[1.0, 1.5, 2.0]` (combines rare 1.0 and 1.5 with 2.0 so every fold gets representative lower scores)
     * Bin 2: `[2.5]`
     * Bin 3: `[3.0]`
     * Bin 4: `[3.5]`
     * Bin 5: `[4.0]`
     * Bin 6: `[4.5]`
     * Bin 7: `[5.0]`
   * Use a fixed seed (`seed=42`) with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   * For critical model comparisons, also verify with 2 seeds (or 10-fold) to guard against split variance.
2. **Evaluation Metrics Tracked in Every Fold**:
   * **RMSE** (primary regression error)
   * **Pearson Correlation** ($r$)
   * Mean and standard deviation across the 5 validation folds.
3. **Out-of-Fold (OOF) Prediction Caching**:
   * Out-of-fold predictions will be strictly accumulated for fair ensembling without data leakage.

---

## 7. Computational Environment & Constraints

* **Platform**: Apple Silicon (`arm64`), macOS 26.5.2.
* **Audio Length**: ~11.8 hours total audio across train (769 × ~55s) and test (216 × ~48s).
* **ASR & NLP Strategy**:
  * Use PyTorch with Apple Silicon MPS / CPU.
  * For Whisper ASR (Speech-to-Text), `whisper-base.en` or `whisper-small.en` provides rapid transcription (~0.5–1s per 60s file with batched/MPS inference).
  * Transcripts and extracted features will be cached in `artifacts/` to guarantee instant re-runs and zero redundant computation.

---

## 8. Proposed Highest-Value Modeling Path

Grammar quality in spontaneous spoken English is dominated by:
1. **Linguistic Fluency & Syntax (from Transcripts)**:
   * Word count, speech rate (words per minute), lexical diversity (Type-Token Ratio, vocd), clause density.
   * Disfluency markers (fillers: "um", "uh", false starts, repetitions).
   * Part-of-Speech distributions, sentence fragmentation, dependency parse tree depth.
   * Grammar error proxies (rule-based agreement checks, preposition mistakes, language model perplexity).
2. **Semantic & Text Representations**:
   * Pretrained sentence embeddings (e.g. `all-MiniLM-L6-v2` or `BGE-small`).
3. **Acoustic & Prosodic Features**:
   * Silence ratio, pause rate, speaking rate, pitch variability, RMS energy variance, spectral centroid/rolloff (which reflect confidence, hesitancy, and flow).
4. **Frozen Speech Embeddings (WavLM / HuBERT / Wav2Vec2)**:
   * Mean + Std pooling over frozen representations.
5. **Model Strategy**:
   * **Phase 2**: Baseline Mean + Handcrafted Audio Features (Ridge, LightGBM/CatBoost).
   * **Phase 3**: Transcripts + Linguistic Grammar Features (Ridge, LightGBM).
   * **Phase 4**: Text Embeddings + Linguistic Features.
   * **Phase 5**: Speech Embeddings + Text Features.
   * **Phase 7**: Ridge / ElasticNet Stacking or Weighted Blending of best distinct representations.
