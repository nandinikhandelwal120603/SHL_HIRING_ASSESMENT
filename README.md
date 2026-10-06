# SHL Hiring Assessment 2026: Spoken English Grammar Scoring Engine

**Author**: Nandini Khandelwal  
**Kaggle Profile**: [nandinik3552](https://www.kaggle.com/nandinik3552)  
**Task**: Build an explainable, interview-defensible ML scoring engine that takes 45–60 second spoken `.wav` audio responses and predicts a continuous **Grammar Score in [0.0, 5.0]** evaluated by **RMSE** and **Pearson Correlation ($r$)**.  
**Competition Dataset**: 769 Training Audio Files, 216 Test Evaluation Audio Files.  
**Public Repository**: [https://github.com/nandinikhandelwal120603/SHL_HIRING_ASSESMENT](https://github.com/nandinikhandelwal120603/SHL_HIRING_ASSESMENT)

---

## 1. Executive Summary & Progression

| Milestone / Phase | Architecture | Key Features | OOF RMSE | OOF Pearson ($r$) | Training RMSE *(Compulsory)* |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Phase 2 Baseline** | Random Forest (`depth=6`) | 75 Handcrafted Acoustic Features | 0.8078 | 0.7591 | 0.6043 |
| **Phase 3 Multimodal** | Random Forest (`depth=6`) | Acoustic (75) + Linguistic (23) | 0.7651 | 0.7866 | 0.5642 |
| **Phase 4 Dual Blend** | 75% Multimodal RF + 25% MiniLM Ridge | Acoustic (75) + Ling (23) + Text Emb (384) | 0.7363 | 0.8194 | 0.5829 |
| **Phase 5 Unified Ridge** *(Candidate B)* | Ridge ($\alpha=100$) | All Modalities Concatenated (2,018) | **0.6170** | **0.8671** | **0.2196** |
| **Phase 5 Tri-Modal Blend** *(Candidate A)* | **50% Speech Ridge + 30% RF + 20% MiniLM** | **Acoustic + Linguistic + Text + Speech** | **0.6304** | **0.8786** | **0.4584** |

*All metrics measured under strict 5-fold stratified cross-validation across multiple random seeds.*

> **Comparative Evaluation**:  
> “The tri-modal blend achieved the strongest Pearson correlation among our shortlisted final models, while the all-modal Ridge achieved the lowest RMSE. Since the competition reports both metrics, we evaluate both as candidate submissions.”

---

## 2. Technical Philosophy & Modeling Architecture

Our modeling narrative follows a disciplined, human-reasoned progression:
> *“We started with interpretable acoustic features, then added linguistic information from transcripts, followed by pretrained text and speech representations. We retained each additional component only when validation showed useful complementary information. The final candidates therefore combine different views of the same speech sample rather than relying on a single representation.”*

```
                           Raw Audio (.wav)
                           /        |       \
                          /         |        \
                         v          v         v
           Handcrafted Acoustic  Whisper ASR  wav2vec2-base
           75 Prosody/Energy     base.en      Frozen Encoder
               Features             |               |
                         |          v               v
                         |   23 Handcrafted    1536-dim Speech
                         |   Linguistic Feats     Embeddings
                         \          |               |
                          \         v               |
                           \  all-MiniLM-L6-v2      |
                            \ 384-dim Text Emb      |
                             \      |               |
                              v     v               v
                        [ Random Forest ]    [ Ridge Regression ]
                               \                    /
                                \                  /
                                 v                v
                           [ Optimal Tri-Modal Blend ]
                                        ↓
                           Continuous Grammar Score [0, 5]
```

### Why Tri-Modal Blending Works
1. **Acoustic + Linguistic Random Forest (30% weight)**: Captures non-linear thresholds on pauses, hesitation counts, speech duration, and syntax proxies.
2. **MiniLM Sentence Transformer Ridge (20% weight)**: Captures grammatical coherence, sentence complexity, and vocabulary sophistication from transcribed text.
3. **wav2vec2 Speech Transformer Ridge (50% weight)**: Captures fine-grained phonetic pronunciation, acoustic formants, and acoustic delivery directly from raw audio waveforms.
4. **Error Independence**: Residual correlation between the two models' prediction errors (RF vs Speech Ridge) is only **$r = 0.449$**, confirming that they make independent errors that cancel out when blended.

---

## 3. Multi-Seed Robustness Evaluation

To confirm that performance improvements are genuine and not artifacts of one lucky split, all shortlisted models were evaluated across seeds `42`, `123`, and `2026`:

| Model Architecture | Seed 42 | Seed 123 | Seed 2026 | Mean Metric $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Acoustic + Linguistic RF** | Pearson: 0.7866<br>RMSE: 0.7651 | Pearson: 0.7857<br>RMSE: 0.7667 | Pearson: 0.7911<br>RMSE: 0.7584 | **Pearson: $0.7878 \pm 0.0024$**<br>**RMSE: $0.7634 \pm 0.0036$** |
| **75/25 RF + MiniLM Blend** | Pearson: 0.8194<br>RMSE: 0.7363 | Pearson: 0.8173<br>RMSE: 0.7399 | Pearson: 0.8199<br>RMSE: 0.7359 | **Pearson: $0.8189 \pm 0.0012$**<br>**RMSE: $0.7374 \pm 0.0018$** |
| **70/30 RF + MiniLM Blend** | Pearson: 0.8232<br>RMSE: 0.7371 | Pearson: 0.8208<br>RMSE: 0.7410 | Pearson: 0.8227<br>RMSE: 0.7381 | **Pearson: $0.8222 \pm 0.0010$**<br>**RMSE: $0.7387 \pm 0.0016$** |
| **wav2vec2 Speech Ridge** | Pearson: 0.8523<br>RMSE: 0.6477 | Pearson: 0.8550<br>RMSE: 0.6421 | Pearson: 0.8509<br>RMSE: 0.6504 | **Pearson: $0.8528 \pm 0.0017$**<br>**RMSE: $0.6467 \pm 0.0035$** |
| **All-Modalities Ridge (Cand. B)** | Pearson: 0.8671<br>RMSE: 0.6170 | Pearson: 0.8684<br>RMSE: 0.6140 | Pearson: 0.8669<br>RMSE: 0.6173 | **Pearson: $0.8675 \pm 0.0007$**<br>**RMSE: $\mathbf{0.6161 \pm 0.0015}$** |
| **Tri-Modal Blend (Cand. A)** | Pearson: 0.8786<br>RMSE: 0.6304 | Pearson: 0.8788<br>RMSE: 0.6304 | Pearson: 0.8766<br>RMSE: 0.6329 | **Pearson: $\mathbf{0.8780 \pm 0.0010}$**<br>**RMSE: $0.6312 \pm 0.0012$** |

*Note on Blend Weights*: The two blends (75/25 and 70/30) produced very similar results across the three tested random seeds. To avoid over-optimizing blend weights and preserve interview defensibility, we retained 75/25 as our primary baseline benchmark.

---

## 4. Final Candidate Submissions & Quality Control

Two final candidate submission files have been prepared and rigorously verified:

| Submission File | Architecture | OOF Pearson | OOF RMSE | Key Strength |
| :--- | :--- | :---: | :---: | :--- |
| **`submission/final_submission_trimodal.csv`** *(Primary)* | 50% Speech Ridge + 30% RF + 20% MiniLM | **0.8786** | 0.6304 | **Highest correlation ($r = 0.878$)**; blends independent error distributions |
| **`submission/final_submission_all_ridge.csv`** *(Secondary)* | All 2,018 features + Ridge ($\alpha=100$) | 0.8671 | **0.6170** | **Lowest RMSE ($0.617$)**; unified linear model |

### Verification Audit Checklist (Passed 100%)
* Exactly **216 rows** matching `test.csv` (resolving the 204-row starter template discrepancy).
* Columns strictly match `filename,label`.
* Filename ordering exactly matches `test.csv` (`audio_0.wav` to `audio_215.wav`).
* Zero NaNs, zero nulls, zero missing IDs.
* All predictions strictly bounded in $[0.0, 5.0]$.
* Continuous float predictions.

---

## 5. Kaggle Upload Order & Submission Protocol

1. **First Submission**: Upload `submission/final_submission_trimodal.csv`.
   - Record: Submission ID, Timestamp, Public Leaderboard RMSE and Pearson.
2. **Second Candidate (Only if needed)**: Upload `submission/final_submission_all_ridge.csv`.
   - Compare leaderboard score against Candidate A.
3. **Stopping Rule**: Once the submission meets the competition threshold, stop model development and record details for the Qualtrics form.

---

## 6. Repository Structure & Artifacts

```
.
├── notebook/
│   └── shl_grammar_scoring.ipynb            # Self-contained executable Jupyter Notebook
├── submission/
│   ├── final_submission_trimodal.csv        # Primary Candidate A (Highest Pearson)
│   ├── final_submission_all_ridge.csv       # Candidate B (Lowest RMSE)
│   └── final_submission.csv                 # Primary submission mirror
├── artifacts/
│   ├── experiment_log.csv                   # Complete log of all 24 evaluated models
│   ├── submission_log.csv                   # Candidate submission tracking table
│   ├── audio_features_train.parquet         # 75 handcrafted acoustic features (train)
│   ├── audio_features_test.parquet          # 75 handcrafted acoustic features (test)
│   ├── text_features_train.parquet          # 23 handcrafted linguistic features (train)
│   ├── text_features_test.parquet           # 23 handcrafted linguistic features (test)
│   ├── transcripts/                         # Cached Whisper ASR JSON transcripts
│   ├── text_embeddings/                     # Cached 384-dim MiniLM embeddings
│   └── audio_embeddings/                    # Cached 1536-dim wav2vec2 embeddings
├── scripts/
│   ├── build_notebook.py                    # Notebook builder and execution script
│   ├── transcribe_dataset.py                # Whisper ASR transcription script
│   └── extract_speech_embeddings.py         # wav2vec2 MPS extraction script
└── README.md                                # Technical overview and documentation
```

---

## 7. How to Reproduce

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full Notebook**:
   ```bash
   python scripts/build_notebook.py
   ```
   Or open and run `notebook/shl_grammar_scoring.ipynb` top-to-bottom in Jupyter Lab / VS Code. All heavy audio/text extractions are cached in `artifacts/`, allowing the entire notebook to execute in under 60 seconds without GPU clusters.
