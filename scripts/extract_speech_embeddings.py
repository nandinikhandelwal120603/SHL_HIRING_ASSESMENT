import os
import sys
import time
import soundfile as sf
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "Dataset_Final")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
EMB_DIR = os.path.join(ARTIFACTS_DIR, "audio_embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

train_out = os.path.join(EMB_DIR, "train_speech_embeddings.parquet")
test_out = os.path.join(EMB_DIR, "test_speech_embeddings.parquet")

if os.path.exists(train_out) and os.path.exists(test_out):
    print("Speech embeddings already exist at", EMB_DIR)
    sys.exit(0)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model_id = "facebook/wav2vec2-base"
print(f"Loading speech model: {model_id}...")
t0 = time.time()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()
t1 = time.time()

num_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {model_id} in {t1-t0:.2f}s.")
print(f"Model parameters: {num_params:,} (~{num_params*4 / (1024**2):.1f} MB in float32)")

train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_audio_dir = os.path.join(DATA_DIR, "train")
test_audio_dir = os.path.join(DATA_DIR, "test")

MAX_SECONDS = 50.0
MAX_SAMPLES = int(16000 * MAX_SECONDS)

def extract_for_dataset(df, audio_dir, desc="train"):
    embeddings = []
    filenames = []
    start_time = time.time()
    
    for idx, row in df.iterrows():
        fn = row['filename']
        audio_path = os.path.join(audio_dir, fn)
        
        try:
            y, sr = sf.read(audio_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            if len(y) > MAX_SAMPLES:
                y = y[:MAX_SAMPLES]
                
            inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state # (1, T, 768)
                
            mean_pool = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            std_pool = hidden_states.std(dim=1).squeeze(0).cpu().numpy()
            pooled = np.concatenate([mean_pool, std_pool])
            
            embeddings.append(pooled)
            filenames.append(fn)
            
            del inputs, input_values, outputs, hidden_states
            if device.type == "mps":
                torch.mps.empty_cache()
                
            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                eta = (len(df) - (idx + 1)) * avg_time
                print(f"[{desc}] Processed {idx+1}/{len(df)} files ({avg_time:.3f}s/file, ETA: {eta/60:.1f}m)...")
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            embeddings.append(np.zeros(768 * 2, dtype=np.float32))
            filenames.append(fn)
            
    total_elapsed = time.time() - start_time
    avg_per_file = total_elapsed / len(df)
    print(f"[{desc}] Finished {len(df)} files in {total_elapsed:.1f}s ({avg_per_file:.3f}s/file).")
    
    emb_arr = np.array(embeddings, dtype=np.float32)
    cols = [f"emb_speech_mean_{i}" for i in range(768)] + [f"emb_speech_std_{i}" for i in range(768)]
    emb_df = pd.DataFrame(emb_arr, columns=cols)
    emb_df.insert(0, "filename", filenames)
    return emb_df

print("Starting extraction on train set...")
train_emb_df = extract_for_dataset(train_df, train_audio_dir, desc="train")
train_emb_df.to_parquet(train_out, index=False)
print(f"Saved train embeddings: {train_emb_df.shape} -> {train_out}")

print("Starting extraction on test set...")
test_emb_df = extract_for_dataset(test_df, test_audio_dir, desc="test")
test_emb_df.to_parquet(test_out, index=False)
print(f"Saved test embeddings: {test_emb_df.shape} -> {test_out}")

print("Done extracting speech embeddings!")
