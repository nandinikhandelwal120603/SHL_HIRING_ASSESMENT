"""
Transcribes training and test audio using Whisper base.en.
Caches results to artifacts/transcripts/train_transcripts.json and test_transcripts.json.
Uses multi-processing across CPU cores for maximum speed on Apple Silicon.
"""

import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm


def transcribe_batch(worker_args):
    """
    Worker function to transcribe a list of (filename, audio_path) tuples.
    Loads Whisper base.en inside the process for isolation and concurrency.
    """
    file_tuples, worker_id = worker_args
    import torch
    import whisper
    
    # 2 threads per worker
    torch.set_num_threads(2)
    model = whisper.load_model("base.en", device="cpu")
    
    results = {}
    for filename, audio_path in file_tuples:
        try:
            res = model.transcribe(audio_path, fp16=False, language="en")
            results[filename] = {
                "text": res["text"].strip(),
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in res.get("segments", [])
                ]
            }
        except Exception as e:
            results[filename] = {"text": "", "error": str(e)}
            
    return results


def run_transcription(csv_path, audio_dir, out_cache_path, num_workers=4):
    if os.path.exists(out_cache_path):
        print(f"Cache already exists at {out_cache_path}. Skipping.")
        with open(out_cache_path, "r") as f:
            return json.load(f)
            
    os.makedirs(os.path.dirname(os.path.abspath(out_cache_path)), exist_ok=True)
    df = pd.read_csv(csv_path)
    file_tuples = [(row["filename"], os.path.join(audio_dir, row["filename"])) for _, row in df.iterrows()]
    
    print(f"Starting Whisper transcription of {len(file_tuples)} files with {num_workers} parallel workers...")
    t0 = time.time()
    
    # Chunk across workers
    chunk_size = (len(file_tuples) + num_workers - 1) // num_workers
    chunks = [
        (file_tuples[i:i + chunk_size], idx)
        for idx, i in enumerate(range(0, len(file_tuples), chunk_size))
    ]
    
    all_results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        worker_results = list(executor.map(transcribe_batch, chunks))
        for res in worker_results:
            all_results.update(res)
            
    elapsed = time.time() - t0
    print(f"Completed {len(all_results)} files in {elapsed:.1f}s ({elapsed/len(all_results):.2f}s per file).")
    
    with open(out_cache_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved transcripts to {out_cache_path}")
    
    return all_results


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data", "Dataset_Final")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    
    train_cache = os.path.join(ARTIFACTS_DIR, "transcripts", "train_transcripts.json")
    test_cache = os.path.join(ARTIFACTS_DIR, "transcripts", "test_transcripts.json")
    
    print("--- Transcribing Train Files ---")
    run_transcription(
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(DATA_DIR, "train"),
        train_cache,
        num_workers=4
    )
    
    print("\n--- Transcribing Test Files ---")
    run_transcription(
        os.path.join(DATA_DIR, "test.csv"),
        os.path.join(DATA_DIR, "test"),
        test_cache,
        num_workers=4
    )
