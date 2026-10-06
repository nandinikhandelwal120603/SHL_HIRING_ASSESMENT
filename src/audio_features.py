"""
Audio feature extraction for SHL Grammar Scoring Engine.
Extracts time-domain, speech-activity/pauses, spectral, MFCC, and pitch features.
Uses multi-threading and STFT reuse for high speed.
"""

import os
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import decimate, correlate
import librosa
from tqdm import tqdm


def extract_features_from_audio(audio_path: str) -> dict:
    """
    Extracts comprehensive acoustic and fluency proxy features from a single audio file.
    """
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    total_samples = len(data)
    duration = float(total_samples / sr)
    
    feats = {"duration": duration}
    
    # 1. Amplitude & Time Domain
    abs_data = np.abs(data)
    peak_amp = float(np.max(abs_data)) if len(abs_data) > 0 else 0.0
    feats["peak_amplitude"] = peak_amp
    
    # 2. Frame-level RMS and Pause Analysis
    # 25ms window, 10ms hop
    hop_vad = int(0.01 * sr)
    win_vad = int(0.025 * sr)
    num_vad_frames = (len(data) - win_vad) // hop_vad + 1
    
    if num_vad_frames > 0:
        strided_frames = np.lib.stride_tricks.as_strided(
            data,
            shape=(num_vad_frames, win_vad),
            strides=(data.strides[0] * hop_vad, data.strides[0])
        )
        frame_rms = np.sqrt(np.mean(strided_frames**2, axis=1))
        
        feats["rms_mean"] = float(np.mean(frame_rms))
        feats["rms_std"] = float(np.std(frame_rms))
        feats["rms_median"] = float(np.median(frame_rms))
        feats["rms_p25"] = float(np.percentile(frame_rms, 25))
        feats["rms_p75"] = float(np.percentile(frame_rms, 75))
        feats["rms_max"] = float(np.max(frame_rms))
        feats["crest_factor"] = float(peak_amp / (feats["rms_mean"] + 1e-8))
        
        # Voice Activity & Pause Detection
        # Threshold: 10% of 90th percentile energy
        silence_threshold = 0.10 * np.percentile(frame_rms, 90)
        is_silent = frame_rms < silence_threshold
        
        pause_durations = []
        for silent, group in groupby(is_silent):
            if silent:
                dur = len(list(group)) * 0.01
                if dur >= 0.25:  # Pause threshold: >= 250ms
                    pause_durations.append(dur)
                    
        total_pause_time = float(sum(pause_durations))
        active_speech_time = max(0.0, duration - total_pause_time)
        
        feats["active_speech_duration"] = active_speech_time
        feats["silence_duration"] = total_pause_time
        feats["speech_ratio"] = float(active_speech_time / (duration + 1e-8))
        feats["silence_ratio"] = float(total_pause_time / (duration + 1e-8))
        feats["num_pauses"] = float(len(pause_durations))
        feats["mean_pause_duration"] = float(np.mean(pause_durations)) if pause_durations else 0.0
        feats["max_pause_duration"] = float(np.max(pause_durations)) if pause_durations else 0.0
        feats["pause_rate_per_min"] = float(len(pause_durations) / (duration / 60.0 + 1e-8))
    else:
        for k in ["rms_mean", "rms_std", "rms_median", "rms_p25", "rms_p75", "rms_max", "crest_factor",
                  "active_speech_duration", "silence_duration", "speech_ratio", "silence_ratio",
                  "num_pauses", "mean_pause_duration", "max_pause_duration", "pause_rate_per_min"]:
            feats[k] = 0.0

    # 3. Spectral Features (STFT reuse for optimal efficiency)
    hop_length = 1024
    n_fft = 2048
    S = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    power_spec = S**2
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(data, hop_length=hop_length)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))
    feats["zcr_median"] = float(np.median(zcr))
    
    # Spectral Centroid
    sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(np.mean(sc))
    feats["spectral_centroid_std"] = float(np.std(sc))
    feats["spectral_centroid_median"] = float(np.median(sc))
    
    # Spectral Bandwidth
    sb = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    feats["spectral_bandwidth_mean"] = float(np.mean(sb))
    feats["spectral_bandwidth_std"] = float(np.std(sb))
    feats["spectral_bandwidth_median"] = float(np.median(sb))
    
    # Spectral Rolloff (85%)
    s_roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    feats["spectral_rolloff_mean"] = float(np.mean(s_roll))
    feats["spectral_rolloff_std"] = float(np.std(s_roll))
    feats["spectral_rolloff_median"] = float(np.median(s_roll))
    
    # Spectral Flatness
    s_flat = librosa.feature.spectral_flatness(S=S)[0]
    feats["spectral_flatness_mean"] = float(np.mean(s_flat))
    feats["spectral_flatness_std"] = float(np.std(s_flat))
    feats["spectral_flatness_median"] = float(np.median(s_flat))
    
    # 4. MFCCs (13 coefficients)
    mel_spec = librosa.feature.melspectrogram(S=power_spec, sr=sr, n_mels=40)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec + 1e-10), n_mfcc=13)
    for i in range(13):
        coeff = mfccs[i]
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(coeff))
        feats[f"mfcc_{i+1}_std"] = float(np.std(coeff))
        feats[f"mfcc_{i+1}_median"] = float(np.median(coeff))
        
    # 5. Pitch / F0 (Autocorrelation on downsampled audio)
    try:
        data_down = decimate(data, 4)
        sr_down = sr // 4
        f_len = int(0.04 * sr_down)  # 40ms
        f_hop = int(0.02 * sr_down)  # 20ms
        min_lag = int(sr_down / 400) # 400 Hz
        max_lag = int(sr_down / 70)  # 70 Hz
        
        f0_list = []
        total_f_count = 0
        for i in range(0, len(data_down) - f_len, f_hop):
            total_f_count += 1
            frame = data_down[i:i+f_len]
            if np.std(frame) < 0.01:
                continue
            corr = correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            r_val = corr[peak_lag] / (corr[0] + 1e-8)
            if r_val > 0.4:
                f0_list.append(sr_down / peak_lag)
                
        if len(f0_list) > 5:
            feats["f0_mean"] = float(np.mean(f0_list))
            feats["f0_std"] = float(np.std(f0_list))
            feats["f0_median"] = float(np.median(f0_list))
            feats["voiced_fraction"] = float(len(f0_list) / max(1, total_f_count))
        else:
            feats["f0_mean"] = 0.0
            feats["f0_std"] = 0.0
            feats["f0_median"] = 0.0
            feats["voiced_fraction"] = 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0
        feats["f0_median"] = 0.0
        feats["voiced_fraction"] = 0.0
        
    return feats


def extract_features_dataset(df: pd.DataFrame, cache_path: str = None, num_workers: int = 4) -> pd.DataFrame:
    """
    Extracts acoustic features for all rows in df['audio_path'].
    Loads from cache if cache_path exists.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)
    
    paths = df["audio_path"].tolist()
    print(f"Extracting handcrafted acoustic features for {len(paths)} files (using {num_workers} threads)...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(extract_features_from_audio, paths), total=len(paths)))
        
    feats_df = pd.DataFrame(results)
    feats_df["filename"] = df["filename"].values
    if "label" in df.columns:
        feats_df["label"] = df["label"].values
        
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        feats_df.to_parquet(cache_path, index=False)
        print(f"Cached features saved to {cache_path}")
        
    return feats_df
