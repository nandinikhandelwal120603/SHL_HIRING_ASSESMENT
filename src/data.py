"""
Data loading and dataset validation utilities for SHL Hiring Assessment 2026.
Ensures full path preservation to prevent train/test filename collisions.
"""

import os
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Dataset_Final"))


def load_datasets(data_dir: str = DATA_DIR):
    """
    Loads train, test, and sample submission dataframes with verified absolute audio paths.
    
    CRITICAL GOTCHA: Filenames like 'audio_128.wav' exist in BOTH train/ and test/,
    referencing completely distinct recordings. This function injects 'audio_path'
    so models never look up audio files by bare filename.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path)
    
    # Add absolute paths
    train_df["audio_path"] = train_df["filename"].apply(
        lambda f: os.path.join(data_dir, "train", f)
    )
    test_df["audio_path"] = test_df["filename"].apply(
        lambda f: os.path.join(data_dir, "test", f)
    )
    
    # Verify file existence
    assert train_df["audio_path"].apply(os.path.exists).all(), "Some train audio files are missing!"
    assert test_df["audio_path"].apply(os.path.exists).all(), "Some test audio files are missing!"
    
    return train_df, test_df, sub_df
