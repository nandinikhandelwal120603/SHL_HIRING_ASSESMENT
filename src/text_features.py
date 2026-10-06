"""
Linguistic and text feature extraction from speech transcripts.
Extracts speech quantity, lexical diversity, disfluencies/fillers,
sentence structure, POS distributions, and grammar error heuristics.
"""

import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag


COMMON_FILLERS = set(["um", "uh", "erm", "ah", "er", "hmm", "like"])

SVA_PATTERNS = [
    re.compile(r"\bthere was (many|several|few|two|three|people|cars|students|problems)\b", re.I),
    re.compile(r"\bthey was\b", re.I),
    re.compile(r"\bwe was\b", re.I),
    re.compile(r"\byou was\b", re.I),
    re.compile(r"\bhe don't\b", re.I),
    re.compile(r"\bshe don't\b", re.I),
    re.compile(r"\bit don't\b", re.I),
    re.compile(r"\bhe have\b", re.I),
    re.compile(r"\bshe have\b", re.I),
]


def extract_linguistic_features(text: str, duration: float = 60.0) -> dict:
    """
    Extracts explainable linguistic, lexical, and grammatical proxy features from text.
    """
    feats = {}
    cleaned_text = text.strip() if isinstance(text, str) else ""
    
    # Raw lengths
    feats["char_count"] = float(len(cleaned_text))
    
    if not cleaned_text:
        default_keys = [
            "word_count", "words_per_sec", "words_per_min",
            "unique_word_count", "type_token_ratio", "avg_word_length",
            "filler_count", "filler_ratio", "repeated_word_count",
            "sentence_count", "avg_sentence_length", "sentence_length_std",
            "comma_count", "question_count",
            "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
            "pronoun_ratio", "prep_ratio",
            "fragment_sentence_count", "sva_error_proxy_count"
        ]
        for k in default_keys:
            feats[k] = 0.0
        return feats

    # 1. Tokenization
    raw_tokens = word_tokenize(cleaned_text)
    words = [w.lower() for w in raw_tokens if w.isalnum()]
    word_count = len(words)
    feats["word_count"] = float(word_count)
    
    # Speech quantity normalized by duration
    safe_dur = max(duration, 5.0)
    feats["words_per_sec"] = float(word_count / safe_dur)
    feats["words_per_min"] = float((word_count / safe_dur) * 60.0)
    
    if word_count == 0:
        for k in ["unique_word_count", "type_token_ratio", "avg_word_length",
                  "filler_count", "filler_ratio", "repeated_word_count",
                  "sentence_count", "avg_sentence_length", "sentence_length_std",
                  "comma_count", "question_count",
                  "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
                  "pronoun_ratio", "prep_ratio",
                  "fragment_sentence_count", "sva_error_proxy_count"]:
            feats[k] = 0.0
        return feats

    # 2. Lexical Diversity
    unique_words = set(words)
    unique_count = len(unique_words)
    feats["unique_word_count"] = float(unique_count)
    feats["type_token_ratio"] = float(unique_count / word_count)
    feats["avg_word_length"] = float(np.mean([len(w) for w in words]))
    
    # 3. Disfluencies & Fillers
    filler_tokens = [w for w in words if w in COMMON_FILLERS]
    feats["filler_count"] = float(len(filler_tokens))
    feats["filler_ratio"] = float(len(filler_tokens) / word_count)
    
    # Immediate word repetitions (e.g. "I I", "the the")
    repeated_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1])
    feats["repeated_word_count"] = float(repeated_count)
    
    # 4. Sentence Structure
    sentences = sent_tokenize(cleaned_text)
    sent_count = max(len(sentences), 1)
    feats["sentence_count"] = float(sent_count)
    
    sent_lengths = [len([w for w in word_tokenize(s) if w.isalnum()]) for s in sentences]
    feats["avg_sentence_length"] = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    feats["sentence_length_std"] = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    
    # Fragment sentences (<= 3 words)
    feats["fragment_sentence_count"] = float(sum(1 for l in sent_lengths if l <= 3))
    
    # Punctuation markers
    feats["comma_count"] = float(cleaned_text.count(","))
    feats["question_count"] = float(cleaned_text.count("?"))
    
    # 5. Part of Speech (POS) Distribution
    pos_tags = pos_tag(words)
    tag_prefixes = Counter([t[1][:2] for t in pos_tags])
    
    feats["noun_ratio"] = float((tag_prefixes.get("NN", 0)) / word_count)
    feats["verb_ratio"] = float((tag_prefixes.get("VB", 0)) / word_count)
    feats["adj_ratio"] = float((tag_prefixes.get("JJ", 0)) / word_count)
    feats["adv_ratio"] = float((tag_prefixes.get("RB", 0)) / word_count)
    feats["pronoun_ratio"] = float((tag_prefixes.get("PR", 0)) / word_count)
    feats["prep_ratio"] = float((tag_prefixes.get("IN", 0)) / word_count)
    
    # 6. Grammatical Error Heuristics (Rule-based proxy)
    sva_errors = 0
    for pat in SVA_PATTERNS:
        sva_errors += len(pat.findall(cleaned_text))
    feats["sva_error_proxy_count"] = float(sva_errors)
    
    return feats


def extract_linguistic_features_df(transcripts_dict: dict, audio_durations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features for all files given a transcript dictionary and audio duration table.
    """
    rows = []
    for _, row in audio_durations_df.iterrows():
        fname = row["filename"]
        dur = float(row.get("duration", 60.0))
        text_info = transcripts_dict.get(fname, {})
        text = text_info.get("text", "") if isinstance(text_info, dict) else str(text_info)
        
        feats = extract_linguistic_features(text, duration=dur)
        feats["filename"] = fname
        if "label" in row:
            feats["label"] = row["label"]
        rows.append(feats)
        
    return pd.DataFrame(rows)
