# massif_utils.py
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import torch
from transformers import GPT2Tokenizer

def safe_phase_coherence(hidden):
    """Robust ρ_R that works on single vectors"""
    try:
        if isinstance(hidden, np.ndarray):
            if hidden.ndim == 1:
                hidden = hidden.reshape(1, -1)
            if hidden.shape[0] == 1:
                return 1.0
            norms = np.linalg.norm(hidden, axis=1, keepdims=True)
            normalized = hidden / (norms + 1e-8)
            gram = normalized @ normalized.T
            eig = np.linalg.eigvalsh(gram)
            return float(np.max(eig) / (np.sum(np.abs(eig)) + 1e-8))
        return np.nan
    except:
        return np.nan

def safe_anisotropy(hidden):
    try:
        if isinstance(hidden, np.ndarray):
            if hidden.ndim == 1:
                hidden = hidden.reshape(1, -1)
            if hidden.shape[0] == 1:
                return 1.0
            cov = np.cov(hidden.T)
            eig = np.linalg.eigvalsh(cov)
            return float(np.max(eig) / (np.sum(eig) + 1e-8))
        return np.nan
    except:
        return np.nan

def safe_complexity(hidden):
    try:
        if isinstance(hidden, np.ndarray):
            if hidden.ndim == 1 or hidden.shape[0] == 1:
                return 0.0
            norms = np.linalg.norm(hidden, axis=1)
            norms = norms / (norms.sum() + 1e-8)
            return float(entropy(norms))
        return np.nan
    except:
        return np.nan

def compute_embedding_gap(hidden):
    try:
        tokens = hidden[0] if len(hidden.shape) == 3 else hidden
        if tokens.shape[0] < 3:
            return np.nan
        dist_raw = pdist(tokens, 'cosine')
        centered = tokens - tokens.mean(axis=0)
        dist_centered = pdist(centered, 'cosine')
        return float(np.mean(dist_raw) - np.mean(dist_centered))
    except:
        return np.nan

# Tokenizer helper
def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
