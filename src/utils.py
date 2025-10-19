"""
Utility functions and constants for genre classification project.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
RESULTS_DIR = EXPERIMENTS_DIR / 'results'
CONFIGS_DIR = EXPERIMENTS_DIR / 'configs'

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Genre constants
GENRES = ['country', 'pop', 'rap', 'rb', 'rock']
GENRE_TO_ID = {genre: i for i, genre in enumerate(GENRES)}
ID_TO_GENRE = {i: genre for genre, i in GENRE_TO_ID.items()}
NUM_GENRES = len(GENRES)

# Data file paths
RAW_DATA_PATH = DATA_DIR / 'song_lyrics.csv'
CLEANED_DATA_PATH = DATA_DIR / 'song_lyrics_cleaned.csv'
BALANCED_DATA_PATH = DATA_DIR / 'song_lyrics_balanced.csv'

# Text preprocessing constants
PLACEHOLDER_PATTERNS = [
    'tell us that you would like to have the lyrics',
    'we do not have the lyrics',
    'lyrics not available'
]


def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }

    if info['cuda_available']:
        for i in range(info['device_count']):
            info['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / 1e9,  # GB
            })

    return info


def save_results(results: Dict[str, Any], filename: str):
    """Save experiment results to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {filepath}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'r') as f:
        return json.load(f)


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Pretty print configuration dictionary."""
    print("=" * 70)
    print(title.upper())
    print("=" * 70)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("=" * 70)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Random seed set to {seed}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
