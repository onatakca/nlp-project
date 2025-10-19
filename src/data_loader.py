"""
Data loading and preprocessing utilities for genre classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from .utils import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, BALANCED_DATA_PATH,
    PLACEHOLDER_PATTERNS, GENRES, GENRE_TO_ID, set_seed
)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw song lyrics dataset.

    Args:
        path: Path to CSV file. If None, uses default RAW_DATA_PATH.

    Returns:
        DataFrame with raw song lyrics data
    """
    if path is None:
        path = RAW_DATA_PATH

    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} songs")
    print(f"   Columns: {list(df.columns)}")
    return df


def is_placeholder(text: str) -> bool:
    """Check if text contains placeholder patterns instead of real lyrics."""
    if pd.isna(text):
        return True

    text_lower = str(text).lower()
    return any(pattern in text_lower for pattern in PLACEHOLDER_PATTERNS)


def clean_lyrics(text: str) -> str:
    """
    Clean lyric text:
    - Convert to lowercase
    - Remove special markers
    - Strip whitespace

    Args:
        text: Raw lyrics text

    Returns:
        Cleaned lyrics text
    """
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove common markers
    text = text.replace('[verse]', '').replace('[chorus]', '')
    text = text.replace('[bridge]', '').replace('[intro]', '')
    text = text.replace('[outro]', '').replace('\\n', ' ')

    # Strip and remove extra whitespace
    text = ' '.join(text.split())

    return text


def preprocess_data(df: pd.DataFrame, remove_placeholders: bool = True) -> pd.DataFrame:
    """
    Preprocess the dataset:
    - Remove placeholder lyrics
    - Clean lyrics text
    - Filter to valid genres

    Args:
        df: Raw DataFrame
        remove_placeholders: Whether to remove songs with placeholder text

    Returns:
        Cleaned DataFrame
    """
    print("\nPreprocessing data...")
    initial_count = len(df)

    # Filter to valid genres (exclude 'misc')
    df = df[df['tag'].isin(GENRES)].copy()
    print(f"  Filtered to {len(GENRES)} genres: {len(df):,} songs")

    # Remove placeholders if requested
    if remove_placeholders:
        df = df[~df['lyrics'].apply(is_placeholder)].copy()
        removed = initial_count - len(df)
        print(f"  Removed {removed:,} placeholder lyrics")

    # Clean lyrics
    df['lyrics_cleaned'] = df['lyrics'].apply(clean_lyrics)

    # Remove empty lyrics
    df = df[df['lyrics_cleaned'].str.len() > 0].copy()
    print(f"  Final dataset: {len(df):,} songs")

    return df


def balance_dataset(df: pd.DataFrame, samples_per_genre: int) -> pd.DataFrame:
    """
    Create balanced dataset by sampling equal numbers from each genre.

    Args:
        df: DataFrame with 'tag' column
        samples_per_genre: Number of samples to draw from each genre

    Returns:
        Balanced DataFrame
    """
    print(f"\nBalancing dataset to {samples_per_genre:,} samples per genre...")

    sampled_dfs = []
    for genre in GENRES:
        genre_df = df[df['tag'] == genre]
        n_available = len(genre_df)
        n_samples = min(n_available, samples_per_genre)

        if n_samples < samples_per_genre:
            print(f"  ⚠️  {genre}: only {n_available:,} available (requested {samples_per_genre:,})")
        else:
            print(f"  {genre}: {n_samples:,}")

        sampled = genre_df.sample(n=n_samples, random_state=42)
        sampled_dfs.append(sampled)

    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n✅ Balanced dataset: {len(balanced_df):,} songs")
    return balanced_df


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        df: DataFrame with 'lyrics_cleaned' and 'tag' columns
        test_size: Fraction of data for test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df['lyrics_cleaned']
    y = df['tag'].map(GENRE_TO_ID)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n✅ Train/Test split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test


def get_dataset_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics about the dataset.

    Args:
        df: DataFrame with 'lyrics_cleaned' and 'tag' columns

    Returns:
        DataFrame with statistics per genre
    """
    stats = []

    for genre in GENRES:
        genre_df = df[df['tag'] == genre]

        if len(genre_df) == 0:
            continue

        lyrics = genre_df['lyrics_cleaned']
        word_counts = lyrics.str.split().str.len()
        char_counts = lyrics.str.len()

        stats.append({
            'genre': genre,
            'count': len(genre_df),
            'avg_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'avg_chars': char_counts.mean(),
            'median_chars': char_counts.median()
        })

    return pd.DataFrame(stats)


# Convenience function for complete pipeline
def load_and_prepare_data(
    samples_per_genre: int = 20000,
    test_size: float = 0.2,
    use_cached: bool = True,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Complete data loading and preparation pipeline.

    Args:
        samples_per_genre: Number of samples per genre for balanced dataset
        test_size: Fraction of data for test set
        use_cached: If True, loads from balanced CSV if it exists
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    set_seed(random_state)

    # Try to load cached balanced data
    if use_cached and BALANCED_DATA_PATH.exists():
        print(f"Loading cached balanced dataset from {BALANCED_DATA_PATH}")
        df = pd.read_csv(BALANCED_DATA_PATH)

        # Filter to requested sample size if needed
        if samples_per_genre * len(GENRES) < len(df):
            df = balance_dataset(df, samples_per_genre)
    else:
        # Full pipeline
        df = load_raw_data()
        df = preprocess_data(df)
        df = balance_dataset(df, samples_per_genre)

        # Save for future use
        print(f"\nSaving balanced dataset to {BALANCED_DATA_PATH}")
        df.to_csv(BALANCED_DATA_PATH, index=False)

    # Create train/test split
    return create_train_test_split(df, test_size, random_state)
