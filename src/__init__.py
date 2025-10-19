"""
Genre Classification from Song Lyrics
======================================

A modular NLP project comparing classical ML (TF-IDF), word embeddings (Word2Vec),
and transformer models (BERT/DistilBERT) for music genre classification.

Modules:
    - data_loader: Data loading, cleaning, and preprocessing
    - models: Model wrappers for TF-IDF, Word2Vec, and BERT
    - train: Training loops and optimization
    - evaluate: Evaluation metrics and visualization
    - utils: Helper functions and constants
"""

__version__ = '1.0.0'
__author__ = 'NLP Project Team'

from . import data_loader
from . import models
from . import utils

__all__ = ['data_loader', 'models', 'utils']
