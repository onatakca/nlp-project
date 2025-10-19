#!/usr/bin/env python3
"""
Master training script for genre classification experiments.

Usage:
    python scripts/train.py --config experiments/configs/tfidf_config.yaml
    python scripts/train.py --config experiments/configs/bert_config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_and_prepare_data
from src.models import TFIDFModel, Word2VecModel, BERTModel
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_per_genre_metrics
from src.utils import save_results, print_config, get_gpu_info, set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_tfidf(config: dict, X_train, y_train, X_test, y_test):
    """Train TF-IDF model."""
    print("\n" + "="*70)
    print("TRAINING TF-IDF MODEL")
    print("="*70)

    model_config = config['model']
    model = TFIDFModel(
        classifier_type=model_config['classifier'],
        max_features=model_config['max_features'],
        ngram_range=tuple(model_config['ngram_range']),
        min_df=model_config['min_df'],
        max_df=model_config['max_df']
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred


def train_word2vec(config: dict, X_train, y_train, X_test, y_test):
    """Train Word2Vec model."""
    print("\n" + "="*70)
    print("TRAINING WORD2VEC MODEL")
    print("="*70)

    model_config = config['model']
    model = Word2VecModel(
        classifier_type=model_config['classifier'],
        vector_size=model_config['vector_size'],
        window=model_config['window'],
        min_count=model_config['min_count'],
        epochs=model_config['epochs']
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred


def train_bert(config: dict, X_train, y_train, X_test, y_test):
    """Train BERT model."""
    print("\n" + "="*70)
    print("TRAINING BERT MODEL")
    print("="*70)

    # Print GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    print(f"  CUDA available: {gpu_info['cuda_available']}")
    print(f"  Number of GPUs: {gpu_info['device_count']}")
    for gpu in gpu_info['devices']:
        print(f"    GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']:.1f} GB)")

    model_config = config['model']
    model = BERTModel(
        model_name=model_config['model_name'],
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'],
        learning_rate=model_config['learning_rate'],
        epochs=model_config['epochs'],
        warmup_ratio=model_config['warmup_ratio'],
        weight_decay=model_config['weight_decay'],
        use_amp=model_config['use_amp']
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train genre classification model")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print_config(config, "Experiment Configuration")

    # Set seed
    set_seed(config['data']['random_state'])

    # Load and prepare data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        samples_per_genre=config['data']['samples_per_genre'],
        test_size=config['data']['test_size'],
        use_cached=config['data']['use_cached'],
        random_state=config['data']['random_state']
    )

    # Train model based on type
    model_type = config['model']['type']

    if model_type == 'tfidf':
        model, y_pred = train_tfidf(config, X_train, y_train, X_test, y_test)
    elif model_type == 'word2vec':
        model, y_pred = train_word2vec(config, X_train, y_train, X_test, y_test)
    elif model_type == 'bert':
        model, y_pred = train_bert(config, X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    experiment_name = config['experiment']['name']
    results = evaluate_model(y_test, y_pred, model_name=experiment_name)

    # Visualizations
    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"Confusion Matrix: {experiment_name}"
    )

    plot_per_genre_metrics(
        results,
        title=f"Per-Genre Performance: {experiment_name}"
    )

    # Save results
    if config['experiment']['save_results'] and not args.no_save:
        results_file = f"{experiment_name}_results.json"
        save_results(results, results_file)

    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_avg']['f1']:.4f}")


if __name__ == '__main__':
    main()
