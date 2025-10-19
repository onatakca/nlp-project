"""
Evaluation metrics and visualization for genre classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from .utils import GENRES, ID_TO_GENRE

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict:
    """
    Comprehensive evaluation of a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display

    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to numpy arrays for consistent indexing
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Convert to genre names if numeric
    if isinstance(y_true_arr[0], (int, np.integer)):
        y_true_labels = [ID_TO_GENRE[int(y)] for y in y_true_arr]
        y_pred_labels = [ID_TO_GENRE[int(y)] for y in y_pred_arr]
    else:
        y_true_labels = y_true_arr.tolist()
        y_pred_labels = y_pred_arr.tolist()

    # Overall accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels,
        y_pred_labels,
        labels=GENRES,
        zero_division=0
    )

    # Print results
    print("=" * 70)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nPer-Genre Performance:")
    print(f"{'Genre':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 70)
    for i, genre in enumerate(GENRES):
        print(f"{genre:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]}")

    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # Package results
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'per_genre': {
            genre: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i, genre in enumerate(GENRES)
        },
        'per_class': {
            str(i): {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i, genre in enumerate(GENRES)
        },
        'macro_avg': {
            'precision': float(precision.mean()),
            'recall': float(recall.mean()),
            'f1': float(f1.mean())
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1)
        }
    }

    return results


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        normalize: Whether to normalize values
        save_path: Path to save figure (optional)
    """
    # Convert to numpy arrays for consistent indexing
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Convert to genre names if numeric
    if isinstance(y_true_arr[0], (int, np.integer)):
        y_true_labels = [ID_TO_GENRE[int(y)] for y in y_true_arr]
        y_pred_labels = [ID_TO_GENRE[int(y)] for y in y_pred_arr]
    else:
        y_true_labels = y_true_arr.tolist()
        y_pred_labels = y_pred_arr.tolist()

    # Compute confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=GENRES)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'YlOrRd'

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=GENRES,
        yticklabels=GENRES,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")

    plt.show()


def plot_per_genre_metrics(
    results: Dict,
    title: str = "Per-Genre Performance",
    save_path: Optional[str] = None
):
    """
    Plot bar chart of per-genre metrics.

    Args:
        results: Results dictionary from evaluate_model
        title: Plot title
        save_path: Path to save figure (optional)
    """
    genres = GENRES
    precision = [results['per_genre'][g]['precision'] for g in genres]
    recall = [results['per_genre'][g]['recall'] for g in genres]
    f1 = [results['per_genre'][g]['f1'] for g in genres]

    x = np.arange(len(genres))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(genres)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-genre metrics plot saved to {save_path}")

    plt.show()


def compare_models(
    results_dict: Dict[str, Dict],
    metric: str = 'f1',
    title: str = "Model Comparison",
    save_path: Optional[str] = None
):
    """
    Compare multiple models across genres.

    Args:
        results_dict: Dictionary mapping model names to results dictionaries
        metric: Metric to compare ('precision', 'recall', or 'f1')
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Convert dict to list if needed
    if isinstance(results_dict, dict):
        results_list = list(results_dict.values())
    else:
        results_list = results_dict

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(GENRES))
    width = 0.8 / len(results_list)

    for i, results in enumerate(results_list):
        values = [results['per_genre'][g][metric] for g in GENRES]
        offset = (i - len(results_list)/2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=results['model_name'],
            alpha=0.8
        )

    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GENRES)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to {save_path}")

    plt.show()


def create_results_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of model results.

    Args:
        results_dict: Dictionary mapping model names to results dictionaries, or list of results dicts

    Returns:
        DataFrame with comparison results
    """
    # Convert dict to list if needed
    if isinstance(results_dict, dict):
        results_list = list(results_dict.values())
    else:
        results_list = results_dict

    rows = []

    for results in results_list:
        row = {'Model': results['model_name'], 'Accuracy': results['accuracy']}

        # Add per-genre F1 scores
        for genre in GENRES:
            row[f'{genre.capitalize()} F1'] = results['per_genre'][genre]['f1']

        # Add macro average
        row['Macro F1'] = results['macro_avg']['f1']

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Convert to numpy arrays for consistent indexing
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Convert to genre names if numeric
    if isinstance(y_true_arr[0], (int, np.integer)):
        y_true_labels = [ID_TO_GENRE[int(y)] for y in y_true_arr]
        y_pred_labels = [ID_TO_GENRE[int(y)] for y in y_pred_arr]
    else:
        y_true_labels = y_true_arr.tolist()
        y_pred_labels = y_pred_arr.tolist()

    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(classification_report(y_true_labels, y_pred_labels, digits=4))
