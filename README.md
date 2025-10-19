# Genre Classification from Song Lyrics

A professional NLP research project comparing classical machine learning, word embeddings, and transformer models for music genre classification using the Genius Song Lyrics dataset.

## Project Overview

This project implements and compares three approaches for classifying song lyrics into genres:

1. **TF-IDF + Logistic Regression** - Keyword-based baseline (~61% accuracy)
2. **Word2Vec + Logistic Regression** - Semantic embeddings (~56% accuracy)
3. **BERT/DistilBERT** - Transformer-based contextual understanding (~61% accuracy)

**Key Finding:** Genre classification is primarily keyword-based. Simple TF-IDF performs competitively with complex BERT models at a fraction of the computational cost.

## Project Structure

```
.
├── src/                          # Core modules (all reusable code)
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading and preprocessing pipeline
│   ├── models.py                 # Model wrappers (TFIDFModel, Word2VecModel, BERTModel)
│   ├── evaluate.py               # Evaluation metrics and visualization
│   └── utils.py                  # Utilities, constants, helper functions
│
├── experiments/
│   └── configs/                  # YAML configuration files
│       ├── tfidf_config.yaml     # TF-IDF experiment config
│       ├── word2vec_config.yaml  # Word2Vec experiment config
│       └── bert_config.yaml      # BERT experiment config
│
├── scripts/
│   └── train.py                  # Command-line training script
│
├── notebooks/                    # Comprehensive, reproducible notebooks
│   ├── 01_data_pipeline.ipynb        # Data download, preprocessing, exploration
│   ├── 02_baseline_models.ipynb      # TF-IDF + classical ML baselines
│   ├── 03_advanced_models.ipynb      # Word2Vec and BERT models
│   ├── 04_evaluation_comparison.ipynb # Compare all models
│   └── archive/                      # Previous versions of notebooks
│
├── docs/                         # Documentation
│   ├── README.md                 # Original project documentation
│   ├── CLAUDE.md                 # Project context for Claude Code
│   └── [other docs...]
│
├── data/                         # Data files (gitignored)
│   ├── song_lyrics.csv           # Original dataset (~9GB)
│   ├── song_lyrics_cleaned.csv   # Cleaned data
│   └── song_lyrics_balanced.csv  # Balanced 5-genre dataset
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git configuration
└── README.md                     # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Place Kaggle API credentials in project root
# Download from https://www.kaggle.com/settings → API → Create New Token
chmod 600 kaggle.json
```

### 2. Download and Process Dataset

Use the data pipeline notebook to download, preprocess, and explore the data:

```bash
jupyter notebook notebooks/01_data_pipeline.ipynb
```

This notebook will:
- Download the dataset from Kaggle (requires `kaggle.json` in project root)
- Clean and preprocess lyrics
- Create balanced dataset across genres
- Perform exploratory data analysis
- Save processed data for modeling

### 3. Train Models

**Option A: Use interactive notebooks (recommended for learning and exploration)**

```bash
# Step 1: Train baseline models (TF-IDF variants)
jupyter notebook notebooks/02_baseline_models.ipynb

# Step 2: Train advanced models (Word2Vec, BERT)
jupyter notebook notebooks/03_advanced_models.ipynb

# Step 3: Compare all models
jupyter notebook notebooks/04_evaluation_comparison.ipynb
```

**Option B: Use command-line scripts (for reproducibility and batch processing)**

```bash
# Train TF-IDF model (fast, ~1 minute)
python scripts/train.py --config experiments/configs/tfidf_config.yaml

# Train Word2Vec model (medium, ~3 minutes)
python scripts/train.py --config experiments/configs/word2vec_config.yaml

# Train BERT model (slow, ~50 minutes with 20K samples/genre on GPU)
python scripts/train.py --config experiments/configs/bert_config.yaml
```

### 4. Notebook Workflow

The project includes 4 comprehensive notebooks that walk through the entire pipeline:

1. **`01_data_pipeline.ipynb`** - Download and preprocess data
   - Kaggle API setup and dataset download
   - Data exploration and statistics
   - Text cleaning and preprocessing
   - Dataset balancing across genres
   - Vocabulary analysis

2. **`02_baseline_models.ipynb`** - Train baseline models
   - TF-IDF + Logistic Regression
   - TF-IDF + Linear SVM
   - TF-IDF + Naive Bayes
   - Feature importance analysis
   - Performance comparison

3. **`03_advanced_models.ipynb`** - Train advanced models
   - Word2Vec + Logistic Regression
   - DistilBERT fine-tuning
   - Semantic exploration
   - Multi-GPU training

4. **`04_evaluation_comparison.ipynb`** - Final evaluation
   - Compare all 5 models
   - Per-genre performance analysis
   - Speed vs accuracy tradeoffs
   - Final recommendations

**Example code (from notebooks):**

```python
from src.data_loader import load_and_prepare_data
from src.models import TFIDFModel
from src.evaluate import evaluate_model, plot_confusion_matrix

# Load data (one line!)
X_train, X_test, y_train, y_test = load_and_prepare_data(
    samples_per_genre=5000
)

# Train model (two lines!)
model = TFIDFModel(classifier_type='logistic')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
results = evaluate_model(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)

print(f"Accuracy: {results['accuracy']:.2%}")
```

## Dataset

**Source:** [Genius Song Lyrics with Language Information](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)

**Original Dataset:**
- 5.1M song lyrics across 6 genres
- ~9GB CSV file
- Highly imbalanced (pop: 2.1M, rap: 1.7M)

**Preprocessed Dataset:**
- 5 genres: country, pop, rap, rb, rock (removed "misc")
- Balanced: equal samples per genre
- Cleaned: removed placeholders, normalized text

**Genres:**
- **Country** - Most distinctive features (truck, whiskey, cowboy)
- **Rap** - Strong vocabulary (rapper, flow, mc, explicit terms)
- **Rock** - Moderate features (guitar, punk, explicit terms)
- **R&B** - Overlaps with rap (love, soul themes)
- **Pop** - Least distinctive (generic language, hardest to classify)

## Model Descriptions

### 1. TF-IDF Model

**Implementation:** `src/models.py` → `TFIDFModel`

**Approach:**
- TF-IDF vectorization with unigrams and bigrams
- Logistic Regression classifier
- 10,000 feature limit

**Performance:**
- Accuracy: ~61%
- Training time: < 1 minute
- Best for: country, rap, rock

**Strengths:**
- Fast training and inference
- Interpretable (can examine feature importance)
- Surprisingly competitive with deep learning

### 2. Word2Vec Model

**Implementation:** `src/models.py` → `Word2VecModel`

**Approach:**
- Train Word2Vec embeddings on lyrics corpus
- Average word vectors to get document representation
- Logistic Regression classifier

**Performance:**
- Accuracy: ~56%
- Training time: 2-3 minutes
- Underperforms TF-IDF

**Why it underperforms:**
- Genre classification needs distinctive keywords, not semantic similarity
- Averaging vectors loses important signals
- Example: "love" is semantically similar across all genres but not discriminative

### 3. BERT Model

**Implementation:** `src/models.py` → `BERTModel`

**Approach:**
- Fine-tune DistilBERT multilingual model
- Sequence classification with 5 genre classes
- Multi-GPU support with DataParallel
- Mixed precision training (FP16) for speed

**Performance:**
- Accuracy: ~61% (competitive with TF-IDF)
- Training time: ~50 minutes (20K samples/genre, 8x A16 GPUs)
- Prone to overfitting after epoch 3

**Optimizations:**
- Multi-GPU training with PyTorch DataParallel
- Automatic Mixed Precision (2x speedup)
- Dynamic padding (memory efficient)
- Gradient accumulation support

**Recommended settings:**
- `max_length=256` (optimal speed/performance tradeoff)
- `epochs=5` with early stopping
- `batch_size=96` (total across all GPUs)
- `learning_rate=2e-5`
- `weight_decay=0.1` (prevent overfitting)

## Configuration System

All experiments use YAML configuration files for reproducibility:

```yaml
# experiments/configs/tfidf_config.yaml
model:
  type: "tfidf"
  classifier: "logistic"
  max_features: 10000
  ngram_range: [1, 2]

data:
  samples_per_genre: 20000
  test_size: 0.2
  random_state: 42

experiment:
  name: "tfidf_baseline"
  save_results: true
```

Modify configs to experiment with different hyperparameters without changing code.

## Module Documentation

### `src/data_loader.py`

**Functions:**
- `load_raw_data()` - Load original CSV
- `clean_lyrics()` - Text preprocessing
- `preprocess_data()` - Remove placeholders, filter genres
- `balance_dataset()` - Create balanced dataset
- `load_and_prepare_data()` - **Main entry point** - Complete pipeline

### `src/models.py`

**Classes:**
- `TFIDFModel` - TF-IDF + classifier wrapper
- `Word2VecModel` - Word2Vec + classifier wrapper
- `BERTModel` - BERT/DistilBERT fine-tuning wrapper
- `LyricsDataset` - PyTorch Dataset for BERT

**Unified Interface:**
All models implement:
- `fit(X_train, y_train)` - Train the model
- `predict(X_test)` - Make predictions

### `src/evaluate.py`

**Functions:**
- `evaluate_model()` - Comprehensive metrics (accuracy, precision, recall, F1)
- `plot_confusion_matrix()` - Visualize confusion matrix
- `plot_per_genre_metrics()` - Per-genre bar charts
- `compare_models()` - Side-by-side model comparison
- `create_results_table()` - Tabular comparison

### `src/utils.py`

**Constants:**
- `GENRES` - List of 5 genres
- `GENRE_TO_ID`, `ID_TO_GENRE` - Label mappings

**Functions:**
- `get_device()` - Get GPU/CPU device
- `get_gpu_info()` - GPU information
- `set_seed()` - Set random seeds for reproducibility
- `save_results()` - Save results to JSON

## Results Summary

Based on experiments with 20,000 samples per genre:

| Model      | Accuracy | Macro F1 | Training Time | Hardware    |
|------------|----------|----------|---------------|-------------|
| TF-IDF     | 60.93%   | 0.605    | < 1 min       | CPU         |
| Word2Vec   | 55.86%   | 0.543    | 2-3 min       | CPU         |
| BERT       | 60.89%   | 0.607    | ~50 min       | 8x GPU      |

**Per-Genre Performance (TF-IDF):**

| Genre   | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Rap     | 0.82      | 0.74   | 0.78     |
| Country | 0.70      | 0.67   | 0.68     |
| R&B     | 0.64      | 0.60   | 0.62     |
| Rock    | 0.60      | 0.54   | 0.57     |
| Pop     | 0.47      | 0.32   | 0.38     |

**Key Insights:**
- Pop genre is consistently hardest to classify (too generic)
- Rap has most distinctive features (highest F1)
- BERT's complexity doesn't justify computational cost for this task
- TF-IDF is the recommended baseline

## Common Issues

### Out of Memory (BERT)

**Problem:** `CUDA out of memory` error during BERT training

**Solutions:**
1. Reduce `max_length` (512 → 256)
2. Reduce `batch_size` (96 → 48)
3. Reduce `samples_per_genre` (20000 → 5000)
4. Use fewer GPUs (set `CUDA_VISIBLE_DEVICES=0,1`)

### Slow Training (BERT)

**Problem:** Training slower than expected

**Check:**
1. GPU utilization: `nvidia-smi`
2. Mixed precision enabled: `use_amp: true` in config
3. Sequence length: 256 is optimal (512 is 2x slower)
4. Multi-GPU working: Should see multiple GPU processes

### BERT Overfitting

**Problem:** Accuracy peaks early then degrades

**Solutions:**
1. Use early stopping (monitor epoch 3 results)
2. Increase `weight_decay` (0.01 → 0.1)
3. Reduce `learning_rate` (3e-5 → 2e-5)
4. Train with more data (`samples_per_genre: 50000`)

## Development

### Adding a New Model

1. Create model class in `src/models.py`:
```python
class NewModel:
    def __init__(self, **params):
        # Initialize model
        pass

    def fit(self, X_train, y_train):
        # Training logic
        pass

    def predict(self, X_test):
        # Prediction logic
        return predictions
```

2. Add config file in `experiments/configs/new_model_config.yaml`

3. Add training function in `scripts/train.py`:
```python
def train_new_model(config, X_train, y_train, X_test, y_test):
    model = NewModel(**config['model'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
```

### Running Tests

```bash
# TODO: Add test suite
# python -m pytest tests/
```

## Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 8GB
- Disk: 20GB free space
- For TF-IDF and Word2Vec only

**Recommended (for BERT):**
- GPU: NVIDIA GPU with 12GB+ VRAM (e.g., A16, V100, RTX 3090)
- RAM: 16GB+
- CUDA: 11.0+
- PyTorch with CUDA support

**Current testing environment:**
- 8x NVIDIA A16 GPUs (16GB each)
- CUDA 11.8
- PyTorch 2.0+ with DataParallel

## Citation

```bibtex
@dataset{genius_lyrics_2023,
  title={Genius Song Lyrics with Language Information},
  author={Carlos Gomes},
  year={2023},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information}
}
```

## License

This project is for educational and research purposes.

## Contributing

For questions or contributions, please:
1. Check existing documentation in `docs/`
2. Review configuration files in `experiments/configs/`
3. Examine notebooks for usage examples

## Reproducible Research Pipeline

This project is designed for complete reproducibility. A user cloning from GitHub can reproduce all results by running the notebooks in order:

**Complete Pipeline (Start from scratch):**

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download and preprocess data (~15 min)
jupyter notebook notebooks/01_data_pipeline.ipynb

# 3. Train baseline models (~5 min)
jupyter notebook notebooks/02_baseline_models.ipynb

# 4. Train advanced models (~60 min with GPU)
jupyter notebook notebooks/03_advanced_models.ipynb

# 5. Compare all results
jupyter notebook notebooks/04_evaluation_comparison.ipynb
```

Each notebook:
- ✓ Uses modular `src/` code (clean and reusable)
- ✓ Includes detailed markdown explanations
- ✓ Shows intermediate results and visualizations
- ✓ Saves results for later comparison
- ✓ Can be run independently (after data preprocessing)

## Next Steps

1. **Error Analysis**: Examine specific misclassified examples
2. **Hyperparameter Tuning**: Modify YAML configs and experiment
3. **Feature Engineering**: Add metadata (artist, year) to improve predictions
4. **Ensemble Methods**: Combine TF-IDF and BERT predictions
5. **Production Deployment**: Package TF-IDF model for serving

## Acknowledgments

This project structure prioritizes:
- **Modularity**: All logic in `src/` modules
- **Reproducibility**: YAML configs, seed setting, versioned notebooks
- **Usability**: Simple notebooks, command-line scripts, clear documentation
- **Scalability**: Multi-GPU support, efficient data pipelines
