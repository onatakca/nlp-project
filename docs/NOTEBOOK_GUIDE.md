# Notebook Guide

This guide explains the new notebook structure and how to use them.

## Overview

The project now includes **4 comprehensive, self-contained notebooks** that allow anyone cloning from GitHub to reproduce the entire research pipeline from scratch.

## Notebook Structure

### 1. `01_data_pipeline.ipynb` - Data Download, Preprocessing & Exploration

**Purpose:** Complete data pipeline from Kaggle download to preprocessed dataset

**What it does:**
- Kaggle API setup and authentication
- Downloads Genius Song Lyrics dataset (~9GB)
- Loads and explores raw data (5.1M songs)
- Removes placeholder lyrics (e.g., "Instrumental", "Lyrics coming soon")
- Cleans text (lowercase, remove markers, normalize whitespace)
- Filters to 5 genres (removes "misc")
- Creates balanced dataset (equal samples per genre)
- Analyzes vocabulary and genre characteristics
- Saves preprocessed data to `data/song_lyrics_balanced.csv`

**Outputs:**
- `data/song_lyrics_balanced.csv` (ready for modeling)
- Data exploration visualizations
- Genre statistics

**Runtime:** ~15 minutes (mostly download time)

---

### 2. `02_baseline_models.ipynb` - TF-IDF + Classical ML

**Purpose:** Train and evaluate baseline models using TF-IDF features

**What it does:**
- Loads preprocessed data
- Trains TF-IDF + Logistic Regression
- Trains TF-IDF + Linear SVM
- Trains TF-IDF + Naive Bayes
- Evaluates each model (accuracy, F1, confusion matrices)
- Analyzes feature importance (top keywords per genre)
- Compares baseline approaches
- Saves results to `experiments/results/baseline_models_results.json`

**Key Findings:**
- TF-IDF + Logistic Regression is the best baseline (~61% accuracy)
- Training time: < 1 minute
- Genre classification is keyword-based
- Distinctive features: "truck"→country, "flow"→rap, "guitar"→rock

**Runtime:** ~5 minutes

---

### 3. `03_advanced_models.ipynb` - Word2Vec and BERT

**Purpose:** Train and evaluate advanced NLP models

**What it does:**
- Checks GPU availability
- Trains Word2Vec + Logistic Regression (~3 min)
- Explores semantic similarities learned by Word2Vec
- Trains DistilBERT with fine-tuning (~50 min on multi-GPU)
- Evaluates both models
- Compares with baselines
- Saves results to `experiments/results/advanced_models_results.json`

**Key Findings:**
- Word2Vec underperforms TF-IDF (~56% accuracy)
- BERT matches TF-IDF performance (~61% accuracy)
- Semantic understanding doesn't help this task
- 50x longer training time for BERT vs TF-IDF

**Hardware Requirements:**
- Word2Vec: CPU (2-3 min)
- BERT: Multi-GPU recommended (50+ min on 8x A16 GPUs)

**Runtime:** ~60 minutes (mostly BERT training)

---

### 4. `04_evaluation_comparison.ipynb` - Final Comparison

**Purpose:** Comprehensive comparison of all models

**What it does:**
- Loads saved results from previous notebooks
- Compares all 5 models side-by-side
- Analyzes per-genre performance
- Visualizes speed vs accuracy tradeoffs
- Examines confusion patterns
- Provides final recommendations

**Analyses:**
- Overall performance comparison (accuracy, F1, precision, recall)
- Per-genre strengths and weaknesses
- Training time comparison
- Best model identification
- Production deployment recommendations

**Key Insight:**
> **Simple is better.** TF-IDF + Logistic Regression provides the best speed/accuracy tradeoff for this task.

**Runtime:** ~2 minutes (just loads and visualizes results)

---

## How to Use

### Option 1: Run All Notebooks (Full Pipeline)

Start from scratch and reproduce everything:

```bash
# 1. Download and preprocess data
jupyter notebook notebooks/01_data_pipeline.ipynb

# 2. Train baseline models
jupyter notebook notebooks/02_baseline_models.ipynb

# 3. Train advanced models
jupyter notebook notebooks/03_advanced_models.ipynb

# 4. Compare all models
jupyter notebook notebooks/04_evaluation_comparison.ipynb
```

**Total runtime:** ~80 minutes (with GPU for BERT)

### Option 2: Use Existing Data

If `data/song_lyrics_balanced.csv` already exists, skip to notebooks 02-04.

### Option 3: Compare Models Only

If you've already run notebooks 02 and 03, just run notebook 04 to see the comparison.

---

## Notebook Features

Each notebook includes:

✓ **Modular code** - Uses `src/` modules for clean, reusable code
✓ **Detailed explanations** - Markdown cells explain every step
✓ **Visualizations** - Plots and charts for data exploration
✓ **Intermediate results** - Print statements show progress
✓ **Saved outputs** - Results saved to disk for later use
✓ **Reproducibility** - Fixed random seeds, clear configurations

---

## Expected Results

Based on 20,000 samples per genre:

| Model                           | Accuracy | Macro F1 | Training Time |
|---------------------------------|----------|----------|---------------|
| TF-IDF + Logistic Regression    | 60.93%   | 0.605    | < 1 min       |
| TF-IDF + Linear SVM             | ~60%     | ~0.60    | ~1 min        |
| TF-IDF + Naive Bayes            | ~58%     | ~0.58    | < 1 min       |
| Word2Vec + Logistic Regression  | 55.86%   | 0.543    | 2-3 min       |
| DistilBERT                      | 60.89%   | 0.607    | ~50 min       |

**Winner:** TF-IDF + Logistic Regression (best speed/accuracy tradeoff)

---

## Configuration

You can adjust these parameters in each notebook:

**`01_data_pipeline.ipynb`:**
- `SAMPLES_PER_GENRE` - Number of samples per genre (default: 100,000)

**`02_baseline_models.ipynb`:**
- `SAMPLES_PER_GENRE` - Number of samples for training (default: 20,000)
- `TEST_SIZE` - Test set fraction (default: 0.2)

**`03_advanced_models.ipynb`:**
- `SAMPLES_PER_GENRE` - Number of samples for training (default: 20,000)
- BERT config: `max_length`, `batch_size`, `epochs`, etc.

**Recommendations:**
- **Quick testing:** 5,000 samples/genre (~25K total)
- **Standard:** 20,000 samples/genre (~100K total)
- **Full dataset:** 100,000 samples/genre (~500K total)

---

## Troubleshooting

### Kaggle Download Issues
**Problem:** Can't download dataset

**Solution:**
1. Get `kaggle.json` from https://www.kaggle.com/settings → API
2. Place in project root
3. Run: `chmod 600 kaggle.json`

### CUDA Out of Memory (BERT)
**Problem:** GPU memory error

**Solutions:**
1. Reduce `SAMPLES_PER_GENRE` (20000 → 5000)
2. Reduce `max_length` (256 → 128)
3. Reduce `batch_size` (96 → 48)

### Slow BERT Training
**Problem:** Training slower than expected

**Check:**
1. GPU utilization: `nvidia-smi`
2. Multi-GPU: Should use all available GPUs
3. Mixed precision: Enabled by default (`use_amp=True`)

---

## Archive

Old notebooks from previous refactor are in `notebooks/archive/`:
- `01_quick_start.ipynb` - Previous quick start guide
- `02_compare_models.ipynb` - Previous comparison notebook
- `bert_best_config.ipynb` - BERT hyperparameter experiments

These are kept for reference but not part of the main workflow.

---

## Next Steps After Running Notebooks

1. **Error Analysis** - Examine specific misclassified examples
2. **Hyperparameter Tuning** - Experiment with different configurations
3. **Feature Engineering** - Add artist/year metadata
4. **Ensemble Methods** - Combine predictions from multiple models
5. **Production Deployment** - Package best model for serving
