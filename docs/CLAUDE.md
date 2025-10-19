# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Complete NLP research project** comparing multiple approaches for song lyrics genre classification using the Genius Song Lyrics dataset from Kaggle.

**Status:** ✅ Complete - All experiments finished, results documented

**Final Results:**
- Best overall model: **TF-IDF + Logistic Regression (60.93% accuracy)**
- Word2Vec performed worse (55.86% accuracy) - semantic understanding not beneficial
- BERT competitive (60.89% accuracy) - context doesn't significantly help
- **Key insight:** Genre classification is keyword-based, not semantic

## Dataset Information

**Source:** [Genius Song Lyrics with Language Information](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)

**Original dataset:**
- 5,134,856 rows, 11 columns, ~9GB
- 6 genres: pop, rap, rock, rb, misc, country
- Severe class imbalance (pop: 2.1M, rap: 1.7M)
- Multilingual (74.6% English)
- Contains placeholder text and quality issues

**Preprocessed dataset:**
- 494,988 rows after cleaning and balancing
- 5 genres (removed "misc" - not a real genre)
- 100K samples per genre (except country: 94,988)
- Removed placeholder text, cleaned lyrics
- Files: `data/song_lyrics_cleaned.csv`, `data/song_lyrics_balanced.csv`

## Complete Experimental Pipeline

### 1. Setup & Exploration
**Notebooks:** `setup.ipynb`, `data_exploration.ipynb`

**Purpose:** Download dataset, initial exploration, understand data quality

**Key findings:**
- High class imbalance
- 96K placeholder rows
- Mixed languages
- Variable lyrics length (avg ~278 words)

---

### 2. Data Preprocessing
**Notebook:** `preprocessing.ipynb`

**What it does:**
1. Removes placeholder text (96,682 rows)
2. Text cleaning (lowercase, remove special chars, strip markers)
3. Creates balanced dataset (100K per genre)
4. Removes "misc" category
5. Analyzes genre characteristics

**Output files:**
- `data/song_lyrics_cleaned.csv` - Full cleaned (4.9M rows)
- `data/song_lyrics_balanced.csv` - Balanced 5 genres (495K rows)

**Key stats by genre:**
| Genre   | Avg Words | Vocab Diversity |
|---------|-----------|-----------------|
| Rap     | 414       | 0.46           |
| R&B     | 307       | 0.38           |
| Country | 234       | 0.43           |
| Pop     | 231       | 0.43           |
| Rock    | 202       | 0.47           |

---

### 3. Baseline Models (TF-IDF)
**Notebook:** `baseline_models.ipynb`

**Models tested:**
- Logistic Regression: **60.93% accuracy** ⭐ Best
- Linear SVM: 60.66% accuracy
- Naive Bayes: 52.62% accuracy

**TF-IDF parameters:**
- max_features: 10,000
- ngram_range: (1, 2)
- min_df: 5, max_df: 0.8

**Per-genre F1 scores (Logistic Regression):**
| Genre   | F1 Score | Performance |
|---------|----------|-------------|
| Rap     | 0.7786   | Excellent   |
| Country | 0.6844   | Good        |
| R&B     | 0.6195   | Good        |
| Rock    | 0.5651   | Moderate    |
| Pop     | 0.3813   | Poor ⚠️     |

**Key features by genre:**
- Country: truck, whiskey, cowboy, beer, Tennessee
- Rap: rap, rapper, fuck, shit, flow, mc
- Rock: fucking, guitar, punk, metal
- Pop: chorus, cos, generic pop terms (not distinctive)

---

### 4. Word Embeddings (Word2Vec)
**Notebook:** `word_embeddings.ipynb`

**Approach:**
- Trained Word2Vec (200D, CBOW, 10 epochs)
- Vocabulary: 236,541 words
- Document vectors: averaged word embeddings

**Results:**
- Logistic Regression: **55.86% accuracy** (-8.3% vs TF-IDF)
- Linear SVM: 54.99% accuracy

**Why it underperformed:**
- Genre classification needs distinctive vocabulary, not semantic similarity
- Averaging word vectors loses important keyword signals
- Example: "love" semantically similar across all genres

**Learned meaningful relationships:**
- guitar → banjo, harmonica, fiddle
- rap → freestyle, hiphop, mumble
- pain → sadness, sorrow, anguish

---

### 5. Error Analysis
**Notebook:** `error_analysis.ipynb`

**Deep dive into misclassifications:**

**Pop genre problems:**
- Only 34.8% recall (worst by far)
- 43.5% misclassified as Rock
- 23.9% misclassified as R&B
- Too generic/diverse, lacks distinctive markers

**Language analysis:**
- English: 62.0% accuracy
- Non-English: 57.8% accuracy
- Mixing languages dilutes genre signals

**Common confusions:**
- Pop ↔ Rock (49.2% of Rock errors → Pop)
- Rap ↔ R&B (51.5% of Rap errors → R&B, expected overlap)

**Insights:**
1. Pop is the major problem (no distinctive features)
2. Multilingual data slightly hurts performance
3. Some genre boundaries are inherently fuzzy
4. Keyword approach (TF-IDF) is appropriate for this task

---

### 6. English-Only Experiment
**Notebook:** `english_only_experiment.ipynb`

**Hypothesis:** Filtering to English-only improves performance

**Results:**
- Overall accuracy: 61.90% (+1.6% improvement)
- Pop F1: 0.2858 (-25% worse!)
- Dataset reduced to 369,362 songs (74.6% of original)

**Per-genre changes:**
- Country: +3.9%
- Rock: +6.3%
- R&B: +5.5%
- Rap: +1.1%
- Pop: **-25%** (much worse)

**Conclusion:**
- Small overall improvement not worth losing Pop samples
- Pop in English even harder to classify (too generic)
- Decided to use full multilingual dataset for BERT

---

### 7. BERT Classification
**Notebook:** `bert_classification.ipynb`

**Approach:**
- Model: `bert-base-multilingual-cased`
- Sampled 50K songs (10K per genre) for computational efficiency
- Fine-tuned for 3 epochs
- Max length: 256 tokens

**Results:**
- Test accuracy: **60.89%** (competitive with TF-IDF)
- Training time: ~45 minutes per epoch (GPU)

**Per-genre F1 scores:**
| Genre   | TF-IDF | BERT   | Winner  |
|---------|--------|--------|---------|
| Country | 0.6844 | 0.6975 | BERT    |
| Rock    | 0.5651 | 0.5778 | BERT    |
| R&B     | 0.6195 | 0.5989 | TF-IDF  |
| Rap     | 0.7786 | 0.7761 | TF-IDF  |
| Pop     | 0.3813 | 0.3675 | TF-IDF  |

**Insights:**
- BERT helps Country and Rock (contextual understanding)
- BERT doesn't fix Pop problem
- Contextualized embeddings not significantly better than keywords
- Simple TF-IDF remains highly competitive

---

## Project File Structure

### Notebooks (in execution order)
```
setup.ipynb                     # Initial setup, download dataset
data_exploration.ipynb          # Explore raw data
preprocessing.ipynb             # Clean and balance data
baseline_models.ipynb           # TF-IDF + classical ML
word_embeddings.ipynb           # Word2Vec experiments
error_analysis.ipynb            # Analyze misclassifications
english_only_experiment.ipynb   # Test English-only filtering
bert_classification.ipynb       # BERT fine-tuning
gpu_test.ipynb                  # GPU availability test
```

### Data Files
```
data/
├── song_lyrics.csv              # Original (5.1M rows, 9GB) - gitignored
├── song_lyrics_cleaned.csv      # Cleaned (4.9M rows) - gitignored
└── song_lyrics_balanced.csv     # Balanced 5 genres (495K rows) - gitignored
```

### Documentation
```
README.md                        # Project overview
CLAUDE.md                        # This file
data_exp.md                      # Exploration summary
data_exploration.txt             # Full exploration report
NLP_ProjectProposal.pdf          # Project proposal
```

### Configuration
```
kaggle.json                      # API credentials - gitignored
.gitignore                       # Allow-list approach
```

## Key Research Findings

### 1. Genre Classification is Keyword-Based
- TF-IDF (keywords) outperforms Word2Vec (semantics)
- Distinctive vocabulary > semantic understanding
- Rap uses "rapper, flow, mc"; Country uses "truck, whiskey, cowboy"

### 2. Pop Genre is Problematic
- Lowest F1 across all methods (0.29-0.38)
- Too diverse/generic, borrows from all genres
- No distinctive linguistic markers
- Gets confused with everything (43% → Rock, 24% → R&B)

### 3. Simple Methods Work Well
- TF-IDF + Logistic Regression achieves 60.93%
- BERT's complexity doesn't provide significant gains
- Word2Vec actively hurts performance (-8%)
- Computational cost vs benefit analysis favors TF-IDF

### 4. Genre Boundaries are Fuzzy
- Rap ↔ R&B confusion (expected, similar styles)
- Rock ↔ Pop overlap (pop-rock is a real subgenre)
- Some misclassifications are arguably correct

### 5. Multilingual Data Impact
- English: 62.0% accuracy
- Non-English: 57.8% accuracy
- Impact modest (~4% difference)
- BERT multilingual handles mixed languages well

## Best Practices & Recommendations

### For This Dataset

**Use TF-IDF baseline:**
- Fast, interpretable, competitive performance
- 60.93% accuracy with Logistic Regression
- Feature importance analysis reveals genre characteristics

**Handle Pop genre carefully:**
- Consider removing it (catch-all category)
- Or treat as binary "Pop vs Not Pop"
- Or merge into supergenres (e.g., Pop-Rock, Pop-RB)

**Stick with multilingual:**
- English-only provides minimal benefit (+1.6%)
- Loses 25% of data
- BERT multilingual handles mixed languages

### For Future Work

**Model improvements:**
1. Hierarchical classification (Rock → Pop-Rock, Hard Rock, etc.)
2. Multi-label classification (songs can be multiple genres)
3. Ensemble methods (combine TF-IDF + BERT predictions)
4. Genre-specific binary classifiers

**Feature engineering:**
1. TF-IDF weighted Word2Vec averaging
2. Combine lyrics + metadata (artist, year, views)
3. Audio features (if available)
4. Longer BERT sequences (512 tokens)

**Dataset modifications:**
1. Remove or redefine Pop
2. Create genre hierarchies
3. Manual quality checks on ambiguous songs
4. Balance by sampling strategy (not just size)

## How to Reproduce Results

### Initial Setup
```bash
# 1. Place kaggle.json in project root
chmod 600 kaggle.json

# 2. Run setup notebook
jupyter notebook setup.ipynb
# Downloads ~9GB dataset, takes time
```

### Run Full Pipeline
```bash
# 3. Preprocess data
jupyter notebook preprocessing.ipynb
# Output: data/song_lyrics_balanced.csv

# 4. Baseline models
jupyter notebook baseline_models.ipynb
# Best: TF-IDF + LogReg, 60.93% accuracy

# 5. Word embeddings
jupyter notebook word_embeddings.ipynb
# Word2Vec: 55.86% accuracy

# 6. Error analysis
jupyter notebook error_analysis.ipynb
# Identifies Pop as main problem

# 7. English-only test
jupyter notebook english_only_experiment.ipynb
# +1.6% improvement, but Pop gets worse

# 8. BERT (requires GPU)
jupyter notebook bert_classification.ipynb
# 60.89% accuracy, competitive with TF-IDF
```

### Quick Start (Use Preprocessed Data)
If you already have `data/song_lyrics_balanced.csv`:
```bash
# Jump straight to modeling
jupyter notebook baseline_models.ipynb
```

## Environment & Dependencies

**Python packages:**
- pandas, numpy
- scikit-learn
- gensim (Word2Vec)
- torch, transformers (BERT)
- matplotlib, seaborn
- kaggle (for download)

**Hardware:**
- CPU sufficient for TF-IDF and Word2Vec
- GPU recommended for BERT (CUDA compatible)
- 16GB+ RAM recommended for full dataset
- ~20GB disk space for data files

**Working directory:** `/home/jovyan/Desktop/NLP/Project`

## Git Configuration

**.gitignore strategy:**
- Allow-list approach (ignore `*`, explicitly allow files)
- Explicitly tracked: All `.ipynb` notebooks, `README.md`, `CLAUDE.md`, `data_exp.md`
- Ignored: Data files (too large), credentials, outputs, checkpoints

**To add new notebook to git:**
```bash
echo "!new_notebook.ipynb" >> .gitignore
git add new_notebook.ipynb
```

## Summary

This project demonstrates a complete NLP research pipeline comparing traditional ML (TF-IDF), word embeddings (Word2Vec), and deep learning (BERT) for genre classification.

**Main takeaway:** Simple keyword-based methods (TF-IDF) are highly competitive for genre classification. The task benefits from distinctive vocabulary rather than semantic or contextual understanding.

**Research contribution:** Comprehensive empirical comparison showing when and why complex models may not outperform simple baselines for specific NLP tasks.

**Status:** Project complete and ready for writeup/presentation.
