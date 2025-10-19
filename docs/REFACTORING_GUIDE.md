# Project Refactoring Guide

This document explains the major refactoring completed on 2025-10-18 to transform the project from a notebook-heavy workflow to a professional ML research project structure.

## What Changed?

### Before Refactoring

**Problems:**
- All logic embedded in notebooks (data loading, model training, evaluation)
- Code duplication across multiple notebooks
- Hard to reproduce experiments
- Difficult to modify and test
- No clear separation of concerns
- Hard to run experiments from command line

**Structure:**
```
.
├── *.ipynb (10+ notebooks with duplicated code)
├── data/
├── kaggle.json
└── README.md
```

### After Refactoring

**Benefits:**
- All reusable code in `src/` Python modules
- Clean notebooks that just import and use modules
- YAML configuration for reproducible experiments
- Command-line scripts for automation
- Professional structure ready for publication
- Easy to test and modify

**New Structure:**
```
.
├── src/                    # All reusable code here!
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluate.py
│   └── utils.py
├── experiments/configs/    # YAML configs for experiments
├── scripts/train.py        # Command-line training
├── notebooks/              # Clean notebooks using src/
│   ├── 01_quick_start.ipynb
│   └── 02_compare_models.ipynb
├── docs/                   # All documentation
└── README.md               # New comprehensive README
```

## Migration Guide

### Old Way vs New Way

#### Data Loading

**OLD (every notebook):**
```python
# 50+ lines of code in each notebook:
df = pd.read_csv('data/song_lyrics.csv')
df = df[df['tag'].isin(GENRES)]
df = df[~df['lyrics'].str.contains("Tell us that you would")]
df['lyrics'] = df['lyrics'].str.lower()
# ... many more lines ...
X_train, X_test, y_train, y_test = train_test_split(...)
```

**NEW (one line):**
```python
from src.data_loader import load_and_prepare_data

X_train, X_test, y_train, y_test = load_and_prepare_data(
    samples_per_genre=20000
)
```

---

#### Training TF-IDF Model

**OLD (every notebook):**
```python
# 30+ lines in each notebook:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
```

**NEW (3 lines):**
```python
from src.models import TFIDFModel

model = TFIDFModel(classifier_type='logistic')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

#### Evaluation

**OLD (every notebook):**
```python
# 40+ lines in each notebook:
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred, output_dict=True)
# ... manual plotting code ...
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
# ... more plotting code ...
```

**NEW (3 lines):**
```python
from src.evaluate import evaluate_model, plot_confusion_matrix

results = evaluate_model(y_test, y_pred, model_name="TF-IDF")
plot_confusion_matrix(y_test, y_pred, title="TF-IDF Results")
```

---

#### Running Experiments

**OLD:**
- Open notebook
- Manually modify hyperparameters in code
- Run cells one by one
- Hard to reproduce exact configuration

**NEW (command line):**
```bash
# Just run with a config file!
python scripts/train.py --config experiments/configs/tfidf_config.yaml
```

**NEW (notebook):**
```python
# Or use the clean notebooks
jupyter notebook notebooks/01_quick_start.ipynb
```

## What's in Each Module?

### `src/data_loader.py`

**Purpose:** All data loading and preprocessing logic

**Key Function:**
```python
load_and_prepare_data(
    samples_per_genre=20000,
    test_size=0.2,
    use_cached=True,
    random_state=42
)
```

**What it does:**
1. Loads raw CSV
2. Removes placeholder text
3. Cleans lyrics (lowercase, strip markers)
4. Filters to 5 genres
5. Creates balanced dataset
6. Splits into train/test
7. Returns ready-to-use data

### `src/models.py`

**Purpose:** Model wrappers with unified interface

**Classes:**

1. **TFIDFModel** - TF-IDF + classifier
   ```python
   model = TFIDFModel(
       classifier_type='logistic',  # or 'svm', 'naive_bayes'
       max_features=10000,
       ngram_range=(1, 2)
   )
   ```

2. **Word2VecModel** - Word2Vec + classifier
   ```python
   model = Word2VecModel(
       classifier_type='logistic',
       vector_size=200,
       window=5,
       epochs=10
   )
   ```

3. **BERTModel** - BERT/DistilBERT fine-tuning
   ```python
   model = BERTModel(
       model_name='distilbert-base-multilingual-cased',
       max_length=256,
       batch_size=96,
       learning_rate=2e-5,
       epochs=5
   )
   ```

**Unified Interface:**
All models have:
- `fit(X_train, y_train)` - Train
- `predict(X_test)` - Predict

### `src/evaluate.py`

**Purpose:** Evaluation and visualization

**Functions:**
- `evaluate_model()` - Get metrics (accuracy, precision, recall, F1)
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_per_genre_metrics()` - Bar charts per genre
- `compare_models()` - Side-by-side comparison
- `create_results_table()` - Tabular comparison

### `src/utils.py`

**Purpose:** Shared utilities and constants

**Constants:**
- `GENRES = ['country', 'pop', 'rap', 'rb', 'rock']`
- `GENRE_TO_ID`, `ID_TO_GENRE` - Label mappings
- `DATA_DIR`, `RESULTS_DIR` - Paths

**Functions:**
- `get_device()` - Get GPU/CPU
- `get_gpu_info()` - GPU information
- `set_seed()` - Set random seeds
- `save_results()` - Save to JSON

## New Workflows

### Workflow 1: Quick Experimentation (Notebook)

```bash
# Open clean notebook
jupyter notebook notebooks/01_quick_start.ipynb
```

The notebook now has minimal code:
```python
# Import
from src.data_loader import load_and_prepare_data
from src.models import TFIDFModel
from src.evaluate import evaluate_model

# Load (1 line)
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Train (2 lines)
model = TFIDFModel()
model.fit(X_train, y_train)

# Evaluate (2 lines)
y_pred = model.predict(X_test)
results = evaluate_model(y_test, y_pred)
```

### Workflow 2: Reproducible Experiments (Command Line)

```bash
# Edit config file
nano experiments/configs/tfidf_config.yaml

# Run experiment
python scripts/train.py --config experiments/configs/tfidf_config.yaml

# Results are automatically saved!
```

### Workflow 3: Compare All Models

```bash
# Use the comparison notebook
jupyter notebook notebooks/02_compare_models.ipynb
```

Or run all three from command line:
```bash
python scripts/train.py --config experiments/configs/tfidf_config.yaml
python scripts/train.py --config experiments/configs/word2vec_config.yaml
python scripts/train.py --config experiments/configs/bert_config.yaml
```

## Configuration Files

All experiments now use YAML configs in `experiments/configs/`:

**Example: `tfidf_config.yaml`**
```yaml
model:
  type: "tfidf"
  classifier: "logistic"
  max_features: 10000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.8

data:
  samples_per_genre: 20000
  test_size: 0.2
  random_state: 42
  use_cached: true

experiment:
  name: "tfidf_baseline"
  description: "TF-IDF + Logistic Regression baseline"
  save_results: true
```

**Benefits:**
- Reproducibility: Exact parameters documented
- Easy tuning: Modify configs without touching code
- Version control: Track experiment history
- Sharing: Send config to collaborators

## What to Do with Old Notebooks?

Your old notebooks have been moved to `notebooks/` folder:
- `baseline_models.ipynb`
- `word_embeddings.ipynb`
- `bert_classification.ipynb`
- `error_analysis.ipynb`
- etc.

**Options:**

1. **Keep as reference** - They document your research journey

2. **Refactor gradually** - Update them to use new modules:
   ```python
   # Add at top of old notebook:
   import sys
   sys.path.insert(0, '..')
   from src.data_loader import load_and_prepare_data
   # Then replace old code with new imports
   ```

3. **Archive** - Move to `notebooks/archive/` if you want to clean up

**Recommendation:** Keep them for now. They contain valuable analysis and visualizations.

## File Organization

### What's Where?

```
Project Root
│
├── src/                           # SOURCE CODE (import from here)
│   ├── __init__.py                # Makes src/ a package
│   ├── data_loader.py             # Data pipeline
│   ├── models.py                  # Model implementations
│   ├── evaluate.py                # Evaluation & visualization
│   └── utils.py                   # Shared utilities
│
├── experiments/
│   └── configs/                   # EXPERIMENT CONFIGS (edit these)
│       ├── tfidf_config.yaml
│       ├── word2vec_config.yaml
│       └── bert_config.yaml
│
├── scripts/
│   └── train.py                   # TRAINING SCRIPT (run this)
│
├── notebooks/                     # NOTEBOOKS (run experiments here)
│   ├── 01_quick_start.ipynb       # NEW: Simple intro
│   ├── 02_compare_models.ipynb    # NEW: Compare all models
│   ├── baseline_models.ipynb      # OLD: Original research
│   ├── bert_classification.ipynb  # OLD: Original research
│   └── ...                        # Other old notebooks
│
├── docs/                          # DOCUMENTATION (reference)
│   ├── README.md                  # Original project docs
│   ├── CLAUDE.md                  # Project context
│   ├── REFACTORING_GUIDE.md       # This file
│   └── ...                        # Other documentation
│
├── data/                          # DATA FILES (gitignored)
│   ├── song_lyrics.csv
│   ├── song_lyrics_cleaned.csv
│   └── song_lyrics_balanced.csv
│
├── requirements.txt               # DEPENDENCIES (pip install)
├── .gitignore                     # GIT CONFIG (updated)
└── README.md                      # PROJECT README (new)
```

## Common Tasks

### Task: Change TF-IDF hyperparameters

**Before:** Edit code in notebook, re-run cells

**After:** Edit config file
```bash
nano experiments/configs/tfidf_config.yaml
# Change max_features: 10000 -> 20000
python scripts/train.py --config experiments/configs/tfidf_config.yaml
```

### Task: Add a new model

1. Add class to `src/models.py`:
   ```python
   class MyNewModel:
       def __init__(self, **params):
           pass
       def fit(self, X_train, y_train):
           pass
       def predict(self, X_test):
           pass
   ```

2. Create config: `experiments/configs/my_model_config.yaml`

3. Add to `scripts/train.py`:
   ```python
   elif model_type == 'my_model':
       model, y_pred = train_my_model(config, ...)
   ```

### Task: Run BERT experiment with different parameters

```bash
# Copy config
cp experiments/configs/bert_config.yaml experiments/configs/bert_experiment.yaml

# Edit new config
nano experiments/configs/bert_experiment.yaml
# Change: epochs: 5 -> 10
#         learning_rate: 0.00002 -> 0.00001

# Run experiment
python scripts/train.py --config experiments/configs/bert_experiment.yaml
```

### Task: Share your work

**Before:** Send multiple notebooks, hard to reproduce

**After:**
1. Commit code: `git add src/ experiments/ scripts/`
2. Share config files: Someone can exactly reproduce with:
   ```bash
   git clone <your-repo>
   pip install -r requirements.txt
   python scripts/train.py --config experiments/configs/your_config.yaml
   ```

## Benefits Summary

### For You

1. **Less Code Duplication**
   - Write once in `src/`, use everywhere
   - Easier to fix bugs (one place)
   - Easier to add features

2. **Cleaner Notebooks**
   - Focus on experiments and visualization
   - Easier to read and share
   - Less cluttered

3. **Better Reproducibility**
   - YAML configs document exact parameters
   - Easy to re-run experiments
   - Version control friendly

4. **Professional Structure**
   - Ready for publication/sharing
   - Follows ML best practices
   - Easier for collaborators

### For Publication

1. **Credibility**
   - Shows engineering skills
   - Demonstrates reproducibility
   - Professional impression

2. **Usability**
   - Others can easily run your experiments
   - Clear documentation
   - Easy to build on your work

3. **Maintenance**
   - Easy to update and improve
   - Easy to add new experiments
   - Easy to fix issues

## Next Steps

1. **Try the new notebooks:**
   ```bash
   jupyter notebook notebooks/01_quick_start.ipynb
   ```

2. **Run a command-line experiment:**
   ```bash
   python scripts/train.py --config experiments/configs/tfidf_config.yaml
   ```

3. **Experiment with configs:**
   - Edit `experiments/configs/tfidf_config.yaml`
   - Change `samples_per_genre: 20000` to `5000` for faster testing
   - Run again and see the difference

4. **Read the new README:**
   - Open `README.md` in the project root
   - Comprehensive guide to the new structure

## Questions?

- **"Can I still use my old notebooks?"** Yes! They're in `notebooks/` and still work.

- **"Do I have to use command line?"** No! The new notebooks (`01_quick_start.ipynb`, `02_compare_models.ipynb`) use the modules too.

- **"What if I want to modify the models?"** Edit `src/models.py` - changes apply everywhere.

- **"How do I add new features?"** Add functions to appropriate `src/` module, then import in notebooks/scripts.

## Summary

The refactoring transformed the project from:
- **Notebook-heavy** → **Module-based**
- **Code duplication** → **Reusable components**
- **Hard to reproduce** → **Config-driven experiments**
- **Disorganized** → **Professional structure**

**The goal:** Make your research easier to run, share, and publish while maintaining all your original work!
