"""
Model wrappers for genre classification.

Provides unified interfaces for:
- TF-IDF + Classical ML (Logistic Regression, SVM, Naive Bayes)
- Word2Vec + Classical ML
- BERT/DistilBERT transformers
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Word2Vec
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from .utils import get_device, count_parameters, NUM_GENRES


# =============================================================================
# TF-IDF Models
# =============================================================================

class TFIDFModel:
    """Wrapper for TF-IDF + Classical ML models."""

    def __init__(
        self,
        classifier_type: str = 'logistic',
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.8,
        **classifier_kwargs
    ):
        """
        Initialize TF-IDF model.

        Args:
            classifier_type: 'logistic', 'svm', or 'naive_bayes'
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            **classifier_kwargs: Additional arguments for classifier
        """
        self.classifier_type = classifier_type

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )

        # Classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                **classifier_kwargs
            )
        elif classifier_type == 'svm':
            self.classifier = LinearSVC(
                max_iter=1000,
                random_state=42,
                **classifier_kwargs
            )
        elif classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB(**classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, X_train, y_train):
        """Train the model."""
        print(f"Training TF-IDF + {self.classifier_type}...")

        # Fit TF-IDF
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  TF-IDF features: {X_train_tfidf.shape[1]:,}")

        # Fit classifier
        self.classifier.fit(X_train_tfidf, y_train)
        print(f"✅ Training complete")

    def predict(self, X_test):
        """Make predictions."""
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, list]:
        """Get top features for each class (Logistic Regression only)."""
        if self.classifier_type != 'logistic':
            raise ValueError("Feature importance only available for Logistic Regression")

        feature_names = self.vectorizer.get_feature_names_out()
        importance = {}

        for genre_id in range(NUM_GENRES):
            # Get coefficients for this class
            coef = self.classifier.coef_[genre_id]
            top_indices = np.argsort(coef)[-top_n:][::-1]
            importance[genre_id] = [
                (feature_names[i], coef[i])
                for i in top_indices
            ]

        return importance


# =============================================================================
# Word2Vec Models
# =============================================================================

class Word2VecModel:
    """Wrapper for Word2Vec + Classical ML models."""

    def __init__(
        self,
        classifier_type: str = 'logistic',
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 5,
        epochs: int = 10,
        **classifier_kwargs
    ):
        """
        Initialize Word2Vec model.

        Args:
            classifier_type: 'logistic' or 'svm'
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            epochs: Training epochs for Word2Vec
            **classifier_kwargs: Additional arguments for classifier
        """
        self.classifier_type = classifier_type
        self.vector_size = vector_size
        self.w2v_model = None

        # Word2Vec parameters
        self.w2v_params = {
            'vector_size': vector_size,
            'window': window,
            'min_count': min_count,
            'workers': -1,
            'sg': 0,  # CBOW
            'epochs': epochs
        }

        # Classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                **classifier_kwargs
            )
        elif classifier_type == 'svm':
            self.classifier = LinearSVC(
                max_iter=1000,
                random_state=42,
                **classifier_kwargs
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def _tokenize(self, texts):
        """Tokenize texts for Word2Vec."""
        return [text.split() for text in texts]

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to averaged word vector."""
        words = text.split()
        vectors = []

        for word in words:
            if word in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[word])

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def fit(self, X_train, y_train):
        """Train the model."""
        print(f"Training Word2Vec + {self.classifier_type}...")

        # Train Word2Vec
        tokenized = self._tokenize(X_train)
        self.w2v_model = Word2Vec(sentences=tokenized, **self.w2v_params)
        print(f"  Word2Vec vocabulary: {len(self.w2v_model.wv):,} words")

        # Convert texts to vectors
        X_train_vectors = np.array([self._text_to_vector(text) for text in X_train])

        # Train classifier
        self.classifier.fit(X_train_vectors, y_train)
        print(f"✅ Training complete")

    def predict(self, X_test):
        """Make predictions."""
        X_test_vectors = np.array([self._text_to_vector(text) for text in X_test])
        return self.classifier.predict(X_test_vectors)


# =============================================================================
# BERT/DistilBERT Models
# =============================================================================

class LyricsDataset(Dataset):
    """PyTorch Dataset for lyrics."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = labels.reset_index(drop=True) if hasattr(labels, 'reset_index') else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = int(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }


class BERTModel:
    """Wrapper for BERT/DistilBERT models."""

    def __init__(
        self,
        model_name: str = 'distilbert-base-multilingual-cased',
        max_length: int = 256,
        batch_size: int = 96,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.1,
        use_amp: bool = True
    ):
        """
        Initialize BERT model.

        Args:
            model_name: Hugging Face model name
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate
            epochs: Number of training epochs
            warmup_ratio: Warmup ratio for scheduler
            weight_decay: Weight decay for regularization
            use_amp: Use automatic mixed precision
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_amp = use_amp

        self.device = get_device()
        self.tokenizer = None
        self.model = None

        print(f"Initializing {model_name}")
        print(f"  Device: {self.device}")

    def _initialize_model(self):
        """Initialize tokenizer and model."""
        from transformers import DataCollatorWithPadding

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            return_tensors='pt'
        )

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_GENRES
        )

        self.model = self.model.to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        print(f"  Parameters: {count_parameters(self.model):,}")

    def fit(self, X_train, y_train):
        """Train the model."""
        if self.tokenizer is None:
            self._initialize_model()

        print(f"\nTraining {self.model_name}...")

        # Create dataset and dataloader
        train_dataset = LyricsDataset(X_train, y_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True
        )

        # Setup training
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(self.warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        if loss.dim() > 0:
                            loss = loss.mean()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    if loss.dim() > 0:
                        loss = loss.mean()
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{self.epochs}: loss = {avg_loss:.4f}")

        print("✅ Training complete")

    def predict(self, X_test):
        """Make predictions."""
        if self.tokenizer is None:
            raise ValueError("Model not trained yet")

        # Create dataset
        # Use dummy labels for prediction
        dummy_labels = np.zeros(len(X_test))
        test_dataset = LyricsDataset(X_test, dummy_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True
        )

        # Predict
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)
