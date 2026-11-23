import re
import string
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional, Dict, Any

class TextPreprocessor:
    """
    Cleans and normalizes natural language questions for ML/DL pipelines.
    """
    def __init__(self, remove_punctuation: bool = True, lowercase: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove punctuation if specified, but retain word boundaries
        if self.remove_punctuation:
            # Replace punctuation characters with space
            for char in string.punctuation:
                text = text.replace(char, " ")

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]


class QuestionDataset(Dataset):
    """
    PyTorch Dataset wrapper for student questions.
    Yields (feature_tensor, label_tensor).
    """
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], torch.tensor(-1, dtype=torch.long)


class DataPipeline:
    """
    Orchestrates dataset loading, stratified train/val/test splitting,
    TF-IDF feature extraction without data leakage, and PyTorch DataLoader creation.
    """
    def __init__(
        self,
        max_features: int = 800,
        ngram_range: Tuple[int, int] = (1, 2),
        batch_size: int = 16,
        random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.batch_size = batch_size
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            stop_words="english"
        )
        self.label_encoder = LabelEncoder()

    def load_and_prepare(
        self,
        csv_path: str,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        if "question" not in df.columns or "category" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'category' columns.")

        # Clean all questions
        cleaned_questions = self.preprocessor.batch_clean(df["question"].tolist())
        raw_labels = df["category"].tolist()

        # Encode labels to integers (0, 1, 2)
        encoded_labels = self.label_encoder.fit_transform(raw_labels)

        # First split: Train + Val vs Test (Stratified)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            cleaned_questions,
            encoded_labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=encoded_labels
        )

        # Second split: Train vs Val (Stratified)
        val_relative_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_relative_ratio,
            random_state=self.random_state,
            stratify=y_train_val
        )

        # Fit TF-IDF on train set ONLY to avoid data leakage
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        X_val_vec = self.vectorizer.transform(X_val).toarray()
        X_test_vec = self.vectorizer.transform(X_test).toarray()

        # Create PyTorch Datasets
        train_ds = QuestionDataset(X_train_vec, y_train)
        val_ds = QuestionDataset(X_val_vec, y_val)
        test_ds = QuestionDataset(X_test_vec, y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "X_test_raw": X_test,
            "y_test": y_test,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "feature_dim": X_train_vec.shape[1],
            "num_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_)
        }
