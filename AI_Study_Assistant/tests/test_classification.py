import os
import sys
from pathlib import Path
import pytest
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.dataset import TextPreprocessor, QuestionDataset
from src.classification.model import QuestionClassifierMLP
from src.classification.predict import QuestionClassifier

def test_text_preprocessor():
    preprocessor = TextPreprocessor()
    raw = "  What are *args & **kwargs in Python? Visit https://python.org  "
    cleaned = preprocessor.clean(raw)
    assert "https" not in cleaned
    assert "*" not in cleaned
    assert "&" not in cleaned
    assert cleaned == "what are args kwargs in python visit"

def test_question_dataset():
    features = np.random.randn(10, 50).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    ds = QuestionDataset(features, labels)
    assert len(ds) == 10
    feat, lbl = ds[0]
    assert feat.shape == (50,)
    assert lbl.item() == 0

def test_model_forward_pass():
    model = QuestionClassifierMLP(input_dim=100, hidden_dim1=64, hidden_dim2=32, num_classes=3)
    x = torch.randn(4, 100)
    logits = model(x)
    assert logits.shape == (4, 3)

def test_model_predict_proba():
    model = QuestionClassifierMLP(input_dim=100, hidden_dim1=64, hidden_dim2=32, num_classes=3)
    x = torch.randn(2, 100)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 3)
    row_sums = probs.sum(dim=-1).numpy()
    assert np.allclose(row_sums, [1.0, 1.0], atol=1e-5)

def test_trained_classifier_predictions():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "models" / "question_classifier.pt")
    vec_path = str(base_dir / "models" / "tfidf_vectorizer.pkl")
    enc_path = str(base_dir / "models" / "label_encoder.pkl")

    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet, skipping inference test.")

    classifier = QuestionClassifier(
        model_path=model_path,
        vectorizer_path=vec_path,
        encoder_path=enc_path
    )

    # 1. Test Python query
    res_py = classifier.predict("How do list comprehensions and generators work in Python?")
    assert res_py["category"] == "Python"
    assert res_py["confidence"] > 0.70
    assert "Python" in res_py["probabilities"]

    # 2. Test AI query
    res_ai = classifier.predict("What is the minimax algorithm with alpha beta pruning in game search?")
    assert res_ai["category"] == "AI"
    assert res_ai["confidence"] > 0.70

    # 3. Test ML query
    res_ml = classifier.predict("How does gradient descent prevent overfitting with L2 ridge regularization?")
    assert res_ml["category"] == "ML"
    assert res_ml["confidence"] > 0.70

def test_batch_prediction():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "models" / "question_classifier.pt")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet.")

    classifier = QuestionClassifier(
        model_path=str(base_dir / "models" / "question_classifier.pt"),
        vectorizer_path=str(base_dir / "models" / "tfidf_vectorizer.pkl"),
        encoder_path=str(base_dir / "models" / "label_encoder.pkl")
    )
    queries = [
        "Explain Python decorators and yield statements.",
        "What is the A* search heuristic function?",
        "What is cross-entropy loss in machine learning?"
    ]
    results = classifier.batch_predict(queries)
    assert len(results) == 3
    assert results[0]["category"] == "Python"
    assert results[1]["category"] == "AI"
    assert results[2]["category"] == "ML"

def test_empty_question_validation():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "models" / "question_classifier.pt")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet.")

    classifier = QuestionClassifier(
        model_path=str(base_dir / "models" / "question_classifier.pt"),
        vectorizer_path=str(base_dir / "models" / "tfidf_vectorizer.pkl"),
        encoder_path=str(base_dir / "models" / "label_encoder.pkl")
    )
    with pytest.raises(ValueError):
        classifier.predict("   ")
