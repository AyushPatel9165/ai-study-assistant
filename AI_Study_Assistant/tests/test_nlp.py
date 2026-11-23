import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.preprocessor import NLPPreprocessor
from src.nlp.transformer_pipeline import (
    TransformerQuestionUnderstander,
    compute_scaled_dot_product_attention
)

def test_nlp_preprocessor_normalization():
    preprocessor = NLPPreprocessor()
    raw = "  What IS a Python Decorator? Check: https://python.org!  "
    norm = preprocessor.normalize(raw)
    assert "https" not in norm
    assert "what is a python decorator check" == norm

    tokens = preprocessor.tokenize_words(raw)
    assert "what" in tokens
    assert "decorator" in tokens

    filtered = preprocessor.remove_stopwords(tokens)
    assert "what" in filtered  # Question words preserved
    assert "is" not in filtered  # Stopword removed
    assert "a" not in filtered   # Stopword removed

def test_stemming_vs_lemmatization():
    preprocessor = NLPPreprocessor()
    
    # Test morphological lemmatization vs suffix-stripping stemming
    # Stemming chops suffixes blindly: 'studies' -> 'studi'
    assert preprocessor.stem_word("studies") == "studi"
    # Lemmatization maps to real dictionary lemma: 'studies' -> 'study'
    assert preprocessor.lemmatize_word("studies") == "study"

    assert preprocessor.lemmatize_word("better") == "good"
    assert preprocessor.lemmatize_word("matrices") == "matrix"

    comp = preprocessor.compare_stem_vs_lemma(["studies", "regularization"])
    assert len(comp) == 2
    assert comp[0]["original"] == "studies"
    assert comp[0]["lemmatized"] == "study"

def test_scaled_dot_product_attention_math():
    batch_size = 2
    num_heads = 1
    seq_len = 4
    d_k = 8
    d_v = 8

    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_v)

    output, weights = compute_scaled_dot_product_attention(Q, K, V)

    # 1. Output shape must be [batch, heads, seq_len, d_v]
    assert output.shape == (batch_size, num_heads, seq_len, d_v)

    # 2. Attention weights shape must be [batch, heads, seq_len, seq_len]
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # 3. Sum across keys dimension must equal 1.0 (softmax property)
    row_sums = weights.sum(dim=-1).detach().numpy()
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

def test_transformer_embedding_generation():
    understander = TransformerQuestionUnderstander()
    emb = understander.encode("How do list comprehensions work in Python?")
    
    # 1. Embedding dimension must be 384 for all-MiniLM-L6-v2
    assert emb.shape == (384,)
    assert emb.dtype == np.float32

    # 2. Embedding is L2-normalized (norm = 1.0)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)

def test_semantic_similarity_ranking():
    understander = TransformerQuestionUnderstander()
    q_base = "How does gradient descent update model weights?"
    q_similar = "Explain optimization using gradients in machine learning."
    q_unrelated = "What is the difference between append and extend in Python?"

    sim_high = understander.compute_similarity(q_base, q_similar)
    sim_low = understander.compute_similarity(q_base, q_unrelated)

    # Conceptual match must have higher similarity than cross-domain question
    assert sim_high > 0.45
    assert sim_low < 0.20
    assert sim_high > sim_low

    candidates = [
        "Explain Python lists",
        "What is gradient descent optimization?",
        "A* search algorithm in AI"
    ]
    ranked = understander.find_most_similar("Tell me about gradient descent", candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["candidate"] == "What is gradient descent optimization?"

def test_intent_extraction():
    understander = TransformerQuestionUnderstander()
    
    intent1 = understander.extract_intent("What is the difference between a list and a tuple?")
    assert intent1 == "Comparison / Contrast"

    intent2 = understander.extract_intent("How do I write a custom decorator in Python?")
    assert intent2 == "Implementation / How-To"

    intent3 = understander.extract_intent("What is the definition of supervised learning?")
    assert intent3 == "Definition / Conceptual"

    intent4 = understander.extract_intent("Can you summarize chapter 3 notes?")
    assert intent4 == "Summarization Request"

    intent5 = understander.extract_intent("Generate 5 MCQs on neural networks.")
    assert intent5 == "Quiz Request"

def test_empty_input_handling():
    understander = TransformerQuestionUnderstander()
    with pytest.raises(ValueError):
        understander.encode("   ")
