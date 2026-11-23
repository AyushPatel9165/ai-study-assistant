import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.nlp.preprocessor import NLPPreprocessor

def compute_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        query: Tensor of shape [batch_size, num_heads, seq_len_q, d_k]
        key: Tensor of shape [batch_size, num_heads, seq_len_k, d_k]
        value: Tensor of shape [batch_size, num_heads, seq_len_v, d_v]
        mask: Optional mask tensor to prevent attending to future tokens or padding
    Returns:
        output: Weighted context representation [batch_size, num_heads, seq_len_q, d_v]
        attention_weights: Normalized attention probabilities [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    d_k = query.size(-1)
    # 1. Compute dot products of Q and K^T
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

    # 2. Apply mask if provided (e.g. for decoder causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax across key dimension to get attention probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Multiply attention weights by Value vectors
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class TransformerQuestionUnderstander:
    """
    Pretrained Transformer pipeline for natural language question understanding
    using Hugging Face Sentence Transformers (all-MiniLM-L6-v2).
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_name = model_name
        self.preprocessor = NLPPreprocessor()
        self._model = None

    @property
    def model(self):
        """Lazy load model from local cache to ensure fast offline/sandboxed execution."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                # Attempt fast load from local Hugging Face cache
                self._model = SentenceTransformer(
                    self.model_name,
                    device=str(self.device),
                    local_files_only=True
                )
            except Exception:
                # Fallback to standard online loader if cache not populated
                self._model = SentenceTransformer(
                    self.model_name,
                    device=str(self.device)
                )
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a single text into a 384-dimensional dense semantic embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of texts into dense semantic embeddings.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes Cosine Similarity between two texts:
        Cosine Similarity = dot(u, v) / (norm(u) * norm(v))
        Because embeddings are normalized to unit length, this is simply the dot product.
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        similarity = float(np.dot(emb1, emb2))
        return round(similarity, 4)

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate questions or document snippets by semantic relevance to the query.
        """
        if not candidates:
            return []

        query_emb = self.encode(query)
        cand_embs = self.encode_batch(candidates)

        # Dot product against all candidates
        similarities = np.dot(cand_embs, query_emb)
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank": rank,
                "candidate": candidates[idx],
                "similarity": round(float(similarities[idx]), 4)
            })
        return results

    def extract_intent(self, question: str) -> str:
        """
        Identifies structural query intention from student question.
        """
        q_lower = question.lower().strip()
        if any(w in q_lower for w in ["difference between", "versus", "vs", "compare"]):
            return "Comparison / Contrast"
        if any(w in q_lower for w in ["how to", "how do i", "how can i", "implement"]):
            return "Implementation / How-To"
        if any(w in q_lower for w in ["what is", "define", "meaning of", "explain the concept"]):
            return "Definition / Conceptual"
        if any(w in q_lower for w in ["why does", "why is", "purpose of", "reason for"]):
            return "Reasoning / Architecture"
        if any(w in q_lower for w in ["summarize", "summary", "key points"]):
            return "Summarization Request"
        if any(w in q_lower for w in ["quiz", "test me", "mcq", "questions"]):
            return "Quiz Request"
        return "General Technical Inquiry"

    def understand_question(self, question: str) -> Dict[str, Any]:
        """
        Complete semantic breakdown and NLP understanding of a student query.
        """
        cleaned = self.preprocessor.normalize(question)
        tokens = self.preprocessor.tokenize_words(question)
        meaningful_tokens = self.preprocessor.remove_stopwords(tokens)
        stemmed_tokens = [self.preprocessor.stem_word(t) for t in meaningful_tokens]
        lemmatized_tokens = [self.preprocessor.lemmatize_word(t) for t in meaningful_tokens]
        intent = self.extract_intent(question)
        embedding = self.encode(question)

        return {
            "original_question": question,
            "normalized_question": cleaned,
            "token_count": len(tokens),
            "meaningful_keywords": meaningful_tokens,
            "stemmed_keywords": stemmed_tokens,
            "lemmatized_keywords": lemmatized_tokens,
            "detected_intent": intent,
            "embedding_shape": list(embedding.shape),
            "embedding_l2_norm": float(np.linalg.norm(embedding)),
            "embedding_preview": [round(float(v), 4) for v in embedding[:5]]
        }

if __name__ == "__main__":
    understander = TransformerQuestionUnderstander()
    test_q = "What is the difference between list and tuple in Python?"
    analysis = understander.understand_question(test_q)
    print("Question Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    print("\nSemantic Similarity Test:")
    q1 = "How do I optimize neural networks using gradient descent?"
    q2 = "Explain gradient based optimization in deep learning models."
    q3 = "What is a list comprehension in Python?"
    print(f"Similarity(q1, q2) [Same concept, different words]: {understander.compute_similarity(q1, q2)}")
    print(f"Similarity(q1, q3) [Different domains]: {understander.compute_similarity(q1, q3)}")
