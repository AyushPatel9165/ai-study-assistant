import os
import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classification.dataset import TextPreprocessor
from src.classification.model import QuestionClassifierMLP

class QuestionClassifier:
    """
    Production inference engine for student question classification with Explainable AI (XAI).
    Loads trained PyTorch MLP model, TF-IDF vectorizer, and label encoder.
    Computes input-gradient saliency feature attributions to explain predictions.
    """
    def __init__(
        self,
        model_path: str = "models/question_classifier.pt",
        vectorizer_path: str = "models/tfidf_vectorizer.pkl",
        encoder_path: str = "models/label_encoder.pkl",
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.encoder_path = encoder_path
        self.preprocessor = TextPreprocessor()

        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Train the model first.")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {self.vectorizer_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at: {self.encoder_path}")

        # Load vectorizer and label encoder
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.encoder_path)
        self.classes = list(self.label_encoder.classes_)

        # Infer input dimension from vectorizer vocabulary
        input_dim = len(self.vectorizer.vocabulary_)
        num_classes = len(self.classes)

        # Inverted vocabulary map for fast word lookup
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Initialize and load PyTorch model
        self.model = QuestionClassifierMLP(
            input_dim=input_dim,
            hidden_dim1=128,
            hidden_dim2=64,
            num_classes=num_classes
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def explain_prediction(self, features: np.ndarray, top_idx: int, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Computes Explainable AI (XAI) feature attributions using Saliency (Input * Gradient).
        Identifies the exact words that influenced the neural network's prediction.
        """
        tensor_x = torch.tensor(features, dtype=torch.float32, device=self.device, requires_grad=True)
        # Forward pass through raw linear layers without softmax
        logits = self.model(tensor_x)
        target_logit = logits[0, top_idx]

        self.model.zero_grad()
        target_logit.backward()

        grads = tensor_x.grad[0].cpu().numpy()
        feat_vals = features[0]

        # Saliency attribution: input * gradient
        attributions = np.maximum(0, grads * feat_vals)
        non_zero_indices = np.where(attributions > 0)[0]

        if len(non_zero_indices) == 0:
            # Fallback to pure TF-IDF weights if gradients are flat
            non_zero_indices = np.where(feat_vals > 0)[0]
            attributions = feat_vals

        if len(non_zero_indices) == 0:
            return []

        # Sort by highest attribution
        sorted_indices = non_zero_indices[np.argsort(attributions[non_zero_indices])[::-1]][:top_k]
        total_att = float(np.sum(attributions[sorted_indices])) or 1.0

        top_features = []
        for idx in sorted_indices:
            word = str(self.feature_names[idx])
            contrib = float(attributions[idx])
            pct = round((contrib / total_att) * 100, 1)
            top_features.append({
                "word": word,
                "importance_percentage": pct
            })

        return top_features

    def predict(self, question: str) -> Dict[str, Any]:
        """
        Classifies a single question into Python, AI, or ML.
        Returns predicted category, confidence score, full distribution, and XAI explanation.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        cleaned = self.preprocessor.clean(question)
        features = self.vectorizer.transform([cleaned]).toarray()
        tensor_x = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probabilities = self.model.predict_proba(tensor_x).cpu().numpy()[0]

        top_idx = int(np.argmax(probabilities))
        top_category = str(self.classes[top_idx])
        confidence = float(probabilities[top_idx])

        # Compute XAI feature attribution
        top_features = self.explain_prediction(features, top_idx)
        top_words_str = ", ".join([f"'{f['word']}' ({f['importance_percentage']}%)" for f in top_features[:3]])
        if top_words_str:
            explanation = f"Model classified this as '{top_category}' primarily influenced by keywords: {top_words_str}."
        else:
            explanation = f"Model classified this as '{top_category}' based on general grammatical structure."

        prob_dist = {
            str(self.classes[i]): round(float(probabilities[i]), 4)
            for i in range(len(self.classes))
        }

        return {
            "question": question,
            "category": top_category,
            "confidence": round(confidence, 4),
            "probabilities": prob_dist,
            "cleaned_question": cleaned,
            "top_features": top_features,
            "explanation": explanation
        }

    def batch_predict(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Efficient batch classification for multiple student questions.
        """
        if not questions:
            return []

        cleaned_list = self.preprocessor.batch_clean(questions)
        features = self.vectorizer.transform(cleaned_list).toarray()
        tensor_x = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probabilities = self.model.predict_proba(tensor_x).cpu().numpy()

        results = []
        for i, q in enumerate(questions):
            probs = probabilities[i]
            top_idx = int(np.argmax(probs))
            results.append({
                "question": q,
                "category": str(self.classes[top_idx]),
                "confidence": round(float(probs[top_idx]), 4),
                "probabilities": {
                    str(self.classes[c]): round(float(probs[c]), 4)
                    for c in range(len(self.classes))
                }
            })
        return results

if __name__ == "__main__":
    classifier = QuestionClassifier()
    test_queries = [
        "How do generators and the yield keyword work in Python?",
        "What is the minimax algorithm and alpha beta pruning?",
        "Explain overfitting and L2 regularization."
    ]
    for q in test_queries:
        res = classifier.predict(q)
        print(f"\nQ: {res['question']}")
        print(f"Predicted Category: {res['category']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Explanation: {res['explanation']}")
        print(f"Top Features: {res['top_features']}")
