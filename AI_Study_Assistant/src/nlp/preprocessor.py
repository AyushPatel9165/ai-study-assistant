import re
import string
from typing import List, Dict, Any, Tuple

class NLPPreprocessor:
    """
    Text preprocessing, normalization, stemming, and lemmatization module for student questions.
    """
    def __init__(self):
        # Common English morphological rules & irregular lemma map
        self.irregular_lemmas = {
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
            "ran": "run",
            "running": "run",
            "runs": "run",
            "studies": "study",
            "studying": "study",
            "studied": "study",
            "classes": "class",
            "matrices": "matrix",
            "indices": "index",
            "corpora": "corpus",
            "algorithms": "algorithm",
            "functions": "function",
            "variables": "variable",
            "losses": "loss",
            "trees": "tree",
            "networks": "network",
            "layers": "layer",
            "weights": "weight",
            "biases": "bias",
            "regularized": "regularize",
            "regularizing": "regularize",
            "regularization": "regularize",
            "classified": "classify",
            "classifying": "classify",
            "classifiers": "classifier"
        }

        # Common stop words for filtering
        self.stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been",
            "it", "this", "that", "these", "those", "i", "you", "he", "she",
            "they", "we"
        }

    def normalize(self, text: str) -> str:
        """
        Normalizes text: lowercase, strip URLs, strip punctuation, remove extra spaces.
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        for p in string.punctuation:
            text = text.replace(p, " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_words(self, text: str) -> List[str]:
        """
        Word-level tokenization.
        """
        cleaned = self.normalize(text)
        return cleaned.split() if cleaned else []

    def stem_word(self, word: str) -> str:
        """
        Rule-based suffix stripping (Stemming).
        Note: Stemming cuts suffixes crudely, often producing non-dictionary roots (e.g., 'studying' -> 'studi').
        """
        w = word.lower()
        if len(w) <= 3:
            return w

        # Common suffix rules
        suffixes = [
            ("ization", "ize"),
            ("ational", "ate"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("iveness", "ive"),
            ("tional", "tion"),
            ("biliti", "ble"),
            ("alism", "al"),
            ("ation", "ate"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("ousli", "ous"),
            ("ing", ""),
            ("ies", "i"),
            ("sses", "ss"),
            ("ed", ""),
            ("ly", ""),
            ("es", ""),
            ("s", "")
        ]
        for suf, repl in suffixes:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[:-len(suf)] + repl
        return w

    def lemmatize_word(self, word: str) -> str:
        """
        Morphological lemmatization.
        Maps words to authentic canonical base dictionary lemmas using vocabulary and grammar rules.
        """
        w = word.lower()
        if w in self.irregular_lemmas:
            return self.irregular_lemmas[w]

        # Standard singularization and base form recovery
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            if w.endswith("sses") or w.endswith("shes") or w.endswith("ches") or w.endswith("xes"):
                return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        if w.endswith("ing") and len(w) > 5:
            # e.g., predicting -> predict
            return w[:-3]
        if w.endswith("ed") and len(w) > 4:
            # e.g., regularized -> regularize, trained -> train
            return w[:-1] if w.endswith("eed") else w[:-2]

        return w

    def compare_stem_vs_lemma(self, words: List[str]) -> List[Dict[str, str]]:
        """
        Returns side-by-side comparison between stemming and lemmatization.
        """
        results = []
        for word in words:
            results.append({
                "original": word,
                "stemmed": self.stem_word(word),
                "lemmatized": self.lemmatize_word(word)
            })
        return results

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.lower() not in self.stop_words]
