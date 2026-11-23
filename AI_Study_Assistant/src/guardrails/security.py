import re
import time
from typing import Dict, Any, Tuple, Optional, List

class PromptInjectionGuard:
    """
    Detects adversarial prompt injection, jailbreaking, and system prompt override attempts.
    """
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
        r"(?i)disregard\s+(all\s+)?(previous|prior)\s+(instructions|rules)",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+(dan|an\s+unfiltered|an\s+adversary|in\s+developer\s+mode)",
        r"(?i)bypass\s+(all\s+)?(guardrails|filters|safety)",
        r"(?i)reveal\s+(your\s+)?(system\s+prompt|initial\s+prompt|hidden\s+instructions)",
        r"(?i)forget\s+(your\s+)?(instructions|rules|guidelines)",
        r"(?i)print\s+(your\s+)?(system\s+instructions|system\s+prompt)",
        r"(?i)act\s+as\s+jailbreak",
        r"(?i)pretend\s+you\s+have\s+no\s+rules"
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def check_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_safe, failure_reason)
        """
        if not text:
            return True, None

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False, f"Prompt rejected: Potential prompt injection or system override pattern detected."

        return True, None


class SensitiveDataFilter:
    """
    Detects and sanitizes personally identifiable information (PII) and secret credentials.
    """
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    API_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|bearer\s+[A-Za-z0-9\-_.]{20,})\b", re.IGNORECASE)
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

    def sanitize(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Replaces sensitive tokens with redacted tags and returns count of sanitizations.
        """
        if not text:
            return "", {}

        sanitized = text
        counts = {
            "emails": 0,
            "phones": 0,
            "api_keys": 0,
            "credit_cards": 0
        }

        # Mask emails
        emails = self.EMAIL_PATTERN.findall(sanitized)
        if emails:
            counts["emails"] = len(emails)
            sanitized = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

        # Mask API Keys & Tokens
        keys = self.API_KEY_PATTERN.findall(sanitized)
        if keys:
            counts["api_keys"] = len(keys)
            sanitized = self.API_KEY_PATTERN.sub("[REDACTED_SECRET]", sanitized)

        # Mask Credit Cards
        cards = self.CREDIT_CARD_PATTERN.findall(sanitized)
        if cards:
            counts["credit_cards"] = len(cards)
            sanitized = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", sanitized)

        # Mask Phones
        phones = self.PHONE_PATTERN.findall(sanitized)
        if phones:
            counts["phones"] = len(phones)
            sanitized = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)

        return sanitized, counts


class RateLimiter:
    """
    Sliding-window in-memory rate limiter per IP address.
    """
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60
        self.request_history: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Returns (is_allowed, remaining_quota)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        history = self.request_history.get(client_id, [])
        # Prune old timestamps
        history = [ts for ts in history if ts > cutoff]

        if len(history) >= self.max_requests:
            self.request_history[client_id] = history
            return False, 0

        history.append(now)
        self.request_history[client_id] = history
        remaining = self.max_requests - len(history)
        return True, remaining
