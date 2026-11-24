import re
from typing import List, Dict, Any, Optional

class RAGGenerator:
    """
    Synthesizes grounded, structured, and pedagogical answers strictly based
    on retrieved study note chunks. Enforces strict guardrails against hallucinating
    or inventing facts.
    """
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    # Metadata headers and boilerplate to filter out from raw text
    HEADER_FILTERS = (
        "sample technical study notes", "ai study assistant", "course:",
        "chapter 1:", "chapter 2:", "chapter 3:", "chunk #", "page "
    )

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer grounded strictly in retrieved chunks.
        If no chunks meet the relevance threshold, explicitly rejects the question.
        """
        # Guardrail 1: Empty or low-confidence retrieval
        if not retrieved_chunks:
            return {
                "answer": (
                    f"Unable to find relevant information in your uploaded study notes regarding: '{query}'.\n\n"
                    "Currently indexed topics include:\n"
                    "• Python Programming Language & Internals (Memory Management, Garbage Collection, GIL, Generators)\n"
                    "• Machine Learning (Optimization, Gradient Descent, Overfitting, L1/L2 Regularization)\n"
                    "• Artificial Intelligence (Search Spaces, A* Algorithm, Heuristics)\n"
                    "• MongoDB NoSQL (Architecture, BSON Documents, mongod, mongosh)\n\n"
                    "Tip: You can click 'Ask Web / External AI' below to get a comprehensive answer from global AI knowledge, or upload additional PDF/TXT notes in 'My Documents'!"
                ),
                "found_in_notes": False,
                "confidence": 0.0,
                "citations": []
            }

        top_score = retrieved_chunks[0]["score"]
        if top_score < self.confidence_threshold:
            return {
                "answer": (
                    f"Unable to find relevant information in your uploaded study notes regarding: '{query}'. "
                    f"(Highest relevance match was only {top_score:.1%}, which is below the required {self.confidence_threshold:.0%} confidence threshold).\n\n"
                    "Currently indexed notes cover Python Programming & Internals, ML Optimization & Regularization, AI A* Search, and MongoDB.\n\n"
                    "Tip: Click 'Ask Web / External AI' below to search global AI knowledge, or upload lecture notes for this topic in 'My Documents'!"
                ),
                "found_in_notes": False,
                "confidence": top_score,
                "citations": []
            }

        # Guardrail 2: Grounded answer formulation from top chunks
        combined_snippets = [c["text"] for c in retrieved_chunks[:4]]
        answer_text = self._synthesize_grounded_answer(query, combined_snippets)

        citations = [
            {
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "relevance_score": c["score"],
                "snippet": c["text"][:160] + "..." if len(c["text"]) > 160 else c["text"]
            }
            for c in retrieved_chunks[:3]
        ]

        return {
            "answer": answer_text,
            "found_in_notes": True,
            "confidence": top_score,
            "citations": citations
        }

    @classmethod
    def _clean_and_extract_sentences(cls, snippets: List[str]) -> List[str]:
        """
        Extracts clean, complete sentences from retrieved text chunks,
        healing boundary truncations and stripping document boilerplate.
        """
        sentences: List[str] = []
        seen = set()

        for snip in snippets:
            # Normalize bullets, carriage returns and colons
            text = snip.replace("•", "\n• ").replace("\r\n", "\n")
            # Strip specific markdown header titles safely without deleting trailing text
            text = re.sub(r"(?i)#+\s*AI Study Assistant\s*[-–]\s*Sample Technical Study Notes", " ", text)
            text = re.sub(r"(?i)#+\s*Course:[^#\n.]*(?:##|\.|\n|$)", " ", text)
            text = re.sub(r"(?i)#+\s*Chapter\s+\d+:[^#\n.]*(?:##|\.|\n|$)", " ", text)
            text = re.sub(r"(?i)sample technical study notes", " ", text)

            raw_parts = re.split(r"(?<=[.?!])\s+|\n+", text)

            for part in raw_parts:
                p = part.strip()
                if not p:
                    continue

                # Strip leading chapter/section prefixes (e.g. '2. MongoDB Basics ')
                p = re.sub(r"^\d+\.\s+[A-Za-z0-9\s]+Basics\s*", "", p, flags=re.IGNORECASE)
                p = re.sub(r"^\d+\.\s+[A-Za-z\s]+(?=(What|The|In|MongoDB|Python|Regularization|Overfitting|An|A\*))", "", p).strip()

                # Fix lowercase boundary cutoff fragments (e.g. 'nd queries.', 'ative gradient...')
                if re.match(r"^[a-z’'\"\)\]\>\,]", p):
                    match_cap = re.search(r"[.?!]\s+([A-Z0-9].*)", p)
                    if match_cap:
                        p = match_cap.group(1).strip()
                    else:
                        continue

                # Clean trailing cutoffs like isolated '1.' or '2.'
                p = re.sub(r"\s+\d+\.\s*$", "", p).strip()

                # Filter pure question headings that mirror syllabus titles (e.g. 'What is MongoDB?')
                if re.match(r"^(What|How|Why)\s+(is|are|does|do)\s+[^?]+\?$", p, re.IGNORECASE):
                    continue

                # Filter out pure heading relics (e.g. 'MongoDB Basics What is MongoDB?')
                if "what is mongodb" in p.lower() and not ("database" in p.lower() or "nosql" in p.lower()):
                    continue

                # Filter boilerplate headers
                p_lower = p.lower()
                if any(h in p_lower for h in cls.HEADER_FILTERS) and len(p.split()) < 12:
                    continue

                if len(p) < 18:
                    continue
                if p.count("•") > 3:
                    continue

                norm = re.sub(r"\s+", " ", p.lower())
                if norm not in seen:
                    seen.add(norm)
                    sentences.append(p)

        return sentences

    def _synthesize_grounded_answer(self, query: str, context_snippets: List[str]) -> str:
        """
        Synthesizes a structured, educational, tutor-grade answer grounded strictly
        in the retrieved study note chunks.
        """
        sentences = self._clean_and_extract_sentences(context_snippets)
        if not sentences:
            return "Based on your uploaded study notes:\n\nNo detailed context could be extracted from notes."

        # Term expansion for accurate relevance scoring
        exp_q = re.sub(r"\bml\b", "machine learning", query, flags=re.IGNORECASE)
        exp_q = re.sub(r"\bai\b", "artificial intelligence", exp_q, flags=re.IGNORECASE)
        exp_q = re.sub(r"\bpy\b", "python", exp_q, flags=re.IGNORECASE)
        exp_q = re.sub(r"\bgil\b", "global interpreter lock mutex cpython thread", exp_q, flags=re.IGNORECASE)
        exp_q = re.sub(r"\bgc\b", "garbage collection memory cyclic", exp_q, flags=re.IGNORECASE)
        exp_q = re.sub(r"\byield\b", "yield generator function lazy frame stack", exp_q, flags=re.IGNORECASE)

        query_words = set(re.findall(r"\w+", exp_q.lower()))
        raw_query_words = set(re.findall(r"\w+", query.lower()))
        stop_words = {
            "what", "is", "the", "difference", "between", "how", "do", "does", "in", "a",
            "an", "and", "or", "of", "to", "notes", "give", "me", "tell", "about", "explain",
            "can", "you", "for", "with", "from", "on", "are", "which", "why"
        }
        key_query_words = query_words - stop_words
        core_raw_keywords = raw_query_words - stop_words

        overview_candidates = []
        detail_candidates = []
        takeaway_candidates = []

        def_indicators = (
            "is a", "is an", "are a", "are an", "are special", "are functions", "refers to",
            "occurs when", "happens when", "defined as", "represents", "used to", "operates by",
            "is designed", "stores data", "is an informed"
        )
        takeaway_indicators = (
            "consequently", "therefore", "as a result", "tip", "perfect for", "must run",
            "developers must", "to solve this", "makes them ideal", "ideal for", "guaranteed to be"
        )
        component_indicators = (
            "–", "—", ":", "1.", "2.", "3.", "mongod", "mongosh", "compass",
            "l1 regularization", "l2 regularization", "lasso", "ridge", "f(n)", "g(n)", "h(n)",
            "yield", "frame stack", "mutex", "reference counting", "cyclic"
        )

        for s in sentences:
            s_lower = s.lower()
            s_words = set(re.findall(r"\w+", s_lower))

            # Overlap scoring: give double weight to primary words directly in the student's question
            overlap = len(key_query_words.intersection(s_words))
            core_overlap = len(core_raw_keywords.intersection(s_words))
            weighted_score = overlap + (core_overlap * 2)

            # Direct definition bonus: if the sentence directly defines the exact query term (e.g. "python is a...")
            for kw in core_raw_keywords:
                if len(kw) > 2:
                    if re.search(rf"\b{re.escape(kw)}\s+(is\s+(a|an|the)|are|refers\s+to|was\s+created)\b", s_lower):
                        weighted_score += 15
                    elif s_lower.startswith(f"{kw} "):
                        weighted_score += 8

            # Categorize sentences
            if any(d in s_lower for d in def_indicators) and weighted_score > 0:
                overview_candidates.append((weighted_score, s))
            elif any(s_lower.startswith(t) or f" {t} " in s_lower for t in takeaway_indicators):
                takeaway_candidates.append((weighted_score, s))
            elif any(c in s_lower for c in component_indicators) or weighted_score > 0:
                detail_candidates.append((weighted_score, s))
            else:
                detail_candidates.append((weighted_score, s))

        # Sort by relevance to query terms
        overview_candidates.sort(key=lambda x: x[0], reverse=True)
        detail_candidates.sort(key=lambda x: x[0], reverse=True)
        takeaway_candidates.sort(key=lambda x: x[0], reverse=True)

        sections: List[str] = []
        used_texts = set()

        # 1. Section: Concept Overview & Definition
        primary_def = []
        if overview_candidates:
            primary_def.append(overview_candidates[0][1])
            used_texts.add(overview_candidates[0][1])
            if len(overview_candidates) > 1 and overview_candidates[1][0] >= overview_candidates[0][0] - 1:
                primary_def.append(overview_candidates[1][1])
                used_texts.add(overview_candidates[1][1])
        elif detail_candidates:
            primary_def.append(detail_candidates[0][1])
            used_texts.add(detail_candidates[0][1])

        overview_text = " ".join(primary_def)
        if overview_text:
            sections.append(f"### Concept Overview\n{overview_text}")

        # 2. Section: Key Technical Details & Mechanisms (Formatted Bullets)
        bullet_items = []
        for score, s in detail_candidates:
            if s in used_texts:
                continue
            if score > 0 or len(bullet_items) < 2:
                clean_s = s.lstrip("-•* ").strip()
                clean_s = re.sub(r"^Key Components\s+", "", clean_s)

                # Format component items nicely with bold titles
                if "–" in clean_s:
                    parts = clean_s.split("–", 1)
                    bullet_items.append(f"• **{parts[0].strip()}**: {parts[1].strip()}")
                elif " - " in clean_s:
                    parts = clean_s.split(" - ", 1)
                    bullet_items.append(f"• **{parts[0].strip()}**: {parts[1].strip()}")
                elif ":" in clean_s and len(clean_s.split(":", 1)[0].split()) <= 4:
                    parts = clean_s.split(":", 1)
                    bullet_items.append(f"• **{parts[0].strip()}**: {parts[1].strip()}")
                else:
                    bullet_items.append(f"• {clean_s}")
                used_texts.add(s)

            if len(bullet_items) >= 4:
                break

        if bullet_items:
            sections.append("### Key Technical Breakdown\n" + "\n".join(bullet_items))

        # 3. Section: Practical Takeaways & Application
        relevant_takeaways = [s for score, s in takeaway_candidates if score > 0 and s not in used_texts]
        if not relevant_takeaways and takeaway_candidates:
            relevant_takeaways = [takeaway_candidates[0][1]]

        if relevant_takeaways:
            takeaway_text = " ".join(relevant_takeaways[:2])
            sections.append(f"### Practical Takeaways & Usage\n{takeaway_text}")

        full_answer = "\n\n".join(sections)
        return f"Based on your uploaded study notes:\n\n{full_answer}"
