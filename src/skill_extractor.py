"""Skill Entity Extraction Engine.

Combines fine-tuned spaCy Named Entity Recognition (NER) model with a
rule-based PhraseMatcher and high-throughput n-gram taxonomy scanner.
Guarantees entity extraction across diverse resume formats.
"""

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.skill_normalizer import SkillNormalizer


class SkillExtractor:
    """Extracts technical skill entities from unstructured resume and job text."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        normalizer: Optional[SkillNormalizer] = None,
    ):
        """Initializes the extractor with spaCy pipeline and phrase matchers.

        Args:
            model_path: Path to trained spaCy NER model directory.
            normalizer: SkillNormalizer instance to fetch taxonomy concepts.
        """
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "skill_ner"
        )
        self.normalizer = normalizer or SkillNormalizer()
        self.nlp = None
        self.phrase_matcher = None
        self._taxonomy_terms: Set[str] = set()

        self._build_taxonomy_term_set()
        self._initialize_pipeline()

    def _build_taxonomy_term_set(self) -> None:
        """Collects canonical skills and aliases into a normalized lookup set."""
        for canon, data in self.normalizer.canonical_skills.items():
            self._taxonomy_terms.add(canon.lower())
            for alias in data.get("aliases", []):
                self._taxonomy_terms.add(alias.lower())

    def _initialize_pipeline(self) -> None:
        """Loads spaCy model and configures PhraseMatcher if spaCy is installed."""
        try:
            import spacy
            from spacy.matcher import PhraseMatcher

            # 1. Try loading trained custom model
            if os.path.exists(self.model_path):
                try:
                    self.nlp = spacy.load(self.model_path)
                except Exception:
                    self.nlp = None

            # 2. Fall back to standard en_core_web_sm or blank english
            if self.nlp is None:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")

            # 3. Add PhraseMatcher for taxonomy concepts
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(term) for term in self._taxonomy_terms if len(term.strip()) > 1]
            self.phrase_matcher.add("SKILL_PHRASE", patterns)

        except ImportError:
            # spaCy not installed; fallback to pure Python n-gram & regex scanner
            self.nlp = None
            self.phrase_matcher = None

    def _extract_via_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extracts skills using spaCy NER and PhraseMatcher."""
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        extracted: List[Dict[str, Any]] = []

        # 1. NER Model entities
        for ent in doc.ents:
            if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:
                # If model is custom SKILL or standard entity that matches taxonomy
                if ent.label_ == "SKILL" or ent.text.lower() in self._taxonomy_terms:
                    extracted.append({
                        "text": ent.text.strip(),
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": "SKILL",
                        "source": "spacy_ner",
                    })

        # 2. PhraseMatcher matches
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start_tok, end_tok in matches:
                span = doc[start_tok:end_tok]
                extracted.append({
                    "text": span.text.strip(),
                    "start": span.start_char,
                    "end": span.end_char,
                    "label": "SKILL",
                    "source": "phrase_matcher",
                })

        return extracted

    def _extract_via_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Pure-Python regex and n-gram taxonomy scanner fallback."""
        extracted: List[Dict[str, Any]] = []
        text_lower = text.lower()

        # Sort taxonomy terms by length descending to prioritize longer phrases
        # e.g., "Natural Language Processing" before "Processing"
        sorted_terms = sorted(self._taxonomy_terms, key=len, reverse=True)

        for term in sorted_terms:
            if len(term) < 2:
                continue

            # Word boundary regex pattern
            # For skills with symbols like C++ or .NET, handle boundaries safely
            escaped_term = re.escape(term)
            pattern = rf"(?<!\w){escaped_term}(?!\w)"
            
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                matched_text = text[start:end]
                extracted.append({
                    "text": matched_text,
                    "start": start,
                    "end": end,
                    "label": "SKILL",
                    "source": "taxonomy_scanner",
                })

        return extracted

    @staticmethod
    def _deduplicate_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolves overlapping entity spans, favoring longer spans and earlier positions."""
        if not spans:
            return []

        # Sort by start ascending, then length descending
        sorted_spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))

        resolved: List[Dict[str, Any]] = []
        last_end = -1

        for span in sorted_spans:
            # Allow non-overlapping spans
            if span["start"] >= last_end:
                resolved.append(span)
                last_end = span["end"]
            else:
                # If current span is strictly longer and completely subsumes previous, replace it
                if resolved and span["end"] > resolved[-1]["end"] and span["start"] == resolved[-1]["start"]:
                    resolved[-1] = span
                    last_end = span["end"]

        return resolved

    def extract_skills(self, text: str) -> List[Dict[str, Any]]:
        """Extracts all skill entity mentions with character offsets.

        Args:
            text: Unstructured plain text from resume or job spec.

        Returns:
            List of dictionaries:
                [{ "text": "PyTorch", "start": 45, "end": 52, "label": "SKILL", "source": "..." }, ...]
        """
        if not text or not isinstance(text, str):
            return []

        # Try spaCy extraction first
        spacy_spans = self._extract_via_spacy(text)

        # Also run fallback scanner to ensure maximum recall
        fallback_spans = self._extract_via_fallback(text)

        # Merge and deduplicate
        all_spans = spacy_spans + fallback_spans
        deduped = self._deduplicate_spans(all_spans)
        return deduped

    def extract_unique_skills(self, text: str) -> List[str]:
        """Extracts unique raw skill entity names, preserving discovery order.

        Args:
            text: Plain text input.

        Returns:
            List of unique string skill mentions.
        """
        entities = self.extract_skills(text)
        seen: Set[str] = set()
        unique_skills: List[str] = []

        for ent in entities:
            mention = ent["text"].strip()
            mention_key = mention.lower()
            if mention_key not in seen:
                seen.add(mention_key)
                unique_skills.append(mention)

        return unique_skills
