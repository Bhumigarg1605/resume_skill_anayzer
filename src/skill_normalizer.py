"""Skill Normalization and Canonicalization Engine.

Maps raw extracted entity mentions to standard ESCO / O*NET taxonomy concepts
using exact alias indexing and RapidFuzz string comparison (with standard library
difflib fallback) for robust entity resolution.
"""

import csv
import difflib
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class SkillNormalizer:
    """Standardizes skill variants and acronyms to canonical taxonomy labels."""

    def __init__(self, taxonomy_path: Optional[str] = None):
        """Initializes normalizer and loads ESCO taxonomy dictionary.

        Args:
            taxonomy_path: Path to skills taxonomy CSV. If None, looks for default
                           dataset/skills/skills_taxonomy.csv or loads built-in cache.
        """
        self.taxonomy_path = taxonomy_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "skills", "skills_taxonomy.csv"
        )
        # Canonical skill registry: canonical_label -> {category, skill_type, description, aliases}
        self.canonical_skills: Dict[str, Dict[str, Any]] = {}
        # Alias lookup: lower_alias -> canonical_label
        self.alias_lookup: Dict[str, str] = {}
        # Precomputed list of lowercase canonical strings for fuzzy matching
        self._canonical_list_lower: List[Tuple[str, str]] = []

        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        """Loads taxonomy dataset from CSV or built-in dictionary."""
        if os.path.isfile(self.taxonomy_path):
            try:
                with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pref = row.get("preferred_label", "").strip()
                        if not pref:
                            continue
                        cat = row.get("category", "General")
                        stype = row.get("skill_type", "skill")
                        desc = row.get("description", "")
                        alts = [a.strip() for a in row.get("alt_labels", "").split("|") if a.strip()]

                        self.canonical_skills[pref] = {
                            "category": cat,
                            "skill_type": stype,
                            "description": desc,
                            "aliases": alts,
                        }

                        # Index preferred label
                        self.alias_lookup[pref.lower()] = pref
                        # Index simplified alphanumeric form (e.g., 'scikit learn' -> 'scikitlearn')
                        self.alias_lookup[self._clean_key(pref)] = pref

                        # Index each alias
                        for alt in alts:
                            self.alias_lookup[alt.lower()] = pref
                            self.alias_lookup[self._clean_key(alt)] = pref
            except Exception:
                self._load_fallback_taxonomy()
        else:
            self._load_fallback_taxonomy()

        # Cache lowercase canonical targets for fuzzy fallback
        self._canonical_list_lower = [(label, label.lower()) for label in self.canonical_skills.keys()]

    def _load_fallback_taxonomy(self) -> None:
        """Minimal fallback taxonomy when file is inaccessible."""
        fallback = {
            "Machine Learning": ("Machine Learning", ["ML", "Statistical Learning"]),
            "Deep Learning": ("Machine Learning", ["DL", "Neural Networks"]),
            "Python": ("Programming", ["Python3", "py"]),
            "PyTorch": ("Machine Learning", ["Torch"]),
            "TensorFlow": ("Machine Learning", ["TF"]),
            "Scikit-Learn": ("Machine Learning", ["sklearn", "scikit learn"]),
            "Docker": ("DevOps", ["Docker Containers", "Containerization"]),
            "Kubernetes": ("DevOps", ["K8s", "Container Orchestration"]),
            "SQL": ("Data Science", ["Structured Query Language"]),
            "Pandas": ("Data Science", ["pd", "Pandas DataFrame"]),
            "NumPy": ("Data Science", ["np"]),
            "React": ("Frontend", ["React.js", "ReactJS"]),
            "TypeScript": ("Frontend", ["TS"]),
            "JavaScript": ("Frontend", ["JS", "ES6"]),
            "Amazon Web Services": ("Cloud", ["AWS"]),
            "Google Cloud Platform": ("Cloud", ["GCP"]),
            "Git": ("Software Engineering", ["GitHub", "GitLab", "Version Control"]),
            "FastAPI": ("Backend", ["Fast API"]),
            "PostgreSQL": ("Databases", ["Postgres"]),
            "Redis": ("Databases", ["Redis Cache"]),
            "CI/CD": ("DevOps", ["Continuous Integration", "Continuous Delivery"]),
            "Natural Language Processing": ("Machine Learning", ["NLP"]),
            "Large Language Models": ("Machine Learning", ["LLMs", "GenAI"]),
        }
        for pref, (cat, alts) in fallback.items():
            self.canonical_skills[pref] = {"category": cat, "skill_type": "skill", "description": "", "aliases": alts}
            self.alias_lookup[pref.lower()] = pref
            self.alias_lookup[self._clean_key(pref)] = pref
            for alt in alts:
                self.alias_lookup[alt.lower()] = pref
                self.alias_lookup[self._clean_key(alt)] = pref

    @staticmethod
    def _clean_key(text: str) -> str:
        """Strip non-alphanumeric characters for normalized comparison."""
        return re.sub(r"[^a-zA-Z0-9]", "", text.lower())

    def _fuzzy_score(self, s1: str, s2: str) -> float:
        """Computes similarity score between 0 and 100."""
        # 1. Try rapidfuzz
        try:
            from rapidfuzz import fuzz
            return fuzz.token_sort_ratio(s1, s2)
        except ImportError:
            pass

        # 2. Standard library difflib fallback
        matcher = difflib.SequenceMatcher(None, s1.lower(), s2.lower())
        return matcher.ratio() * 100.0

    def normalize(self, skill_text: str, threshold: float = 80.0) -> Dict[str, Any]:
        """Normalizes a single skill mention into canonical taxonomy format.

        Resolution Hierarchy:
        1. Exact lowercase match against canonical name
        2. Exact lowercase match against indexed alias/synonym
        3. Cleaned alphanumeric match (removing dashes, slashes, spaces)
        4. High-confidence fuzzy string matching (RapidFuzz / SequenceMatcher)

        Args:
            skill_text: Raw skill entity mention (e.g. "sklearn", "p-torch", "ML")
            threshold: Fuzzy matching cutoff threshold between 0.0 and 100.0.

        Returns:
            Dictionary containing:
                - canonical_name: Standardized label
                - original_name: Raw input
                - category: Taxonomy domain
                - confidence: 0.0 to 1.0 confidence score
                - matched_by: 'exact', 'alias', 'clean_key', 'fuzzy', or 'unmatched'
        """
        raw = skill_text.strip()
        if not raw:
            return {
                "canonical_name": "",
                "original_name": raw,
                "category": "Unknown",
                "confidence": 0.0,
                "matched_by": "unmatched",
            }

        low = raw.lower()
        cleaned_key = self._clean_key(raw)

        # 1 & 2. Exact lookup
        if low in self.alias_lookup:
            canonical = self.alias_lookup[low]
            info = self.canonical_skills.get(canonical, {})
            matched_by = "exact" if low == canonical.lower() else "alias"
            return {
                "canonical_name": canonical,
                "original_name": raw,
                "category": info.get("category", "General"),
                "confidence": 1.0,
                "matched_by": matched_by,
            }

        # 3. Cleaned alphanumeric key lookup
        if cleaned_key and cleaned_key in self.alias_lookup:
            canonical = self.alias_lookup[cleaned_key]
            info = self.canonical_skills.get(canonical, {})
            return {
                "canonical_name": canonical,
                "original_name": raw,
                "category": info.get("category", "General"),
                "confidence": 0.95,
                "matched_by": "clean_key",
            }

        # 4. Fuzzy comparison against canonical labels and aliases
        best_match = None
        best_score = 0.0

        for canon_label, canon_lower in self._canonical_list_lower:
            score = self._fuzzy_score(low, canon_lower)
            if score > best_score:
                best_score = score
                best_match = canon_label

            # Also check aliases if score is not yet near-perfect
            if score < 95.0:
                for alias in self.canonical_skills[canon_label]["aliases"]:
                    alt_score = self._fuzzy_score(low, alias.lower())
                    if alt_score > best_score:
                        best_score = alt_score
                        best_match = canon_label

        if best_match and best_score >= threshold:
            info = self.canonical_skills.get(best_match, {})
            return {
                "canonical_name": best_match,
                "original_name": raw,
                "category": info.get("category", "General"),
                "confidence": round(best_score / 100.0, 3),
                "matched_by": "fuzzy",
            }

        # Unmatched fallback: Capitalize raw mention
        return {
            "canonical_name": raw.title(),
            "original_name": raw,
            "category": "Other",
            "confidence": 0.50,
            "matched_by": "unmatched",
        }

    def batch_normalize(self, skills: List[str], threshold: float = 80.0) -> List[Dict[str, Any]]:
        """Batch processes a list of raw skill mentions."""
        return [self.normalize(s, threshold=threshold) for s in skills]

    def get_all_canonical_skills(self) -> List[str]:
        """Returns sorted list of all canonical skill names in taxonomy."""
        return sorted(list(self.canonical_skills.keys()))

    def get_categories(self) -> List[str]:
        """Returns sorted list of all skill category domains."""
        cats: Set[str] = {info.get("category", "General") for info in self.canonical_skills.values()}
        return sorted(list(cats))
