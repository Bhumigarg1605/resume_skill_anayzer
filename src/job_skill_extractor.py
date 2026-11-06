"""Job Description Skill Extraction Module.

Supports parsing unstructured job postings, extracting required/preferred
skill requirements, and fetching predefined industry roles from the catalog.
"""

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional
from src.skill_extractor import SkillExtractor
from src.skill_normalizer import SkillNormalizer


class JobSkillExtractor:
    """Extracts target skill profiles from raw job descriptions or predefined roles."""

    def __init__(
        self,
        jobs_csv_path: Optional[str] = None,
        skill_extractor: Optional[SkillExtractor] = None,
        normalizer: Optional[SkillNormalizer] = None,
    ):
        """Initializes job skill extractor.

        Args:
            jobs_csv_path: Path to dataset/jobs/job_roles.csv.
            skill_extractor: SkillExtractor instance.
            normalizer: SkillNormalizer instance.
        """
        self.jobs_csv_path = jobs_csv_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "jobs", "job_roles.csv"
        )
        self.normalizer = normalizer or SkillNormalizer()
        self.skill_extractor = skill_extractor or SkillExtractor(normalizer=self.normalizer)
        self.predefined_roles: List[Dict[str, Any]] = []

        self._load_predefined_roles()

    def _load_predefined_roles(self) -> None:
        """Loads predefined tech job roles from CSV or JSON catalog."""
        if os.path.isfile(self.jobs_csv_path):
            try:
                with open(self.jobs_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        req_skills = [s.strip() for s in row.get("required_skills", "").split("|") if s.strip()]
                        pref_skills = [s.strip() for s in row.get("preferred_skills", "").split("|") if s.strip()]
                        self.predefined_roles.append({
                            "role_id": row.get("role_id", ""),
                            "role_name": row.get("role_name", ""),
                            "category": row.get("category", ""),
                            "experience_level": row.get("experience_level", ""),
                            "required_skills": req_skills,
                            "preferred_skills": pref_skills,
                            "raw_description": row.get("raw_description", ""),
                        })
                return
            except Exception:
                pass

        # Try JSON catalog
        json_path = os.path.join(os.path.dirname(self.jobs_csv_path), "job_descriptions.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.predefined_roles = json.load(f)
                return
            except Exception:
                pass

        # Built-in fallback roles
        self.predefined_roles = [
            {
                "role_id": "JOB_MLE_DEFAULT",
                "role_name": "Machine Learning Engineer",
                "category": "Artificial Intelligence",
                "experience_level": "Mid-Senior",
                "required_skills": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "Scikit-Learn", "Docker", "SQL", "Git"],
                "preferred_skills": ["MLflow", "Kubernetes", "Amazon Web Services", "FastAPI"],
                "raw_description": "We need an ML Engineer proficient in Python, PyTorch, Scikit-Learn, Docker, and SQL.",
            },
            {
                "role_id": "JOB_DS_DEFAULT",
                "role_name": "Data Scientist",
                "category": "Data Science",
                "experience_level": "Mid-Level",
                "required_skills": ["Python", "SQL", "Pandas", "NumPy", "Statistics", "Machine Learning", "Scikit-Learn"],
                "preferred_skills": ["Tableau", "Power BI", "Hypothesis Testing"],
                "raw_description": "Data Scientist needed with Python, Pandas, SQL, Statistics, and Scikit-Learn experience.",
            },
        ]

    def get_predefined_roles(self) -> List[Dict[str, Any]]:
        """Returns the list of all predefined roles in catalog."""
        return self.predefined_roles

    def get_role_by_id(self, role_id: str) -> Optional[Dict[str, Any]]:
        """Finds a predefined role by its role ID."""
        for role in self.predefined_roles:
            if role.get("role_id") == role_id:
                return role
        return None

    def get_role_by_name(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Finds a predefined role by its display name (case-insensitive)."""
        target = role_name.strip().lower()
        for role in self.predefined_roles:
            if role.get("role_name", "").strip().lower() == target:
                return role
        return None

    def extract_from_job_description(self, raw_text: str) -> Dict[str, Any]:
        """Parses arbitrary unstructured job posting text and extracts required skills.

        Segments text into Requirements and Preferred/Bonus sections if present,
        extracts skill entities, and normalizes them into canonical representations.

        Args:
            raw_text: Raw job posting description.

        Returns:
            Dictionary containing:
                - required_skills: Canonical names of essential skills
                - preferred_skills: Canonical names of bonus/preferred skills
                - all_skills: Complete combined set of unique canonical skills
                - raw_mentions: List of raw extracted skill entities
        """
        if not raw_text or not isinstance(raw_text, str):
            return {
                "required_skills": [],
                "preferred_skills": [],
                "all_skills": [],
                "raw_mentions": [],
            }

        # Identify section boundaries using regex
        # Look for headers like "Preferred Qualifications:", "Nice to have:", "Bonus:"
        preferred_regex = re.compile(
            r"(?:preferred|nice\s*to\s*have|bonus|plus|desired|optional)\s*(?:qualifications|skills|requirements|experience)?:?",
            re.IGNORECASE,
        )
        match = preferred_regex.search(raw_text)

        required_text = raw_text
        preferred_text = ""
        if match:
            split_idx = match.start()
            required_text = raw_text[:split_idx]
            preferred_text = raw_text[split_idx:]

        # Extract entities from both sections
        req_raw = self.skill_extractor.extract_unique_skills(required_text)
        pref_raw = self.skill_extractor.extract_unique_skills(preferred_text) if preferred_text else []

        # Normalize into canonical representations
        req_norm = [self.normalizer.normalize(s)["canonical_name"] for s in req_raw]
        pref_norm = [self.normalizer.normalize(s)["canonical_name"] for s in pref_raw]

        # Deduplicate while preserving order
        req_canonical = list(dict.fromkeys(req_norm))
        pref_canonical = [s for s in dict.fromkeys(pref_norm) if s not in req_canonical]

        # If no split was detected, all extracted skills are classified as required
        all_canonical = list(dict.fromkeys(req_canonical + pref_canonical))

        return {
            "required_skills": req_canonical,
            "preferred_skills": pref_canonical,
            "all_skills": all_canonical,
            "raw_mentions": req_raw + pref_raw,
        }
