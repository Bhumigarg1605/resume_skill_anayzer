"""Skill Gap Analysis and Scoring Module.

Computes candidate match percentage, classifies skills into matched, missing,
and supplementary sets, and generates actionable upskilling recommendations.
"""

from typing import Any, Dict, List, Optional, Set
from src.semantic_matcher import SemanticSkillMatcher
from src.skill_normalizer import SkillNormalizer


class SkillGapAnalyzer:
    """Calculates skill gap metrics and generates candidate career recommendations."""

    def __init__(
        self,
        semantic_matcher: Optional[SemanticSkillMatcher] = None,
        normalizer: Optional[SkillNormalizer] = None,
    ):
        """Initializes the gap analyzer.

        Args:
            semantic_matcher: SemanticSkillMatcher instance.
            normalizer: SkillNormalizer instance.
        """
        self.normalizer = normalizer or SkillNormalizer()
        self.matcher = semantic_matcher or SemanticSkillMatcher()

    def analyze(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        preferred_skills: Optional[List[str]] = None,
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Performs comprehensive skill gap analysis between resume and job profile.

        Calculates match percentage using the formula:
            Match % = (|Matched Unique Required Skills| / |Total Required Job Skills|) * 100

        Args:
            candidate_skills: List of skill names from resume.
            required_skills: List of mandatory skill requirements from job posting.
            preferred_skills: Optional list of nice-to-have skill requirements.
            threshold: Cosine similarity cutoff (default 0.75).

        Returns:
            Dictionary containing:
                - match_percentage: float (0.0 - 100.0)
                - matched_required: List of matched required skills with similarity
                - missing_required: List of required skills not covered
                - matched_preferred: List of matched preferred skills
                - missing_preferred: List of preferred skills not covered
                - additional_skills: Skills on resume not in job requirement
                - total_required_count: int
                - total_candidate_count: int
                - recommendations: List of personalized learning recommendations
        """
        # Canonicalize all inputs
        cand_norm = [self.normalizer.normalize(s)["canonical_name"] for s in candidate_skills]
        cand_unique = list(dict.fromkeys([s for s in cand_norm if s]))

        req_norm = [self.normalizer.normalize(s)["canonical_name"] for s in required_skills]
        req_unique = list(dict.fromkeys([s for s in req_norm if s]))

        pref_unique: List[str] = []
        if preferred_skills:
            p_norm = [self.normalizer.normalize(s)["canonical_name"] for s in preferred_skills]
            pref_unique = [s for s in dict.fromkeys(p_norm) if s and s not in req_unique]

        # 1. Match against Required Skills
        req_match_result = self.matcher.match_skills(
            candidate_skills=cand_unique,
            job_skills=req_unique,
            threshold=threshold,
        )

        matched_req = req_match_result["matched_pairs"]
        missing_req = req_match_result["missing_skills"]

        # Calculate Match Percentage
        total_req = len(req_unique)
        matched_req_count = len(matched_req)
        match_percentage = round((matched_req_count / total_req * 100.0), 1) if total_req > 0 else 100.0

        # 2. Match against Preferred Skills if provided
        matched_pref: List[Dict[str, Any]] = []
        missing_pref: List[Dict[str, Any]] = []

        if pref_unique:
            pref_match_result = self.matcher.match_skills(
                candidate_skills=cand_unique,
                job_skills=pref_unique,
                threshold=threshold,
            )
            matched_pref = pref_match_result["matched_pairs"]
            missing_pref = pref_match_result["missing_skills"]

        # 3. Identify Additional / Bonus Candidate Skills
        # Skills in candidate resume that were not matched to any required or preferred job skill
        covered_candidate_skills: Set[str] = set()
        for m in matched_req + matched_pref:
            if m.get("best_candidate_skill"):
                covered_candidate_skills.add(m["best_candidate_skill"].lower())

        additional_skills = [
            s for s in cand_unique if s.lower() not in covered_candidate_skills
        ]

        # 4. Generate Upskilling Recommendations for missing required skills
        missing_skill_names = [m["job_skill"] for m in missing_req]
        recommendations = self.generate_recommendations(missing_skill_names)

        return {
            "match_percentage": match_percentage,
            "matched_required": matched_req,
            "missing_required": missing_req,
            "matched_preferred": matched_pref,
            "missing_preferred": missing_pref,
            "additional_skills": additional_skills,
            "total_required_count": total_req,
            "total_candidate_count": len(cand_unique),
            "recommendations": recommendations,
            "threshold": threshold,
        }

    def generate_recommendations(self, missing_skills: List[str]) -> List[Dict[str, Any]]:
        """Generates structured advice and resources for missing target skills.

        Args:
            missing_skills: List of canonical skill names that were missing.

        Returns:
            List of recommendation dicts sorted by priority.
        """
        recs = []
        high_priority_categories = {"Machine Learning", "Programming", "Data Science", "Databases"}

        for skill in missing_skills:
            info = self.normalizer.canonical_skills.get(skill, {})
            category = info.get("category", "General Technical")
            description = info.get("description", f"Foundational {skill} skills.")

            priority = "Critical" if category in high_priority_categories else "High"

            recs.append({
                "skill": skill,
                "category": category,
                "priority": priority,
                "description": description,
                "suggested_action": f"Build a portfolio project or complete a hands-on certification covering {skill}.",
            })

        # Sort Critical first, then High
        priority_order = {"Critical": 0, "High": 1, "Medium": 2}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 9))
        return recs
