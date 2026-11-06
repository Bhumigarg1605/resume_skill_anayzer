"""Semantic Skill Matching Engine.

Computes semantic similarity vectors between candidate resume skills and
target job skill requirements using Sentence-BERT (all-MiniLM-L6-v2)
and pairwise cosine similarity scoring with configurable thresholding.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


class SemanticSkillMatcher:
    """Computes semantic cosine similarity between skill terms and requirements."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initializes the semantic matcher.

        Args:
            model_name: HuggingFace model hub ID or local path for SentenceTransformer.
        """
        self.model_name = model_name
        self.model = None
        self._is_transformer = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load SentenceTransformer model; sets fallback if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._is_transformer = True
        except Exception:
            # Fallback to subword character n-gram TF-IDF cosine vectorizer
            self.model = None
            self._is_transformer = False

    def _encode_fallback(self, texts: List[str]) -> List[Dict[str, float]]:
        """Encodes texts into character and token n-gram TF-IDF vectors in pure Python."""
        vectors = []
        for text in texts:
            clean = text.lower().strip()
            tokens = clean.split()
            # Character 3-grams and 4-grams + word tokens
            features: Dict[str, float] = {}
            for t in tokens:
                features[f"w_{t}"] = features.get(f"w_{t}", 0.0) + 2.0
            for n in (3, 4):
                padded = f"_{clean}_"
                for i in range(len(padded) - n + 1):
                    ngram = padded[i : i + n]
                    features[f"c_{ngram}"] = features.get(f"c_{ngram}", 0.0) + 1.0

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in features.values())) or 1.0
            norm_features = {k: v / norm for k, v in features.items()}
            vectors.append(norm_features)
        return vectors

    @staticmethod
    def _cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Computes dot product between two L2-normalized sparse vectors."""
        # Iterate over smaller dict
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        dot = sum(val * v2.get(k, 0.0) for k, val in v1.items())
        return min(max(dot, 0.0), 1.0)

    def encode(self, texts: List[str]) -> Any:
        """Generates embedding representations for an input list of strings.

        Args:
            texts: List of skill names or phrases.

        Returns:
            NumPy array of dense embeddings (if SentenceTransformer loaded)
            or list of normalized sparse vectors (fallback).
        """
        if not texts:
            return []

        if self._is_transformer and self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings
            except Exception:
                pass

        return self._encode_fallback(texts)

    def compute_similarity_matrix(
        self, candidate_skills: List[str], job_skills: List[str]
    ) -> List[List[float]]:
        """Computes a 2D pairwise cosine similarity matrix.

        Rows correspond to candidate skills; columns correspond to job skills.
        matrix[i][j] is the similarity between candidate_skills[i] and job_skills[j].

        Args:
            candidate_skills: List of skills present on the resume.
            job_skills: List of required skills for target job.

        Returns:
            2D float matrix of dimensions [len(candidate_skills) x len(job_skills)].
        """
        if not candidate_skills or not job_skills:
            return []

        # If transformer is active and numpy is available
        if self._is_transformer and self.model is not None:
            try:
                import numpy as np
                cand_emb = self.encode(candidate_skills)
                job_emb = self.encode(job_skills)
                # Dense dot product since embeddings are L2 normalized
                sim_matrix = np.dot(cand_emb, job_emb.T)
                # Clip values to [0.0, 1.0]
                sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
                return sim_matrix.tolist()
            except Exception:
                pass

        # Fallback sparse cosine computation
        cand_vecs = self._encode_fallback(candidate_skills)
        job_vecs = self._encode_fallback(job_skills)

        matrix = []
        for c_vec in cand_vecs:
            row = []
            for j_vec in job_vecs:
                row.append(round(self._cosine_sparse(c_vec, j_vec), 4))
            matrix.append(row)
        return matrix

    def match_skills(
        self,
        candidate_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Aligns candidate skills to job skills via semantic similarity.

        For every job skill, computes the maximum similarity against candidate
        skills and classifies the requirement as:
        - Exact Match (score >= 0.95 or exact text match)
        - Soft Match (threshold <= score < 0.95)
        - Missing (score < threshold)

        Args:
            candidate_skills: Unique skills extracted from resume.
            job_skills: Required skills from target role.
            threshold: Calibrated minimum cosine similarity for a match (default 0.75).

        Returns:
            Dictionary with:
                - matched_pairs: List of dicts describing each matched skill
                - missing_skills: List of job skills not adequately covered
                - matrix: Full pairwise similarity matrix
                - candidate_skills: Cleaned input candidate skills
                - job_skills: Cleaned input job skills
                - threshold: Applied similarity threshold
        """
        clean_cand = [s.strip() for s in candidate_skills if s.strip()]
        clean_jobs = [s.strip() for s in job_skills if s.strip()]

        if not clean_jobs:
            return {
                "matched_pairs": [],
                "missing_skills": [],
                "matrix": [],
                "candidate_skills": clean_cand,
                "job_skills": clean_jobs,
                "threshold": threshold,
            }

        if not clean_cand:
            return {
                "matched_pairs": [],
                "missing_skills": [
                    {"job_skill": js, "best_candidate_skill": None, "similarity": 0.0, "status": "Missing"}
                    for js in clean_jobs
                ],
                "matrix": [],
                "candidate_skills": [],
                "job_skills": clean_jobs,
                "threshold": threshold,
            }

        matrix = self.compute_similarity_matrix(clean_cand, clean_jobs)

        matched_pairs: List[Dict[str, Any]] = []
        missing_skills: List[Dict[str, Any]] = []

        # For each job skill (column j), find the best matching candidate skill (row i)
        for j, job_skill in enumerate(clean_jobs):
            best_score = -1.0
            best_cand_idx = -1

            for i, cand_skill in enumerate(clean_cand):
                score = matrix[i][j]
                # Boost if exact case-insensitive match
                if cand_skill.lower() == job_skill.lower():
                    score = 1.0

                if score > best_score:
                    best_score = score
                    best_cand_idx = i

            best_cand_skill = clean_cand[best_cand_idx] if best_cand_idx >= 0 else None
            match_status = "Exact Match" if best_score >= 0.95 else ("Semantic Match" if best_score >= threshold else "Missing")

            item = {
                "job_skill": job_skill,
                "best_candidate_skill": best_cand_skill,
                "similarity": round(float(best_score), 4),
                "status": match_status,
            }

            if best_score >= threshold:
                matched_pairs.append(item)
            else:
                missing_skills.append(item)

        return {
            "matched_pairs": matched_pairs,
            "missing_skills": missing_skills,
            "matrix": matrix,
            "candidate_skills": clean_cand,
            "job_skills": clean_jobs,
            "threshold": threshold,
        }
