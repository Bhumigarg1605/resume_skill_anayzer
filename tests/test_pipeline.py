"""Comprehensive Test Suite for Resume Skill Analyzer Pipeline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.resume_parser import clean_resume_text, parse_resume
from src.skill_normalizer import SkillNormalizer
from src.skill_extractor import SkillExtractor
from src.job_skill_extractor import JobSkillExtractor
from src.semantic_matcher import SemanticSkillMatcher
from src.gap_analyzer import SkillGapAnalyzer
from src.evaluation import BenchmarkEvaluator


class TestResumeParser(unittest.TestCase):
    def test_clean_text_preserves_code_symbols(self):
        raw = "Experienced with C++, C#, and .NET core! Node.js & React-based apps."
        cleaned = clean_resume_text(raw)
        self.assertIn("C++", cleaned)
        self.assertIn("C#", cleaned)
        self.assertIn(".NET", cleaned)
        self.assertIn("Node.js", cleaned)

    def test_clean_text_removes_ligatures(self):
        raw = "Ligatures like \ufb01le and \ufb02oat should be normalized."
        cleaned = clean_resume_text(raw)
        self.assertIn("file", cleaned)
        self.assertIn("float", cleaned)

    def test_parse_resume_txt(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "resumes", "alex_morgan_ml_engineer.txt")
        if os.path.exists(sample_path):
            res = parse_resume(sample_path)
            self.assertTrue(res["success"])
            self.assertGreater(res["word_count"], 50)
            self.assertEqual(res["format"], "txt")

    def test_parse_resume_docx(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "resumes", "alex_morgan_ml_engineer.docx")
        if os.path.exists(sample_path):
            res = parse_resume(sample_path)
            self.assertTrue(res["success"])
            self.assertGreater(res["word_count"], 50)
            self.assertEqual(res["format"], "docx")

    def test_parse_resume_pdf(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "resumes", "alex_morgan_ml_engineer.pdf")
        if os.path.exists(sample_path):
            res = parse_resume(sample_path)
            self.assertTrue(res["success"])
            self.assertGreater(res["word_count"], 50)
            self.assertEqual(res["format"], "pdf")


class TestSkillNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = SkillNormalizer()

    def test_exact_lookup(self):
        res = self.normalizer.normalize("Python")
        self.assertEqual(res["canonical_name"], "Python")
        self.assertEqual(res["confidence"], 1.0)

    def test_alias_lookup(self):
        res = self.normalizer.normalize("sklearn")
        self.assertEqual(res["canonical_name"], "Scikit-Learn")
        self.assertEqual(res["matched_by"], "alias")

    def test_acronym_lookup(self):
        res = self.normalizer.normalize("ML")
        self.assertEqual(res["canonical_name"], "Machine Learning")

    def test_clean_key_lookup(self):
        res = self.normalizer.normalize("p-torch")
        self.assertEqual(res["canonical_name"], "PyTorch")

    def test_categories(self):
        cats = self.normalizer.get_categories()
        self.assertIn("Machine Learning", cats)
        self.assertIn("DevOps", cats)


class TestSkillExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = SkillExtractor()

    def test_extract_unique_skills(self):
        text = "I build machine learning models with PyTorch and deploy them using Docker and Kubernetes on AWS."
        skills = self.extractor.extract_unique_skills(text)
        self.assertTrue(any("pytorch" in s.lower() for s in skills))
        self.assertTrue(any("docker" in s.lower() for s in skills))
        self.assertTrue(any("kubernetes" in s.lower() for s in skills))

    def test_no_overlapping_spans(self):
        text = "Natural Language Processing and Natural Language Processing algorithms."
        spans = self.extractor.extract_skills(text)
        for i in range(len(spans) - 1):
            self.assertLessEqual(spans[i]["end"], spans[i + 1]["start"])


class TestJobSkillExtractor(unittest.TestCase):
    def setUp(self):
        self.job_extractor = JobSkillExtractor()

    def test_predefined_roles_loaded(self):
        roles = self.job_extractor.get_predefined_roles()
        self.assertGreaterEqual(len(roles), 5)

    def test_get_role_by_name(self):
        role = self.job_extractor.get_role_by_name("Machine Learning Engineer")
        self.assertIsNotNone(role)
        self.assertIn("Python", role["required_skills"])

    def test_extract_from_job_description(self):
        job_text = (
            "We need a backend developer with Python, FastAPI, and PostgreSQL.\n"
            "Nice to have: Docker and Redis."
        )
        res = self.job_extractor.extract_from_job_description(job_text)
        self.assertIn("Python", res["required_skills"])
        self.assertIn("FastAPI", res["required_skills"])


class TestSemanticMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = SemanticSkillMatcher()

    def test_compute_similarity_matrix(self):
        cands = ["Python", "PyTorch"]
        jobs = ["Python", "Deep Learning", "SQL"]
        matrix = self.matcher.compute_similarity_matrix(cands, jobs)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 3)
        # Exact match should be high
        self.assertGreater(matrix[0][0], 0.85)

    def test_match_skills(self):
        res = self.matcher.match_skills(
            candidate_skills=["Python", "PyTorch"],
            job_skills=["Python", "Deep Learning", "Kubernetes"],
            threshold=0.75,
        )
        matched = [m["job_skill"] for m in res["matched_pairs"]]
        missing = [m["job_skill"] for m in res["missing_skills"]]
        self.assertIn("Python", matched)
        self.assertIn("Kubernetes", missing)


class TestGapAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SkillGapAnalyzer()

    def test_match_percentage_calculation(self):
        cands = ["Python", "Docker", "SQL"]
        reqs = ["Python", "Docker", "Kubernetes", "AWS"]
        res = self.analyzer.analyze(cands, reqs, threshold=0.75)
        # 2 out of 4 matched => 50.0%
        self.assertEqual(res["match_percentage"], 50.0)
        self.assertEqual(len(res["matched_required"]), 2)
        self.assertEqual(len(res["missing_required"]), 2)
        self.assertEqual(len(res["recommendations"]), 2)


class TestBenchmarkEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = BenchmarkEvaluator()

    def test_proposed_outperforms_naive(self):
        results = self.evaluator.run_full_benchmark(threshold=0.75)
        naive_f1 = results["Naive Keyword Matching"]["F1"]
        prop_f1 = results["Proposed SBERT + ESCO Pipeline"]["F1"]
        self.assertGreater(prop_f1, naive_f1)
        self.assertGreater(prop_f1, 0.85)


if __name__ == "__main__":
    unittest.main()
