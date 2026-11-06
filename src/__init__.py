"""Resume Skill Analyzer - ML-based skill gap detection system.

Modules:
    resume_parser: Extract and sanitize text from PDF and DOCX files.
    skill_extractor: Extract technical skill entities using NER and phrase matching.
    skill_normalizer: Map extracted terms to canonical ESCO taxonomy standards.
    job_skill_extractor: Extract required skill profiles from job postings and descriptions.
    semantic_matcher: Sentence-Transformer cosine similarity engine.
    gap_analyzer: Skill gap algebraic analysis and match percentage calculations.
    evaluation: Benchmarking against naive baselines with precision, recall, and F1 metrics.
"""

from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, clean_resume_text, parse_resume
from src.skill_normalizer import SkillNormalizer
from src.skill_extractor import SkillExtractor
from src.job_skill_extractor import JobSkillExtractor
from src.semantic_matcher import SemanticSkillMatcher
from src.gap_analyzer import SkillGapAnalyzer
from src.evaluation import BenchmarkEvaluator

__version__ = "1.0.0"
__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "clean_resume_text",
    "parse_resume",
    "SkillNormalizer",
    "SkillExtractor",
    "JobSkillExtractor",
    "SemanticSkillMatcher",
    "SkillGapAnalyzer",
    "BenchmarkEvaluator",
]
