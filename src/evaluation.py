"""Model Evaluation and Baseline Benchmarking Module.

Implements rigorous experimental comparison between Naive Keyword Matching
and the Proposed Semantic ML Pipeline across Precision, Recall, and F1 metrics.
Provides threshold sweep curves and Plotly/Matplotlib visualization utilities.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.semantic_matcher import SemanticSkillMatcher
from src.skill_normalizer import SkillNormalizer


# Ground truth evaluation dataset representing diverse real-world matching scenarios:
# - Exact lexical matches (e.g., Python == Python)
# - Acronyms & Abbreviations (e.g., ML == Machine Learning, K8s == Kubernetes)
# - Synonyms & Variants (e.g., text mining == NLP, sklearn == Scikit-Learn)
# - Related domain terms (e.g., PyTorch == Deep Learning)
# - True negatives (e.g., Photoshop != Machine Learning, Java != JavaScript)
BENCHMARK_DATASET = [
    # True Positives / Positive Pairs (is_match = True)
    ("Python", "Python", True, "exact"),
    ("ML", "Machine Learning", True, "acronym"),
    ("Machine Learning", "ML", True, "acronym"),
    ("PyTorch", "PyTorch", True, "exact"),
    ("sklearn", "Scikit-Learn", True, "alias"),
    ("scikit-learn", "Machine Learning", True, "semantic_related"),
    ("deep learning", "Neural Networks", True, "synonym"),
    ("DL", "Deep Learning", True, "acronym"),
    ("NLP", "Natural Language Processing", True, "acronym"),
    ("text mining", "Natural Language Processing", True, "semantic_related"),
    ("Docker", "Containerization", True, "synonym"),
    ("K8s", "Kubernetes", True, "acronym"),
    ("Kubernetes", "Container Orchestration", True, "synonym"),
    ("AWS", "Amazon Web Services", True, "acronym"),
    ("GCP", "Google Cloud Platform", True, "acronym"),
    ("Postgres", "PostgreSQL", True, "alias"),
    ("SQL", "Structured Query Language", True, "acronym"),
    ("pd", "Pandas", True, "alias"),
    ("np", "NumPy", True, "alias"),
    ("FastAPI", "RESTful APIs", True, "semantic_related"),
    ("React.js", "React", True, "alias"),
    ("TypeScript", "JavaScript", True, "semantic_related"),
    ("Continuous Integration", "CI/CD", True, "synonym"),
    ("Terraform", "Infrastructure as Code", True, "synonym"),
    ("Airflow", "Workflow Orchestration", True, "synonym"),
    ("Apache Spark", "PySpark", True, "semantic_related"),
    ("A/B Testing", "Hypothesis Testing", True, "semantic_related"),
    ("EDA", "Exploratory Data Analysis", True, "acronym"),
    ("XGBoost", "Gradient Boosting", True, "semantic_related"),
    ("Kafka", "Event Streaming", True, "synonym"),

    # True Negatives / Negative Pairs (is_match = False)
    ("Java", "JavaScript", False, "false_lexical_similarity"),
    ("Python", "Photoshop", False, "unrelated"),
    ("Docker", "PostgreSQL", False, "different_domain"),
    ("React", "Apache Spark", False, "frontend_vs_bigdata"),
    ("Kubernetes", "CSS3", False, "infra_vs_frontend"),
    ("HTML5", "Machine Learning", False, "unrelated"),
    ("Git", "PyTorch", False, "vcs_vs_ml"),
    ("Tableau", "Ansible", False, "bi_vs_devops"),
    ("SQL", "TypeScript", False, "db_vs_frontend"),
    ("Linux", "Figma", False, "os_vs_design"),
    ("C++", "HTML", False, "systems_vs_markup"),
    ("Redis", "Seaborn", False, "cache_vs_viz"),
    ("Go", "Excel", False, "lang_vs_spreadsheet"),
    ("MongoDB", "Prometheus", False, "nosql_vs_monitoring"),
    ("Jenkins", "Pandas", False, "cicd_vs_datascience"),
    ("Keras", "Tailwind CSS", False, "ml_vs_styling"),
    ("Terraform", "Vue.js", False, "iac_vs_frontend"),
    ("NumPy", "GraphQL", False, "numerical_vs_api"),
    ("Snowflake", "Jest", False, "warehouse_vs_testing"),
    ("Optuna", "Angular", False, "tuning_vs_frontend"),
]


class BenchmarkEvaluator:
    """Evaluates and compares skill matching algorithms against ground truth."""

    def __init__(
        self,
        semantic_matcher: Optional[SemanticSkillMatcher] = None,
        normalizer: Optional[SkillNormalizer] = None,
    ):
        """Initializes evaluator with matching engines and benchmark data."""
        self.normalizer = normalizer or SkillNormalizer()
        self.matcher = semantic_matcher or SemanticSkillMatcher()
        self.dataset = BENCHMARK_DATASET

    @staticmethod
    def naive_keyword_match(term1: str, term2: str) -> bool:
        """Baseline 1: Substring containment and exact equality check."""
        t1 = term1.lower().strip()
        t2 = term2.lower().strip()
        return (t1 == t2) or (t1 in t2) or (t2 in t1)

    @staticmethod
    def jaccard_token_match(term1: str, term2: str, threshold: float = 0.5) -> bool:
        """Baseline 2: Token-level Jaccard similarity."""
        s1 = set(term1.lower().replace("-", " ").replace("_", " ").split())
        s2 = set(term2.lower().replace("-", " ").replace("_", " ").split())
        if not s1 or not s2:
            return False
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return (intersection / union) >= threshold

    def proposed_pipeline_match(self, term1: str, term2: str, threshold: float = 0.75) -> bool:
        """Proposed ML Pipeline: ESCO Normalization + Semantic Cosine Similarity."""
        # 1. Check normalized canonical forms
        n1 = self.normalizer.normalize(term1)["canonical_name"]
        n2 = self.normalizer.normalize(term2)["canonical_name"]
        if n1.lower() == n2.lower():
            return True

        # 2. Semantic matching with SentenceTransformer
        sim_result = self.matcher.match_skills([term1], [term2], threshold=threshold)
        return len(sim_result["matched_pairs"]) > 0

    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """Calculates Precision, Recall, F1, Accuracy, and Specificity."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "Specificity": round(specificity, 4),
        }

    def evaluate_model(self, model_type: str = "proposed", threshold: float = 0.75) -> Dict[str, Any]:
        """Evaluates a specified matching model across the benchmark dataset.

        Args:
            model_type: 'naive', 'jaccard', or 'proposed'.
            threshold: Cosine similarity cutoff for proposed model.

        Returns:
            Dictionary with confusion counts and metrics.
        """
        tp = fp = fn = tn = 0
        error_cases = []

        for term1, term2, ground_truth, scenario in self.dataset:
            if model_type == "naive":
                pred = self.naive_keyword_match(term1, term2)
            elif model_type == "jaccard":
                pred = self.jaccard_token_match(term1, term2)
            else:
                pred = self.proposed_pipeline_match(term1, term2, threshold=threshold)

            if pred and ground_truth:
                tp += 1
            elif pred and not ground_truth:
                fp += 1
                error_cases.append({"term1": term1, "term2": term2, "type": "False Positive", "scenario": scenario})
            elif not pred and ground_truth:
                fn += 1
                error_cases.append({"term1": term1, "term2": term2, "type": "False Negative", "scenario": scenario})
            else:
                tn += 1

        metrics = self.calculate_metrics(tp, fp, fn, tn)
        metrics["model"] = model_type
        metrics["error_cases"] = error_cases
        return metrics

    def run_full_benchmark(self, threshold: float = 0.75) -> Dict[str, Any]:
        """Runs comparative benchmark comparing all models side-by-side."""
        naive = self.evaluate_model(model_type="naive")
        jaccard = self.evaluate_model(model_type="jaccard")
        proposed = self.evaluate_model(model_type="proposed", threshold=threshold)

        return {
            "Naive Keyword Matching": naive,
            "Token Jaccard Baseline": jaccard,
            "Proposed SBERT + ESCO Pipeline": proposed,
        }

    def threshold_sensitivity_sweep(
        self, thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """Sweeps thresholds to determine optimal balance of Precision and Recall."""
        if thresholds is None:
            thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        curve_points = []
        for tau in thresholds:
            m = self.evaluate_model(model_type="proposed", threshold=tau)
            curve_points.append({
                "threshold": tau,
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
            })
        return curve_points

    def create_plotly_comparison_chart(self, benchmark_results: Dict[str, Any]) -> Any:
        """Builds an interactive Plotly grouped bar chart comparing models."""
        try:
            import plotly.graph_objects as go

            models = list(benchmark_results.keys())
            precisions = [benchmark_results[m]["Precision"] * 100 for m in models]
            recalls = [benchmark_results[m]["Recall"] * 100 for m in models]
            f1_scores = [benchmark_results[m]["F1"] * 100 for m in models]

            fig = go.Figure(data=[
                go.Bar(name="Precision (%)", x=models, y=precisions, marker_color="#3B82F6"),
                go.Bar(name="Recall (%)", x=models, y=recalls, marker_color="#10B981"),
                go.Bar(name="F1-Score (%)", x=models, y=f1_scores, marker_color="#8B5CF6"),
            ])
            fig.update_layout(
                barmode="group",
                title="Model Performance Comparison (Baseline vs. Proposed Pipeline)",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 105]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=450,
            )
            return fig
        except ImportError:
            return None

    def create_plotly_threshold_curve(self, sweep_data: List[Dict[str, float]]) -> Any:
        """Builds a Plotly line chart illustrating threshold sensitivity."""
        try:
            import plotly.graph_objects as go

            taus = [pt["threshold"] for pt in sweep_data]
            p = [pt["Precision"] * 100 for pt in sweep_data]
            r = [pt["Recall"] * 100 for pt in sweep_data]
            f1 = [pt["F1"] * 100 for pt in sweep_data]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=taus, y=p, mode="lines+markers", name="Precision", line=dict(color="#3B82F6", width=2)))
            fig.add_trace(go.Scatter(x=taus, y=r, mode="lines+markers", name="Recall", line=dict(color="#10B981", width=2)))
            fig.add_trace(go.Scatter(x=taus, y=f1, mode="lines+markers", name="F1-Score", line=dict(color="#8B5CF6", width=3, dash="dot")))

            fig.update_layout(
                title="Threshold (tau) Sensitivity Sweep on Evaluation Benchmark",
                xaxis_title="Cosine Similarity Cutoff (tau)",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 105]),
                template="plotly_white",
                height=420,
            )
            return fig
        except ImportError:
            return None
