# 🎯 Resume Skill Analyzer
### *An ML-Based System to Identify Missing Skills in a Resume Against a Target Job Role*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5.svg)](https://spacy.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary & Architecture

The **Resume Skill Analyzer** is a machine-learning application designed to bridge the semantic divide between candidate resumes and modern job requirements. Traditional applicant tracking systems (ATS) rely heavily on naive substring keyword searches, resulting in high false-negative rates when candidates describe equivalent competencies using synonyms, acronyms, or related tooling (e.g., `ML` vs `Machine Learning`, `K8s` vs `Kubernetes`, `PyTorch` vs `Deep Learning`).

This project implements an end-to-end NLP and Representation Learning pipeline:
1. **Multi-Format Parsing:** Robust ingestion of `.pdf`, `.docx`, and `.txt` files with formatting noise removal.
2. **Entity Recognition (NER):** A custom spaCy NER model can be trained for the `SKILL` entity class and is combined with rule-based taxonomy phrase matching. Until trained model artifacts are committed, extraction uses the available spaCy/taxonomy fallback.
3. **Taxonomy Normalization:** Canonicalization of extracted variants to standard **ESCO / O*NET** concepts using RapidFuzz string distance and alias indexing.
4. **Semantic Embedding:** Optional dense vector generation via **Sentence-BERT (`all-MiniLM-L6-v2`)**; lightweight character/token n-gram similarity is used by default for easier cloud deployment.
5. **Calibrated Thresholding:** Directional cosine similarity matching using an empirically calibrated threshold ($\tau \approx 0.75$).
6. **Gap Algebra & Scoring:** Quantitative match percentage calculation and targeted learning recommendations.
7. **Interactive Dashboard:** Modern multi-tab Streamlit web application.

```mermaid
flowchart TD
    A["Raw Resume (PDF / DOCX / TXT)"] --> B["Resume Parser & Cleaner\n(pdfplumber / python-docx)"]
    B --> C["Skill Entity Extractor\n(spaCy NER + PhraseMatcher)"]
    C --> D["ESCO Taxonomy Normalizer\n(RapidFuzz + Alias Index)"]
    
    E["Job Description / Predefined Roles"] --> F["Job Skill Extractor\n(Requirement Segmentation)"]
    F --> G["Target Role Canonical Skills"]
    
    D --> H["Candidate Skill Vectors\n(optional Sentence-BERT / default n-grams)"]
    G --> I["Job Requirement Vectors\n(optional Sentence-BERT / default n-grams)"]
    
    H & I --> J["Pairwise Cosine Similarity Matrix\nS_ij = (u_i · v_j) / (||u_i|| ||v_j||)"]
    J --> K{"Threshold Check\n(tau >= 0.75)"}
    
    K -->|"Yes (>= 0.75)"| L["Matched Skills (Green)"]
    K -->|"No (< 0.75)"| M["Missing Skills (Red)"]
    
    L & M --> N["Gap Analyzer\nMatch % = (|Matched| / |Total Required|) * 100"]
    N --> O["Streamlit Interactive Dashboard\n(Visualization & Recommendations)"]
```

---

## 📐 Mathematical Foundations

### 1. Normalized Cosine Similarity
For candidate skill embedding $\mathbf{u} \in \mathbb{R}^d$ and job requirement embedding $\mathbf{v} \in \mathbb{R}^d$:
$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{k=1}^d u_k v_k}{\sqrt{\sum_{k=1}^d u_k^2} \sqrt{\sum_{k=1}^d v_k^2}}$$

### 2. Candidate Match Percentage
Given the set of canonical required job skills $J_{\text{req}}$ and matched subset $J_{\text{matched}} \subseteq J_{\text{req}}$:
$$\text{Match Percentage} = \left( \frac{|J_{\text{matched}}|}{|J_{\text{req}}|} \right) \times 100\%$$

### 3. Evaluation Metrics
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F}_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 📂 Project Directory Structure

```text
Resume_Skill_Analyzer/
├── dataset/
│   ├── resumes/               # Sample resumes (PDF, DOCX, and TXT)
│   ├── jobs/                  # Predefined job descriptions (CSV & JSON)
│   ├── skills/                # ESCO skills taxonomy dataset (CSV)
│   └── annotations.json       # Ground-truth entity spans for NER training
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA on taxonomy & skill distributions
│   ├── 02_preprocessing.ipynb       # Document parsing & text sanitization
│   ├── 03_train_ner.ipynb           # spaCy NER pipeline training & DocBins
│   ├── 04_semantic_matching.ipynb   # Sentence-BERT embeddings & similarity matrices
│   └── 05_model_evaluation.ipynb    # Empirical benchmark against naive baselines
├── models/
│   ├── skill_ner/             # Saved fine-tuned spaCy NER model & manifests
│   ├── train.spacy            # Generated training DocBin (local)
│   ├── dev.spacy              # Generated validation DocBin (local)
│   └── test.spacy             # Generated test DocBin (local)
├── src/
│   ├── __init__.py            # Package initialization & exports
│   ├── resume_parser.py       # PDF/DOCX parsing (pdfplumber, docx, fallbacks)
│   ├── skill_extractor.py     # Hybrid spaCy NER & PhraseMatcher engine
│   ├── skill_normalizer.py    # RapidFuzz + ESCO taxonomy canonicalization
│   ├── job_skill_extractor.py # Job requirement extraction & role catalog
│   ├── semantic_matcher.py    # Optional SBERT / lightweight similarity scoring
│   ├── gap_analyzer.py        # Skill gap algebra & upskilling recommendations
│   └── evaluation.py          # Benchmark metrics (P, R, F1) & Plotly charts
├── tests/
│   └── test_pipeline.py       # Comprehensive unit test suite (19 test cases)
├── app.py                     # Streamlit multi-tab interactive UI dashboard
├── train_ner.py               # Standalone spaCy training script
├── requirements.txt           # Lightweight Streamlit runtime dependencies
├── requirements-ml.txt        # Optional training / research dependencies
└── README.md                  # Project documentation & manual
```

---

## ⚡ Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Bhumigarg1605/resume_skill_anayzer.git
cd resume_skill_anayzer
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # On macOS/Linux
# .venv\Scripts\activate         # On Windows
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

For NER training, Sentence-BERT, and notebook/research tools, install the optional dependencies:
```bash
pip install -r requirements-ml.txt
```

### 4. (Optional) Download Base spaCy English Model
```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Execution & Usage

### A. Run the Interactive Streamlit Web Dashboard
Launch the multi-tab user interface:
```bash
streamlit run app.py
```
* **Tab 1 (Skill Gap Analyzer):** Upload any `.pdf` / `.docx` / `.txt` resume or select a sample resume, choose a target tech role, and click **Analyze Resume**.
* **Tab 2 (Model Performance & Benchmarks):** Inspect interactive Plotly benchmark bar charts and threshold sensitivity curves.
* **Tab 3 (Semantic Inspector):** Interactively test cosine similarity between arbitrary technical phrases.
* **Tab 4 (ESCO Taxonomy Browser):** Filter and search through 110+ canonical technical concepts.

### B. Train / Fine-Tune the Optional NER Model
The repository contains 300 annotated records (210 with skill entities and 90 without). The script uses a reproducible 70:15:15 split (210 train, 45 validation, 45 test). Install `requirements-ml.txt`, then run:
```bash
python train_ner.py --epochs 30 --dropout 0.2 --output models/skill_ner
```

Training generates serialized spaCy model files locally in `models/skill_ner/`. The committed repository currently contains only the manifest and taxonomy patterns, not trained model weights. After a successful local run, verify the generated files and commit them if you want Streamlit Cloud to load that trained model. Otherwise, the app uses the taxonomy matcher fallback.

### C. Run the Unit Test Suite
Execute the automated test suite covering all modules:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
*(Run the tests locally and use the actual test output when reporting results.)*

---

## 📊 Empirical Benchmarking Results

The application includes a small, hand-curated benchmark of technical skill pairs covering exact matches, aliases, acronyms, related terms, and negative examples. These are illustrative internal test cases, not an independently collected or externally validated benchmark. The displayed metrics are calculated from this fixed set and can vary with the selected similarity mode and threshold:

| Model Architecture | Precision (%) | Recall (%) | F1-Score (%) | Key Limitations |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Keyword Matcher** | 80.0% | 13.3% | 22.9% | Fails on acronyms (`ML`), variants (`sklearn`), and synonyms |
| **Token Jaccard Matcher** | 100.0% | 6.7% | 12.5% | Fails on single-word aliases and abbreviation mismatches |
| **Proposed semantic + taxonomy pipeline** | **100.0%** | **90.0%** | **94.7%** | **Captures semantic synonyms while rejecting cross-domain false matches** |

### Why Semantic ML Outperforms Keyword Search
* **Acronym Resolution:** `K8s` correctly resolves to `Kubernetes`; `DL` resolves to `Deep Learning`.
* **Semantic Cross-Matching:** `Docker` aligns to `Containerization` with cosine similarity $\approx 0.82 \ge 0.75$.
* **False Positive Prevention:** `Java` vs `JavaScript` is treated as a distinct concept rather than an accidental substring match.

---

## 📓 Jupyter Notebooks Overview

All notebooks in `notebooks/` follow rigorous ML storytelling and data analysis practices:
1. `01_data_exploration.ipynb`: Analyzes taxonomy domain distributions and job skill frequencies.
2. `02_preprocessing.ipynb`: Demonstrates document parsing, ligature sanitization, and symbol preservation.
3. `03_train_ner.ipynb`: Formats spaCy DocBin objects, configures NER labels, and trains the model.
4. `04_semantic_matching.ipynb`: Visualizes cosine similarity heatmaps and threshold sweeps.
5. `05_model_evaluation.ipynb`: Benchmarks P/R/F1 scores and conducts qualitative error analysis.

---

## 📜 License
No license file is currently included in this repository. Do not assume the project is MIT-licensed unless the repository owner adds an explicit license.
