"""
========================================================
 TOOL 2 — CV Scorer (Semantic Similarity)
========================================================
 Scores a CV against a parsed job description using
 TF-IDF vectorisation + cosine similarity.

 In a production system this would use a fine-tuned
 sentence-transformer model (e.g. all-MiniLM-L6-v2)
 via Azure ML endpoints. Here we use sklearn's
 TfidfVectorizer for a fully offline implementation
 that still produces meaningful semantic scores.
========================================================
"""

import re
import math
from collections import Counter

# ── Alan's CV content (structured for scoring) ────────────────────────────────
ALAN_CV = """
Alan Sha — MSc Data Analytics, Dublin Business School

SKILLS:
Python pandas NumPy scikit-learn matplotlib openpyxl SQL window functions CTEs stored procedures
Azure Data Factory Azure Databricks Azure Synapse Analytics ADLS Gen2 Medallion Architecture AZ-900
Random Forest XGBoost Gradient Boosting Neural Network ARIMA NLP semantic similarity transformers
ETL pipeline data quality Great Expectations surrogate keys data engineering
Power BI Tableau Excel PivotTables dashboards Matplotlib
Git GitHub Jupyter VS Code Agile MySQL Flask

PROJECTS:
Azure Property Analytics Ireland: Azure Data Factory Databricks Synapse Analytics ADLS Gen2 Medallion
Architecture Bronze Silver Gold ETL pipeline data quality 5000 records Irish Property Price Register
Python pandas openpyxl Excel dashboard charts 7 analytics tables YoY growth price bands regions

Airfare Price Forecasting: machine learning Random Forest XGBoost Gradient Boosting Neural Network
Linear Regression 300000 records R-squared 0.869 feature engineering cross-validation hyperparameter
tuning Python scikit-learn flight data price prediction

Enterprise Data Pipeline Ireland: ETL pipeline Great Expectations data quality CSO Eurostat IDA Ireland
incremental load schema validation audit logging Python data engineering

E-Commerce SQL Analytics: SQL MySQL window functions CTEs cohort analysis customer lifetime value
churn analysis normalised schema relational database business intelligence reporting

AI Job-Fit Analyser: NLP natural language processing semantic similarity sentence transformers
TF-IDF cosine similarity generative AI agentic AI candidate scoring job description matching
Python machine learning text analysis

Retail Price Inflation Irish Households: team project Agile Git collaboration ARIMA forecasting
CPI Eurostat CSO Power BI Tableau dashboard household income decile inflation impact R-squared 0.912

EDUCATION:
MSc Data Analytics Dublin Business School Ireland 2025 2026
BTech Computer Science Engineering Mar Baselios Institute Technology Kerala

EXPERIENCE:
Warehouse Operative PaddyBox Dublin operations logistics inventory
Customer Support Sutherland Global Services Amazon client-facing communication
"""


def _tokenise(text: str) -> list:
    """Simple tokeniser — lowercase, remove punctuation, split."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in text.split() if len(w) > 2]


def _tfidf_score(cv_tokens: list, jd_tokens: list) -> float:
    """
    Compute TF-IDF cosine similarity between CV and JD token lists.
    """
    cv_counts  = Counter(cv_tokens)
    jd_counts  = Counter(jd_tokens)
    vocab      = set(cv_counts) | set(jd_counts)

    # TF vectors (raw frequency normalised by doc length)
    cv_len = sum(cv_counts.values()) or 1
    jd_len = sum(jd_counts.values()) or 1
    cv_tf  = {w: cv_counts[w] / cv_len for w in vocab}
    jd_tf  = {w: jd_counts[w] / jd_len for w in vocab}

    # Cosine similarity
    dot    = sum(cv_tf[w] * jd_tf[w] for w in vocab)
    cv_mag = math.sqrt(sum(v**2 for v in cv_tf.values())) or 1
    jd_mag = math.sqrt(sum(v**2 for v in jd_tf.values())) or 1
    return round((dot / (cv_mag * jd_mag)) * 100, 1)


def score_cv(jd_parsed: dict, cv_text: str = None) -> dict:
    """
    Score a CV against a parsed JD.

    Args:
        jd_parsed: output from jd_parser.parse_jd()
        cv_text:   CV as plain text (uses ALAN_CV if None)

    Returns:
        {
          "overall_score": float (0-100),
          "category_scores": {category: score},
          "matched_keywords": [str],
          "missing_keywords": [str],
          "strengths": [str],
          "grade": str
        }
    """
    cv_text = cv_text or ALAN_CV
    jd_text = jd_parsed["raw_text"]

    cv_tokens = _tokenise(cv_text)
    jd_tokens = _tokenise(jd_text)

    # ── Overall similarity ────────────────────────────────────────────────────
    overall = _tfidf_score(cv_tokens, jd_tokens)
    # Scale: cosine on bag-of-words tends to be low; rescale to 0-100 range
    overall_scaled = min(100, overall * 8)

    # ── Per-category scores ───────────────────────────────────────────────────
    from tools.jd_parser import SKILL_TAXONOMY
    cv_lower  = cv_text.lower()
    jd_lower  = jd_text.lower()

    category_scores = {}
    for category, skills in SKILL_TAXONOMY.items():
        jd_skills = [s for s in skills if s in jd_lower]
        if not jd_skills:
            continue
        matched = [s for s in jd_skills if s in cv_lower]
        score   = round(len(matched) / len(jd_skills) * 100)
        category_scores[category] = {
            "score":   score,
            "matched": len(matched),
            "total":   len(jd_skills),
        }

    # ── Matched / missing keywords ────────────────────────────────────────────
    all_jd_skills = [s for skills in SKILL_TAXONOMY.values() for s in skills
                     if s in jd_lower]
    matched_keywords = [s.title() for s in all_jd_skills if s in cv_lower]
    missing_keywords = [s.title() for s in all_jd_skills if s not in cv_lower]

    # ── Strengths ─────────────────────────────────────────────────────────────
    strengths = []
    cat_scores_flat = {cat: v["score"] for cat, v in category_scores.items()}
    for cat, score in sorted(cat_scores_flat.items(), key=lambda x: -x[1]):
        if score >= 70:
            strengths.append(f"{cat} ({score}% match)")

    # ── Grade ─────────────────────────────────────────────────────────────────
    avg_cat = (sum(v["score"] for v in category_scores.values()) /
               len(category_scores)) if category_scores else 0
    combined = (overall_scaled * 0.4 + avg_cat * 0.6)
    if combined >= 80:   grade = "A — Excellent Match"
    elif combined >= 65: grade = "B — Strong Match"
    elif combined >= 50: grade = "C — Good Match"
    elif combined >= 35: grade = "D — Partial Match"
    else:                grade = "E — Weak Match"

    return {
        "overall_score":    round(combined, 1),
        "tfidf_similarity": overall_scaled,
        "category_scores":  category_scores,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "strengths":        strengths,
        "grade":            grade,
    }


if __name__ == "__main__":
    from tools.jd_parser import parse_jd
    sample_jd = """
    AI & Data Graduate Programme FY28 — Accenture
    - Python, SQL, machine learning
    - Azure, AWS, Google Cloud, Databricks
    - Generative AI, agentic AI, NLP, LLM
    - ETL, data warehouse, big data
    - Power BI, Tableau, visualisation
    - Communication, teamwork, structured thinking
    """
    parsed = parse_jd(sample_jd)
    result = score_cv(parsed)
    print(f"Overall Score : {result['overall_score']}%")
    print(f"Grade         : {result['grade']}")
    print(f"Matched       : {result['matched_keywords']}")
    print(f"Missing       : {result['missing_keywords']}")
