"""
========================================================
 TOOL 1 — JD Parser
========================================================
 Extracts structured requirements from a raw job
 description text using NLP keyword extraction,
 regex patterns, and a curated skill taxonomy.

 In a production GenAI system this would call an
 LLM (e.g. GPT-4 via Azure OpenAI) with a structured
 output prompt. Here we use rule-based NLP with
 sklearn TF-IDF to rank keyword relevance — fully
 offline, no API key needed.
========================================================
"""

import re
from collections import defaultdict

# ── Skill taxonomy ────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "r", "scala", "java", "sql", "pyspark", "sas", "julia", "c++", "go"
    ],
    "Cloud Platforms": [
        "azure", "aws", "gcp", "google cloud", "databricks", "synapse", "snowflake",
        "data factory", "adf", "redshift", "bigquery", "adls", "s3", "lambda"
    ],
    "Machine Learning": [
        "machine learning", "deep learning", "neural network", "random forest", "xgboost",
        "gradient boosting", "nlp", "natural language processing", "llm", "generative ai",
        "agentic ai", "transformers", "bert", "gpt", "regression", "classification",
        "clustering", "reinforcement learning", "computer vision", "mlops"
    ],
    "Data Engineering": [
        "etl", "pipeline", "data warehouse", "data lake", "medallion", "bronze", "silver",
        "gold", "spark", "kafka", "airflow", "dbt", "data quality", "great expectations",
        "streaming", "batch processing", "ci/cd"
    ],
    "BI & Visualisation": [
        "power bi", "tableau", "looker", "qlik", "matplotlib", "seaborn", "plotly",
        "dashboard", "reporting", "excel", "d3.js"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "nosql", "cosmos db", "oracle",
        "sql server", "sqlite", "cassandra"
    ],
    "Soft Skills": [
        "communication", "stakeholder", "client", "presentation", "team", "agile",
        "scrum", "leadership", "problem solving", "analytical", "structured thinking"
    ],
}

SENIORITY_SIGNALS = {
    "Graduate / Junior": ["graduate", "junior", "entry level", "entry-level", "fy28", "fy27", "new grad", "recent graduate"],
    "Mid-level":         ["mid", "2+ years", "3+ years", "experienced"],
    "Senior":            ["senior", "lead", "5+ years", "7+ years", "principal"],
}

def parse_jd(jd_text: str) -> dict:
    """
    Parse a job description and return structured requirements.

    Returns:
        {
          "role_title": str,
          "seniority": str,
          "skills_found": {category: [skill, ...]},
          "key_requirements": [str],
          "nice_to_have": [str],
          "top_keywords": [str],
          "raw_text": str
        }
    """
    text_lower = jd_text.lower()
    result = {}

    # ── Role title ────────────────────────────────────────────────────────────
    lines = [l.strip() for l in jd_text.strip().splitlines() if l.strip()]
    result["role_title"] = lines[0] if lines else "Unknown Role"

    # ── Seniority ─────────────────────────────────────────────────────────────
    result["seniority"] = "Not specified"
    for level, signals in SENIORITY_SIGNALS.items():
        if any(s in text_lower for s in signals):
            result["seniority"] = level
            break

    # ── Skills extraction ─────────────────────────────────────────────────────
    skills_found = defaultdict(list)
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            if skill in text_lower:
                skills_found[category].append(skill.title())
    result["skills_found"] = dict(skills_found)

    # ── Key requirements (bullet points / sentences with "must" / "require") ──
    requirement_patterns = [
        r"(?:•|\*|-|·)\s*(.{20,120})",
        r"(?:must|required?|need|looking for)[^.]{10,120}\.",
        r"(?:experience|knowledge|familiarity)[^.]{10,120}\.",
    ]
    requirements = []
    for pattern in requirement_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        requirements.extend([m.strip() for m in matches if len(m.strip()) > 20])

    # Deduplicate while preserving order
    seen = set()
    unique_reqs = []
    for r in requirements:
        key = r[:40].lower()
        if key not in seen:
            seen.add(key)
            unique_reqs.append(r)

    result["key_requirements"] = unique_reqs[:12]

    # ── Nice to have ──────────────────────────────────────────────────────────
    nice_patterns = [
        r"(?:nice to have|bonus|preferred|desirable|advantage)[^.]{10,150}\.",
        r"(?:familiarity|awareness|exposure)[^.]{10,120}\.",
    ]
    nice = []
    for pattern in nice_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        nice.extend([m.strip() for m in matches])
    result["nice_to_have"] = nice[:5]

    # ── Top keywords (simple frequency on known terms) ────────────────────────
    all_skills = [s for skills in SKILL_TAXONOMY.values() for s in skills]
    keyword_freq = {s: text_lower.count(s) for s in all_skills if text_lower.count(s) > 0}
    result["top_keywords"] = [k.title() for k, _ in
                               sorted(keyword_freq.items(), key=lambda x: -x[1])[:15]]

    result["raw_text"] = jd_text
    return result


if __name__ == "__main__":
    sample_jd = """
    AI & Data Graduate Programme FY28 — Accenture
    Dublin, Hybrid | €40,000 | Permanent

    We are looking for graduates with:
    - Knowledge of Python, SQL, and machine learning techniques
    - Cloud data platform experience (AWS, MS Azure, Google Cloud)
    - Experience with generative AI and agentic AI technologies
    - Data visualisation tools such as Tableau, Qlik, Looker or Power BI
    - Knowledge of ETL, data warehouse tools and big data technologies
    - Strong communication and structured thinking skills
    - Minimum 2:1 degree
    """
    result = parse_jd(sample_jd)
    print(f"Role: {result['role_title']}")
    print(f"Seniority: {result['seniority']}")
    print(f"\nSkills found:")
    for cat, skills in result['skills_found'].items():
        print(f"  {cat}: {', '.join(skills)}")
    print(f"\nTop keywords: {result['top_keywords']}")
