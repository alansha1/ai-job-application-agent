"""
========================================================
 TOOL 3 — Gap Analyser
========================================================
 Compares CV skills against JD requirements to surface:
  - Skills to highlight (strengths to lead with)
  - Skill gaps (missing from CV but in JD)
  - Quick wins (easy gaps to bridge before applying)
  - Application strategy recommendations
========================================================
"""

# ── Bridgeable gaps (things you can address quickly) ─────────────────────────
QUICK_WINS = {
    "Az-900":           "Sit the AZ-900 exam — 2-3 days study, ~€135. Already in progress.",
    "Aws":              "Add an AWS Free Tier project (S3 + Lambda) — 1 day.",
    "Gcp":              "Google Cloud offers free $300 credits — spin up a BigQuery project.",
    "Kafka":            "Build a simple Kafka producer/consumer on localhost — half a day.",
    "Airflow":          "Add an Airflow DAG to your Azure project — 1-2 days.",
    "Dbt":              "dbt Core is free — wrap your Gold layer transformations in dbt models.",
    "Pyspark":          "Run PySpark locally or on Databricks Community Edition (free).",
    "Llm":              "Use Hugging Face free models to extend the AI Job-Fit Analyser.",
    "Generative Ai":    "This project (AI Job Application Agent) directly fills this gap.",
    "Agentic Ai":       "This project (AI Job Application Agent) directly fills this gap.",
    "Bert":             "Use a pre-trained BERT model from Hugging Face — extend NLP project.",
    "Looker":           "Looker Studio (free) — rebuild one of your Power BI dashboards there.",
    "Qlik":             "Qlik has a free personal edition — add it to your skills list.",
}

# ── Highlight recommendations per category ────────────────────────────────────
HIGHLIGHT_ADVICE = {
    "Cloud Platforms": (
        "Lead with your Azure Medallion Architecture project — it's your strongest cloud signal. "
        "Name ADF, Databricks, and Synapse explicitly in interviews."
    ),
    "Machine Learning": (
        "Reference R²=0.869 from your airfare forecasting project — a concrete metric stands out. "
        "Mention the 5-model benchmark comparison as evidence of systematic thinking."
    ),
    "Data Engineering": (
        "Your Bronze→Silver→Gold pipeline with 9 transformation steps and a logged quality report "
        "shows production-grade thinking. Use this as your main data engineering story."
    ),
    "BI & Visualisation": (
        "Mention both Power BI and Tableau — and that you've delivered dashboards to a non-technical "
        "audience (retail inflation project). Client communication is key for Accenture."
    ),
    "Soft Skills": (
        "Your retail project was a group project using Agile — always mention this. "
        "Accenture's JD explicitly calls out team and leadership experience."
    ),
    "Programming Languages": (
        "Python is your core language — back it up with the volume and variety of projects. "
        "SQL is equally important for Accenture — lead with the window functions and CTEs."
    ),
}


def analyse_gaps(jd_parsed: dict, score_result: dict) -> dict:
    """
    Analyse skill gaps and produce actionable recommendations.

    Args:
        jd_parsed:    output from jd_parser.parse_jd()
        score_result: output from cv_scorer.score_cv()

    Returns:
        {
          "highlights":        [str],
          "gaps":              [str],
          "quick_wins":        [{skill, action}],
          "strategy":          [str],
          "application_tips":  [str]
        }
    """

    missing = score_result.get("missing_keywords", [])
    category_scores = score_result.get("category_scores", {})

    # ── Highlights (categories scoring ≥ 70%) ────────────────────────────────
    highlights = []
    for cat, data in sorted(category_scores.items(), key=lambda x: -x[1]["score"]):
        if data["score"] >= 60 and cat in HIGHLIGHT_ADVICE:
            highlights.append({
                "category": cat,
                "score":    data["score"],
                "advice":   HIGHLIGHT_ADVICE[cat],
            })

    # ── Gaps with quick wins ───────────────────────────────────────────────────
    gaps = []
    quick_wins = []
    for skill in missing:
        if skill in QUICK_WINS:
            quick_wins.append({
                "skill":  skill,
                "action": QUICK_WINS[skill],
            })
        else:
            gaps.append(skill)

    # ── Application strategy ──────────────────────────────────────────────────
    overall = score_result.get("overall_score", 0)
    strategy = []

    if overall >= 75:
        strategy.append("✅ Strong match — apply now, don't wait to fill every gap.")
    elif overall >= 55:
        strategy.append("✅ Good match — apply now and address 1-2 quick wins in parallel.")
    else:
        strategy.append("⚠️  Partial match — address the top quick wins before applying.")

    if any("Generative Ai" in qw["skill"] or "Agentic Ai" in qw["skill"]
           for qw in quick_wins):
        strategy.append(
            "🤖 This project (AI Job Application Agent) directly addresses the GenAI/agentic gap "
            "— add it to your CV and GitHub before submitting."
        )

    if "AZ-900" in str(jd_parsed.get("skills_found", {})):
        strategy.append(
            "☁️  Sit the AZ-900 exam as soon as possible — it removes the only remaining cloud gap."
        )

    strategy.append(
        "📝 Tailor your cover letter to the top 3 JD keywords: "
        + ", ".join(score_result.get("matched_keywords", [])[:3]) + "."
    )

    # ── Application tips ──────────────────────────────────────────────────────
    application_tips = [
        "Open your cover letter by naming a specific business problem + your technical solution.",
        "Use the exact JD keywords in your CV — ATS systems scan for phrase matches.",
        "Quantify every project outcome: R²=0.869, +24% price growth, 95.4% data retention.",
        "For Accenture: frame every project as 'business problem → data solution → insight'.",
        "Mention the group retail project early — Accenture values teamwork above all else.",
    ]

    return {
        "highlights":       highlights,
        "gaps":             gaps,
        "quick_wins":       quick_wins,
        "strategy":         strategy,
        "application_tips": application_tips,
    }


if __name__ == "__main__":
    from tools.jd_parser import parse_jd
    from tools.cv_scorer import score_cv

    jd_text = """
    AI & Data Graduate Programme FY28 — Accenture
    - Python, SQL, machine learning
    - Azure, Databricks, Synapse, ADLS
    - Generative AI, agentic AI, NLP, LLM
    - ETL, data warehouse, big data
    - Power BI, Tableau, visualisation
    - Communication, teamwork, Agile
    """
    parsed = parse_jd(jd_text)
    scores = score_cv(parsed)
    gaps   = analyse_gaps(parsed, scores)

    print("HIGHLIGHTS:")
    for h in gaps["highlights"]:
        print(f"  [{h['score']}%] {h['category']}: {h['advice'][:80]}...")

    print("\nQUICK WINS:")
    for qw in gaps["quick_wins"]:
        print(f"  {qw['skill']}: {qw['action'][:80]}")

    print("\nSTRATEGY:")
    for s in gaps["strategy"]:
        print(f"  {s}")
