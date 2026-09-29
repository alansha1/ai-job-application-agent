"""
========================================================
 TOOL 4 — Interview Coach (GenAI Layer)
========================================================
 Generates tailored interview questions + model answers
 using a template-based generative approach.

 In a production system this would call:
   - Azure OpenAI Service (GPT-4o) via REST API
   - Prompt: structured JSON with role, skills, CV
   - Output: JSON with questions + answers

 Here we use a rich template engine that personalises
 questions and answers from Alan's actual CV data —
 producing output indistinguishable from LLM output
 for portfolio demonstration purposes.
========================================================
"""

import random

# ── Question bank — categorised ───────────────────────────────────────────────

TECHNICAL_TEMPLATES = [
    {
        "q": "Walk me through how you designed the Medallion Architecture in your Azure Property Analytics project.",
        "a": (
            "I structured the pipeline into three layers mirroring a production Azure stack. "
            "The Bronze layer — simulating Azure Data Factory — ingested 5,000 raw records from the Irish "
            "Property Price Register with no transformations, preserving the original data quality issues "
            "like missing prices and casing inconsistencies. The Silver layer, simulating Azure Databricks, "
            "applied 9 transformation steps including date parsing, outlier removal, price band derivation, "
            "and region mapping — achieving a 95.4% data retention rate and logging a full quality report. "
            "The Gold layer, simulating Azure Synapse Analytics, produced 7 analytics-ready tables consumed "
            "by an 8-sheet Excel dashboard. I designed it this way because in production you never want "
            "transformations in the ingestion layer — raw data should be preserved for auditability."
        ),
        "tags": ["Cloud", "Data Engineering", "Azure"],
    },
    {
        "q": "How did you approach model selection in your airfare forecasting project, and why did Random Forest outperform the others?",
        "a": (
            "I benchmarked five models systematically: Linear Regression as a baseline, then Random Forest, "
            "XGBoost, Gradient Boosting, and a Neural Network on 300,000 real flight records. Random Forest "
            "achieved R²=0.869, outperforming the others primarily because of its robustness to the "
            "non-linear interactions between features — particularly between days-to-departure, route "
            "competitiveness, and seasonal demand. Linear regression couldn't capture those interactions, "
            "while the neural network overfit on the training set without enough hyperparameter tuning. "
            "I used cross-validation throughout to avoid data leakage, and the key insight was that "
            "advance booking window was the strongest single predictor of price."
        ),
        "tags": ["Machine Learning", "Python"],
    },
    {
        "q": "Explain how semantic similarity works in your AI Job-Fit Analyser and how you'd extend it with a large language model.",
        "a": (
            "The current implementation uses TF-IDF vectorisation with cosine similarity to score how "
            "closely a CV's vocabulary matches a job description's vocabulary. It works well for keyword "
            "matching but misses semantic equivalences — for example, 'data pipeline' and 'ETL workflow' "
            "mean the same thing but score as different. To extend it with an LLM, I'd replace TF-IDF "
            "with a sentence-transformer model like all-MiniLM-L6-v2, which maps sentences to a "
            "768-dimensional embedding space where semantically similar sentences are geometrically close. "
            "For a fully agentic version, I'd add a reasoning layer — using GPT-4o via Azure OpenAI — "
            "that not only scores the match but explains the gap and generates tailored cover letter "
            "bullets automatically. That's essentially what this project does."
        ),
        "tags": ["NLP", "Generative AI", "Agentic AI"],
    },
    {
        "q": "How did you handle data quality in your enterprise pipeline, and what would you do differently in a production Azure environment?",
        "a": (
            "In the enterprise pipeline I used Great Expectations to define 15+ validation rules — "
            "checking for null rates, value ranges, referential integrity, and schema consistency. "
            "Any record failing validation was routed to a quarantine layer with a rejection reason logged. "
            "In a production Azure environment I'd add a few things: first, I'd use Azure Data Factory's "
            "built-in data flow validation combined with Azure Monitor alerts so the data team is notified "
            "immediately when quality degrades. Second, I'd implement a data lineage layer using Azure Purview "
            "so every column's origin is traceable end-to-end. Third, I'd move from batch validation to "
            "real-time stream validation using Azure Event Hubs if the data were arriving continuously."
        ),
        "tags": ["Data Engineering", "Azure", "Data Quality"],
    },
    {
        "q": "What SQL window functions have you used and when would you use ROW_NUMBER vs RANK vs DENSE_RANK?",
        "a": (
            "In my e-commerce SQL project I used ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, and "
            "SUM OVER PARTITION BY extensively. The distinction between the three ranking functions matters "
            "when there are ties. ROW_NUMBER gives every row a unique number regardless of ties — useful "
            "when you need to deduplicate and pick exactly one row per group. RANK skips numbers after "
            "ties — so if two rows tie for position 2, the next rank is 4 — useful for competition-style "
            "rankings. DENSE_RANK doesn't skip — the next rank after two ties at 2 is 3 — better for "
            "customer segmentation where you want contiguous tier numbers. In my cohort analysis I used "
            "LAG to calculate month-over-month retention and LEAD to identify customers likely to churn "
            "before they actually did."
        ),
        "tags": ["SQL", "Analytics"],
    },
]

BEHAVIOURAL_TEMPLATES = [
    {
        "q": "Tell me about a time you had to explain a technical finding to a non-technical audience.",
        "a": (
            "In the retail price inflation project, our team needed to present the household impact "
            "of CPI trends to a non-technical panel. The challenge was that ARIMA forecasting outputs "
            "are inherently statistical and easy to misread. I built a Power BI dashboard that replaced "
            "confidence intervals with plain-language ranges — 'prices likely to rise between 3% and 5%' "
            "instead of showing error bands. I also restructured the findings around income deciles — "
            "showing that the bottom quartile of households experienced 40% more inflationary pressure "
            "than the top quartile — because that framing made the data immediately actionable for "
            "policy purposes. The panel were able to engage with the findings directly without needing "
            "to understand the model."
        ),
        "tags": ["Communication", "Soft Skills"],
    },
    {
        "q": "Describe a situation where your analysis led to a specific recommendation or decision.",
        "a": (
            "In the airfare forecasting project, my analysis showed that the advance booking window "
            "was the strongest single predictor of price — flights booked more than 60 days out were "
            "on average 34% cheaper than last-minute bookings on the same routes. That's a concrete, "
            "actionable insight for any travel business: the recommendation is to design loyalty "
            "programmes and promotions that incentivise early booking, because the price elasticity "
            "is highest in the 60-30 day window. I presented this as a business recommendation "
            "alongside the model metrics, because an R² of 0.869 is only useful if someone knows "
            "what to do with it."
        ),
        "tags": ["Analytics", "Business Impact"],
    },
    {
        "q": "How do you approach learning a new technology quickly?",
        "a": (
            "My approach is always to build something real with it rather than just following tutorials. "
            "When I needed to understand the Azure data stack for my portfolio, I didn't just read the "
            "documentation — I designed a full Medallion Architecture pipeline that mirrors production "
            "Azure services, built it in Python, and mapped every component to its Azure equivalent. "
            "That forced me to understand not just what the services do but why you'd choose one over "
            "another — for example, why you'd use Azure Databricks for transformation rather than "
            "doing it inside ADF directly. I find that a concrete project exposes the edges of what "
            "I don't know far faster than any course."
        ),
        "tags": ["Growth Mindset", "Soft Skills"],
    },
    {
        "q": "Tell me about a time you worked in a team and had to manage disagreements.",
        "a": (
            "In the retail price inflation group project, we had a disagreement early on about which "
            "model to prioritise — two teammates wanted to go straight to neural networks, while I "
            "argued we should establish ARIMA as a baseline first so we had something to beat. "
            "Rather than just asserting a position, I suggested we allocate two days to each approach "
            "in parallel and compare on a held-out test set. The ARIMA baseline ended up achieving "
            "R²=0.912, which actually outperformed the neural network on our dataset size. "
            "The experience taught me that technical disagreements are best resolved with evidence "
            "rather than opinion — and that setting up a fair comparison early saves a lot of "
            "time later."
        ),
        "tags": ["Teamwork", "Soft Skills"],
    },
]

ACCENTURE_SPECIFIC = [
    {
        "q": "Why Accenture specifically, and why the AI & Data graduate programme?",
        "a": (
            "Accenture's AI & Data practice is distinctive because it works across the full stack — "
            "from AI strategy through to technical delivery at scale. Most organisations either do "
            "the strategy or the engineering; Accenture does both, which is where I want to develop. "
            "The graduate programme's structure — live client projects from week one, combined with "
            "dedicated training in Gen AI and analytics strategy — matches exactly how I learn best: "
            "by doing real work on real problems. And frankly, the question every major organisation "
            "is working on right now — how do we get real value from AI — is the most interesting "
            "problem in data right now. I want to be in the room where that question gets answered."
        ),
        "tags": ["Motivation", "Accenture"],
    },
    {
        "q": "Where do you see generative AI having the most impact on enterprise data work in the next two years?",
        "a": (
            "I think the biggest near-term impact will be in data engineering rather than consumer-facing "
            "AI. Specifically, LLMs are already good enough to generate SQL, write dbt transformation "
            "logic, and explain data quality anomalies in plain language — which compresses the feedback "
            "loop between a business analyst asking a question and a data engineer building the pipeline "
            "to answer it. The second area is agentic orchestration — multi-step AI pipelines that can "
            "monitor a data product, detect drift, diagnose the root cause, and trigger a remediation "
            "workflow without human intervention. That's what I was exploring in this AI Job Application "
            "Agent project — a pipeline where each tool hands its output to the next and the agent "
            "synthesises a recommendation. At enterprise scale, that pattern applied to data operations "
            "would be transformative."
        ),
        "tags": ["Generative AI", "Accenture", "Vision"],
    },
]


def generate_interview_prep(jd_parsed: dict, gap_result: dict, n_technical=3, n_behavioural=2) -> dict:
    """
    Generate personalised interview questions and model answers.

    Returns:
        {
          "technical":   [{q, a, tags}],
          "behavioural": [{q, a, tags}],
          "accenture":   [{q, a, tags}],
          "prep_tips":   [str]
        }
    """
    jd_lower = jd_parsed["raw_text"].lower()

    # Score technical questions by relevance to JD
    def relevance(q_dict):
        return sum(1 for tag in q_dict["tags"] if tag.lower() in jd_lower)

    ranked_technical   = sorted(TECHNICAL_TEMPLATES,   key=relevance, reverse=True)
    ranked_behavioural = sorted(BEHAVIOURAL_TEMPLATES, key=relevance, reverse=True)

    prep_tips = [
        "Use the STAR method (Situation, Task, Action, Result) for every behavioural question.",
        "Always quantify your outcomes — R²=0.869, +24% growth, 95.4% retention rate.",
        "For Accenture: frame answers as 'business problem → technical solution → business outcome'.",
        "Prepare a 90-second project walkthrough for each GitHub project — practice out loud.",
        "Research Accenture's latest AI announcements before the interview — show genuine curiosity.",
        "Have 2-3 questions ready to ask the interviewer about the type of client work you'd do.",
    ]

    return {
        "technical":   ranked_technical[:n_technical],
        "behavioural": ranked_behavioural[:n_behavioural],
        "accenture":   ACCENTURE_SPECIFIC,
        "prep_tips":   prep_tips,
    }


if __name__ == "__main__":
    from tools.jd_parser import parse_jd
    from tools.cv_scorer import score_cv
    from tools.gap_analyser import analyse_gaps

    jd_text = "AI Data Graduate Accenture Python Azure machine learning generative AI agentic NLP ETL Power BI"
    parsed  = parse_jd(jd_text)
    scores  = score_cv(parsed)
    gaps    = analyse_gaps(parsed, scores)
    prep    = generate_interview_prep(parsed, gaps)

    print(f"Generated {len(prep['technical'])} technical, "
          f"{len(prep['behavioural'])} behavioural, "
          f"{len(prep['accenture'])} Accenture-specific questions.")
    print(f"\nSample Q: {prep['technical'][0]['q']}")
