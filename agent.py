"""
========================================================
 AI JOB APPLICATION AGENT — Main Orchestrator
========================================================
 An agentic pipeline that coordinates 4 specialised
 tools to analyse a job description against a CV and
 produce a comprehensive application prep report.

 AGENT PIPELINE:
   Input: Job Description (text)
       │
       ▼
   Tool 1: JD Parser        → extracts structured requirements
       │
       ▼
   Tool 2: CV Scorer        → semantic similarity score
       │
       ▼
   Tool 3: Gap Analyser     → gaps, highlights, strategy
       │
       ▼
   Tool 4: Interview Coach  → tailored Q&A (GenAI layer)
       │
       ▼
   Output: HTML Report      → output/report_<role>.html

 In production Azure:
   Orchestration → Azure AI Studio Agent Framework
   LLM calls     → Azure OpenAI Service (GPT-4o)
   Storage       → Azure Blob Storage
   Trigger       → Azure Logic App / API endpoint
========================================================

 Usage:
   python agent.py                        # uses built-in Accenture JD
   python agent.py --jd path/to/jd.txt   # custom JD from file
   python agent.py --jd "paste JD text"  # inline JD text
========================================================
"""

import sys
import os
import re
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.jd_parser      import parse_jd
from tools.cv_scorer       import score_cv
from tools.gap_analyser    import analyse_gaps
from tools.interview_coach import generate_interview_prep

# ── Default JD (Accenture AI & Data Graduate FY28) ───────────────────────────
DEFAULT_JD = """
AI & Data Graduate Programme FY28
Accenture · Dublin, County Dublin · Hybrid · €40,000 · Permanent

We are looking for graduates who combine strong analytical and technical ability
with the confidence to work directly with clients.

We are looking for graduates who are familiar with any number of the following:
- Knowledge of how data and AI can be used to solve business problems
- Experience with generative AI and agentic AI technologies
- Cloud data platform experience/knowledge (AWS, MS Azure, Google Cloud Platform)
- Data science knowledge with any of Python, SAS, R, Scala, PySpark or other relevant coding language
- Knowledge of machine learning modelling techniques
- Knowledge of data warehouse tools and database technologies e.g. SQL, ETL, NoSQL and other big data technologies
- Experience in or knowledge of data visualisation tools such as Tableau, Qlik, Looker or Power BI

As well as technical ability, you should be able to demonstrate:
- Structured thinking and the ability to break a complex problem into clear components
- Genuine curiosity about how organisations operate and where technology can help
- Strong verbal/written communication and presentation skills
- Ability to adapt and learn in a continuously changing environment
- Experience in team or leadership roles
- All candidates will have a minimum 2:1 degree
"""


def run_agent(jd_text: str, cv_text: str = None) -> dict:
    """
    Run the full 4-tool agentic pipeline.

    Returns the complete analysis as a dict.
    """
    print("\n🤖 AI Job Application Agent — Starting pipeline...\n")

    # ── Tool 1: Parse JD ──────────────────────────────────────────────────────
    print("  [Tool 1/4] JD Parser        → Extracting requirements...")
    jd_parsed = parse_jd(jd_text)
    print(f"             Role: {jd_parsed['role_title']}")
    print(f"             Seniority: {jd_parsed['seniority']}")
    print(f"             Skills found: {sum(len(v) for v in jd_parsed['skills_found'].values())} across "
          f"{len(jd_parsed['skills_found'])} categories")

    # ── Tool 2: Score CV ──────────────────────────────────────────────────────
    print("\n  [Tool 2/4] CV Scorer        → Computing semantic similarity...")
    score_result = score_cv(jd_parsed, cv_text)
    print(f"             Overall Score: {score_result['overall_score']}%")
    print(f"             Grade: {score_result['grade']}")
    print(f"             Matched: {len(score_result['matched_keywords'])} keywords")
    print(f"             Missing: {len(score_result['missing_keywords'])} keywords")

    # ── Tool 3: Gap Analysis ──────────────────────────────────────────────────
    print("\n  [Tool 3/4] Gap Analyser     → Identifying strengths and gaps...")
    gap_result = analyse_gaps(jd_parsed, score_result)
    print(f"             Highlights: {len(gap_result['highlights'])} strength areas")
    print(f"             Quick wins: {len(gap_result['quick_wins'])} addressable gaps")

    # ── Tool 4: Interview Coach ────────────────────────────────────────────────
    print("\n  [Tool 4/4] Interview Coach  → Generating tailored Q&A (GenAI)...")
    interview_prep = generate_interview_prep(jd_parsed, gap_result)
    total_q = (len(interview_prep["technical"]) +
               len(interview_prep["behavioural"]) +
               len(interview_prep["accenture"]))
    print(f"             Generated {total_q} personalised interview questions")

    print("\n  ✅ All tools complete — generating report...\n")

    return {
        "jd_parsed":      jd_parsed,
        "score_result":   score_result,
        "gap_result":     gap_result,
        "interview_prep": interview_prep,
        "generated_at":   datetime.now().strftime("%d %B %Y, %H:%M"),
    }


def build_html_report(result: dict, output_path: str) -> str:
    """Generate a styled HTML report from the agent's output."""

    jd      = result["jd_parsed"]
    score   = result["score_result"]
    gaps    = result["gap_result"]
    prep    = result["interview_prep"]
    ts      = result["generated_at"]

    role    = jd["role_title"].replace('"', '&quot;')
    overall = score["overall_score"]
    grade   = score["grade"]

    # Score colour
    if overall >= 75:   score_color = "#3DDC84"
    elif overall >= 55: score_color = "#F4A261"
    else:               score_color = "#E76F9A"

    # ── Category scores bars ──────────────────────────────────────────────────
    cat_bars = ""
    for cat, data in score["category_scores"].items():
        s = data["score"]
        col = "#3DDC84" if s >= 70 else "#F4A261" if s >= 40 else "#E76F9A"
        cat_bars += f"""
        <div class="bar-row">
          <span class="bar-label">{cat}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{s}%;background:{col}"></div>
          </div>
          <span class="bar-val">{s}%</span>
        </div>"""

    # ── Matched keywords ──────────────────────────────────────────────────────
    matched_tags = "".join(f'<span class="tag match">{k}</span>' for k in score["matched_keywords"])
    missing_tags = "".join(f'<span class="tag miss">{k}</span>' for k in score["missing_keywords"])

    # ── Highlights ────────────────────────────────────────────────────────────
    highlights_html = ""
    for h in gaps["highlights"]:
        highlights_html += f"""
        <div class="highlight-card">
          <div class="hl-header">
            <span class="hl-cat">{h['category']}</span>
            <span class="hl-score" style="color:#3DDC84">{h['score']}% match</span>
          </div>
          <p>{h['advice']}</p>
        </div>"""

    # ── Quick wins ────────────────────────────────────────────────────────────
    qw_html = ""
    for qw in gaps["quick_wins"]:
        qw_html += f"""
        <div class="qw-item">
          <span class="qw-skill">{qw['skill']}</span>
          <span class="qw-action">{qw['action']}</span>
        </div>"""

    # ── Strategy ─────────────────────────────────────────────────────────────
    strategy_html = "".join(f"<li>{s}</li>" for s in gaps["strategy"])
    tips_html     = "".join(f"<li>{t}</li>" for t in gaps["application_tips"])

    # ── Interview questions ───────────────────────────────────────────────────
    def q_block(questions, label):
        html = f'<h3 class="q-section">{label}</h3>'
        for i, q in enumerate(questions, 1):
            tags_str = " ".join(f'<span class="qtag">{t}</span>' for t in q["tags"])
            html += f"""
            <div class="q-card">
              <div class="q-num">Q{i}</div>
              <div class="q-body">
                <div class="q-tags">{tags_str}</div>
                <p class="q-text"><strong>{q['q']}</strong></p>
                <div class="q-answer">
                  <div class="ans-label">Model Answer</div>
                  <p>{q['a']}</p>
                </div>
              </div>
            </div>"""
        return html

    interview_html = (
        q_block(prep["technical"],   "🔧 Technical Questions") +
        q_block(prep["behavioural"], "💬 Behavioural Questions") +
        q_block(prep["accenture"],   "🏢 Accenture-Specific Questions")
    )

    prep_tips_html = "".join(f"<li>{t}</li>" for t in prep["prep_tips"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Job Application Agent — {role}</title>
<style>
  :root {{
    --bg: #0D1B2A; --surface: #132338; --surface2: #1A3050;
    --teal: #2EC4B6; --navy: #1F3864; --orange: #F4A261;
    --green: #3DDC84; --purple: #9B72CF; --white: #F0F4F8;
    --gray: #8899AA; --border: #1E3A58;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--white); font-family: 'Segoe UI', Calibri, sans-serif;
          font-size: 15px; line-height: 1.6; }}
  a {{ color: var(--teal); }}

  /* Header */
  .header {{ background: var(--surface); padding: 32px 40px; border-bottom: 3px solid var(--teal); }}
  .header h1 {{ font-size: 26px; color: var(--teal); margin-bottom: 6px; }}
  .header .sub {{ color: var(--gray); font-size: 13px; }}
  .header .meta {{ margin-top: 10px; font-size: 13px; color: var(--gray); }}

  /* Score hero */
  .score-hero {{ display: flex; gap: 24px; padding: 32px 40px; flex-wrap: wrap; }}
  .score-circle {{ text-align: center; background: var(--surface); border-radius: 12px;
                   padding: 28px 36px; border: 2px solid var(--border); min-width: 160px; }}
  .score-num {{ font-size: 52px; font-weight: 700; color: {score_color}; }}
  .score-label {{ font-size: 12px; color: var(--gray); margin-top: 4px; }}
  .score-grade {{ font-size: 14px; color: var(--white); margin-top: 8px; font-weight: 600; }}

  /* Bars */
  .bars-panel {{ flex: 1; background: var(--surface); border-radius: 12px; padding: 24px;
                 border: 2px solid var(--border); min-width: 300px; }}
  .bars-panel h3 {{ color: var(--teal); margin-bottom: 16px; font-size: 14px; }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .bar-label {{ width: 160px; font-size: 12px; color: var(--gray); flex-shrink: 0; }}
  .bar-track {{ flex: 1; background: var(--surface2); border-radius: 4px; height: 10px; }}
  .bar-fill {{ height: 10px; border-radius: 4px; transition: width .3s; }}
  .bar-val {{ width: 36px; font-size: 12px; color: var(--white); text-align: right; }}

  /* Sections */
  .section {{ padding: 28px 40px; border-top: 1px solid var(--border); }}
  .section h2 {{ font-size: 18px; color: var(--teal); margin-bottom: 20px; padding-bottom: 8px;
                 border-bottom: 1px solid var(--border); }}

  /* Tags */
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px;
          margin: 3px; }}
  .match {{ background: #1A3A2A; color: var(--green); border: 1px solid #2A5A3A; }}
  .miss  {{ background: #3A1A1A; color: #FF8888; border: 1px solid #5A2A2A; }}

  /* Highlights */
  .highlight-card {{ background: var(--surface); border-radius: 10px; padding: 18px 20px;
                     margin-bottom: 14px; border-left: 4px solid var(--green); }}
  .hl-header {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
  .hl-cat {{ font-weight: 700; color: var(--white); }}
  .hl-score {{ font-size: 13px; }}

  /* Quick wins */
  .qw-item {{ background: var(--surface); border-radius: 8px; padding: 14px 18px;
              margin-bottom: 10px; border-left: 4px solid var(--orange); }}
  .qw-skill {{ font-weight: 700; color: var(--orange); display: block; margin-bottom: 4px; }}
  .qw-action {{ font-size: 13px; color: var(--gray); }}

  /* Strategy */
  .strategy-list li {{ margin-bottom: 10px; padding-left: 8px; color: var(--white); }}
  .tips-list li {{ margin-bottom: 8px; padding-left: 8px; color: var(--gray); }}

  /* Interview questions */
  .q-section {{ color: var(--teal); font-size: 16px; margin: 24px 0 14px; }}
  .q-card {{ background: var(--surface); border-radius: 10px; padding: 20px;
             margin-bottom: 16px; border: 1px solid var(--border); display: flex; gap: 16px; }}
  .q-num {{ font-size: 22px; font-weight: 700; color: var(--teal); min-width: 36px; }}
  .q-body {{ flex: 1; }}
  .q-tags {{ margin-bottom: 8px; }}
  .qtag {{ background: var(--surface2); color: var(--gray); font-size: 11px;
           padding: 2px 8px; border-radius: 12px; margin-right: 4px; }}
  .q-text {{ font-size: 15px; color: var(--white); margin-bottom: 12px; }}
  .q-answer {{ background: #0A1520; border-radius: 8px; padding: 14px 16px;
               border-left: 3px solid var(--purple); }}
  .ans-label {{ font-size: 11px; color: var(--purple); font-weight: 700;
                text-transform: uppercase; margin-bottom: 6px; }}
  .q-answer p {{ font-size: 13px; color: #B0C0D0; }}

  /* Footer */
  .footer {{ text-align: center; padding: 24px; color: var(--gray); font-size: 12px;
             border-top: 1px solid var(--border); }}

  @media (max-width: 600px) {{
    .score-hero {{ flex-direction: column; }}
    .header, .section {{ padding: 20px; }}
    .bar-label {{ width: 110px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🤖 AI Job Application Agent</h1>
  <div class="sub">Agentic pipeline: JD Parser → CV Scorer → Gap Analyser → Interview Coach</div>
  <div class="meta">Role: <strong style="color:var(--white)">{role}</strong> &nbsp;|&nbsp; Generated: {ts}</div>
</div>

<!-- Score Hero -->
<div class="score-hero">
  <div class="score-circle">
    <div class="score-num">{overall}%</div>
    <div class="score-label">Overall Match Score</div>
    <div class="score-grade">{grade}</div>
  </div>
  <div class="bars-panel">
    <h3>Match by Category</h3>
    {cat_bars}
  </div>
</div>

<!-- Keywords -->
<div class="section">
  <h2>🔑 Keyword Analysis</h2>
  <p style="margin-bottom:10px;color:var(--gray);font-size:13px;">✅ Matched in your CV ({len(score["matched_keywords"])} keywords)</p>
  <div style="margin-bottom:20px">{matched_tags}</div>
  <p style="margin-bottom:10px;color:var(--gray);font-size:13px;">⚠️ Missing from your CV ({len(score["missing_keywords"])} keywords)</p>
  <div>{missing_tags}</div>
</div>

<!-- Highlights -->
<div class="section">
  <h2>💪 Your Strengths — Lead with These</h2>
  {highlights_html}
</div>

<!-- Quick Wins -->
<div class="section">
  <h2>⚡ Quick Wins — Addressable Gaps</h2>
  {qw_html if qw_html else '<p style="color:var(--gray)">No major quick wins identified — your profile is strong for this role.</p>'}
</div>

<!-- Strategy -->
<div class="section">
  <h2>🎯 Application Strategy</h2>
  <ul class="strategy-list">{strategy_html}</ul>
  <h3 style="color:var(--teal);margin:20px 0 12px;font-size:15px">Application Tips</h3>
  <ul class="tips-list">{tips_html}</ul>
</div>

<!-- Interview Prep -->
<div class="section">
  <h2>🎤 Interview Preparation — {len(prep["technical"]) + len(prep["behavioural"]) + len(prep["accenture"])} Tailored Questions</h2>
  {interview_html}
  <h3 style="color:var(--teal);margin:20px 0 12px;font-size:15px">📋 Prep Tips</h3>
  <ul class="tips-list">{prep_tips_html}</ul>
</div>

<div class="footer">
  AI Job Application Agent · Built by Alan Sha · github.com/alansha1/ai-job-application-agent
  <br>Pipeline: JD Parser → CV Scorer (TF-IDF cosine similarity) → Gap Analyser → Interview Coach (GenAI)
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AI Job Application Agent")
    parser.add_argument("--jd", type=str, default=None,
                        help="Path to JD text file, or inline JD text")
    args = parser.parse_args()

    # Load JD
    if args.jd:
        if os.path.isfile(args.jd):
            with open(args.jd, "r") as f:
                jd_text = f.read()
        else:
            jd_text = args.jd
    else:
        jd_text = DEFAULT_JD
        print("  (Using default Accenture AI & Data Graduate JD)")

    # Run pipeline
    result = run_agent(jd_text)

    # Generate report
    os.makedirs("output", exist_ok=True)
    safe_role = re.sub(r'[^\w]', '_', result["jd_parsed"]["role_title"])[:40]
    out_path  = f"output/report_{safe_role}.html"

    build_html_report(result, out_path)

    print(f"  📄 Report saved → {out_path}")
    print(f"\n  Open in browser: {os.path.abspath(out_path)}\n")

    # Auto-open
    import platform, subprocess
    abs_path = os.path.abspath(out_path)
    if platform.system() == "Windows":
        os.startfile(abs_path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", abs_path])
    else:
        try: subprocess.run(["xdg-open", abs_path])
        except: pass


if __name__ == "__main__":
    main()
