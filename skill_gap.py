"""
skill_gap.py — Skill Gap Analyzer
===================================
Extracts required skills from internship descriptions, compares
them against the student's skill set, and identifies gaps.
"""

import re

# ─── Comprehensive Skill Dictionary ──────────────────────────────────────────
# Organized by domain for accurate extraction from free-text descriptions.
SKILL_DATABASE = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab", "perl",
    "sql", "bash", "shell scripting", "solidity",

    # Data Science & ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "matplotlib", "seaborn", "plotly", "scipy", "statsmodels", "xgboost",
    "lightgbm", "opencv", "spacy", "nltk", "hugging face", "transformers",
    "deep learning", "machine learning", "neural networks", "nlp",
    "computer vision", "reinforcement learning", "data wrangling",
    "feature engineering", "model deployment", "mlops",

    # Web Development
    "html", "css", "react", "angular", "vue.js", "node.js", "express",
    "django", "flask", "fastapi", "next.js", "tailwind", "bootstrap",
    "jquery", "webpack", "graphql", "rest apis", "websockets",

    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "firebase", "sqlite", "oracle", "dynamodb",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "ci/cd", "linux", "nginx", "apache",
    "cloudformation", "monitoring", "prometheus", "grafana",

    # Data Engineering
    "apache spark", "airflow", "kafka", "hadoop", "etl", "data modeling",
    "data pipelines", "dbt", "snowflake", "bigquery",

    # Tools & Practices
    "git", "github", "gitlab", "jira", "confluence", "agile",
    "scrum", "unit testing", "testing", "figma", "adobe xd",
    "prototyping", "wireframing", "tableau", "power bi", "excel",
    "jupyter notebooks", "vscode",

    # Cybersecurity
    "network security", "penetration testing", "siem", "firewalls",
    "wireshark", "risk assessment", "compliance", "owasp",
    "vulnerability assessment", "incident response",

    # Soft Skills (relevant for internships)
    "communication", "teamwork", "problem solving", "leadership",
    "research", "academic writing", "presentation", "critical thinking",

    # Misc
    "blockchain", "web3.js", "smart contracts", "defi",
    "microservices", "system design", "data structures", "algorithms",
    "design systems", "responsive design", "accessibility",
    "user research", "image processing", "cuda", "statistics",
    "probability", "linear algebra", "reporting", "data cleaning",
}


def extract_skills_from_text(text: str) -> set[str]:
    """
    Extract known skills from free-form text using keyword matching.

    Uses case-insensitive matching against the SKILL_DATABASE.
    Handles multi-word skills (e.g. "machine learning", "rest apis").
    """
    text_lower = text.lower()
    found_skills = set()

    for skill in SKILL_DATABASE:
        # Escape regex-special characters in skills like "c++"
        pattern = r'\b' + re.escape(skill) + r'\b'
        # For very short skills, be more careful to avoid false positives
        if len(skill) <= 2:
            # Match only uppercase or isolated occurrences for short names
            if re.search(re.escape(skill), text, re.IGNORECASE):
                found_skills.add(skill.title() if len(skill) > 2 else skill.upper())
        else:
            if re.search(pattern, text_lower):
                # Capitalize nicely
                found_skills.add(skill.title())

    return found_skills


def analyze_skill_gap(
    student_skills: list[str],
    internships: list[dict],
    top_n: int = 5,
) -> dict:
    """
    Perform a comprehensive skill gap analysis.

    Parameters
    ----------
    student_skills : list[str]
        Skills the student currently has.
    internships : list[dict]
        List of internship records (should be the recommended ones).
    top_n : int
        Number of top internships to analyze.

    Returns
    -------
    dict with keys:
        - student_skills: set of normalized student skills
        - required_skills: set of all skills found in internship descriptions
        - matching_skills: intersection
        - missing_skills: skills the student is missing
        - match_percentage: how well the student matches
        - skill_frequency: dict mapping each required skill to its occurrence count
        - per_internship: list of per-internship gap analysis dicts
    """
    # Normalize student skills
    student_set = set()
    for s in student_skills:
        s_clean = s.strip().lower()
        if s_clean:
            student_set.add(s_clean)

    # Extract required skills from internship descriptions
    all_required: dict[str, int] = {}
    per_internship_analysis = []

    for internship in internships[:top_n]:
        desc = internship.get("description", "")
        title = internship.get("title", "")
        combined_text = f"{title} {desc}"
        extracted = extract_skills_from_text(combined_text)

        # Count frequency
        for skill in extracted:
            skill_lower = skill.lower()
            all_required[skill_lower] = all_required.get(skill_lower, 0) + 1

        # Per-internship analysis
        extracted_lower = {s.lower() for s in extracted}
        matching = student_set & extracted_lower
        missing = extracted_lower - student_set

        per_internship_analysis.append({
            "title": internship.get("title", ""),
            "company": internship.get("company", ""),
            "required": extracted,
            "matching": {s.title() for s in matching},
            "missing": {s.title() for s in missing},
            "coverage": round(len(matching) / max(len(extracted_lower), 1) * 100, 1),
        })

    # Overall analysis
    all_required_set = set(all_required.keys())
    matching_skills = student_set & all_required_set
    missing_skills = all_required_set - student_set

    match_percentage = round(
        len(matching_skills) / max(len(all_required_set), 1) * 100, 1
    )

    # Sort frequency map
    skill_frequency = dict(
        sorted(all_required.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "student_skills": {s.title() for s in student_set},
        "required_skills": {s.title() for s in all_required_set},
        "matching_skills": {s.title() for s in matching_skills},
        "missing_skills": {s.title() for s in missing_skills},
        "match_percentage": match_percentage,
        "skill_frequency": {k.title(): v for k, v in skill_frequency.items()},
        "per_internship": per_internship_analysis,
    }


def get_top_missing_skills(gap_analysis: dict, top_n: int = 10) -> list[str]:
    """
    Return the most commonly required skills that the student is missing,
    sorted by how frequently they appear across internships.
    """
    missing = gap_analysis.get("missing_skills", set())
    freq = gap_analysis.get("skill_frequency", {})

    missing_with_freq = [
        (skill, freq.get(skill, 0))
        for skill in missing
    ]
    missing_with_freq.sort(key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in missing_with_freq[:top_n]]
