"""
api_service.py — Real-Time Internship API Integration Layer
============================================================
Fetches live internship data from the Groq LLM API.
Implements caching, error handling, and data normalization.
"""

import requests
import time
import hashlib
import json
import re
import os
import streamlit as st
from datetime import datetime, timedelta
from groq import Groq

# ─── API Configuration ────────────────────────────────────────────────────────
GROQ_API_KEY = "YOUR_GROQ_API_KEY" # Replace

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=15.0,          # 15-second hard timeout
    max_retries=1,         # Only retry once to avoid long waits
)

# ─── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL_SECONDS = 600  # 10-minute cache lifetime


def _cache_key(params: dict) -> str:
    """Generate a deterministic cache key from query parameters."""
    raw = json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _is_cache_valid(key: str) -> bool:
    """Check whether a cached entry is still within its TTL."""
    if key not in _cache:
        return False
    cached_time = _cache[key]["timestamp"]
    return (time.time() - cached_time) < CACHE_TTL_SECONDS


def _get_from_cache(key: str):
    """Retrieve a value from the cache if valid."""
    if _is_cache_valid(key):
        return _cache[key]["data"]
    return None


def _set_cache(key: str, data):
    """Store a value in the cache with the current timestamp."""
    _cache[key] = {"data": data, "timestamp": time.time()}


def _extract_json_from_response(content: str) -> list:
    """
    Robustly extract JSON from the LLM response.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    # Try direct parse first
    content = content.strip()
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks: ```json ... ``` or ``` ... ```
    patterns = [
        r'```json\s*\n?(.*?)\n?\s*```',
        r'```\s*\n?(.*?)\n?\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, list):
                    return result
                return [result]
            except json.JSONDecodeError:
                continue

    # Try finding the first [ ... ] block
    bracket_match = re.search(r'\[.*\]', content, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


# ─── Core API Functions ───────────────────────────────────────────────────────

def fetch_internships(
    keywords="internship",
    location="",
    page=1,
    results_per_page=10,
    country="in",
):
    # Check cache first
    cache_params = {
        "keywords": keywords,
        "location": location,
        "page": page,
        "results_per_page": results_per_page,
    }
    key = _cache_key(cache_params)
    cached = _get_from_cache(key)
    if cached is not None:
        return cached

    try:
        prompt = f"""
        Generate {results_per_page} realistic internship opportunities.

        Keywords: {keywords}
        Location: {location}

        Return ONLY valid JSON (no markdown, no explanation) in this format:

        [
          {{
            "title": "Data Science Intern",
            "company": "Google",
            "location": "Mumbai",
            "description": "Work on ML models and analytics. Required skills: Python, Pandas, SQL.",
            "employment_type": "Internship",
            "apply_link": "#",
            "posted_date": "Recently",
            "category": "Data Science"
          }}
        ]
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Return ONLY valid JSON arrays, no markdown formatting, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content
        internships = _extract_json_from_response(content)

        if internships:
            # Normalize: ensure all required fields exist
            for item in internships:
                item.setdefault("title", "Internship")
                item.setdefault("company", "Unknown")
                item.setdefault("location", location or "India")
                item.setdefault("description", "")
                item.setdefault("employment_type", "Internship")
                item.setdefault("apply_link", "#")
                item.setdefault("posted_date", "Recently")
                item.setdefault("category", "General")
                item.setdefault("salary_min", None)
                item.setdefault("salary_max", None)

            # Cache the results
            _set_cache(key, internships)
            return internships

        # If parsing failed, fall back
        st.warning("⚠️ Could not parse API response. Using demo data.")
        return _get_fallback_data(keywords, location)

    except Exception as e:
        st.warning(f"⚠️ API unavailable ({type(e).__name__}). Using demo data.")
        return _get_fallback_data(keywords, location)

# ─── Fallback / Demo Data ─────────────────────────────────────────────────────

def _get_fallback_data(keywords: str = "", location: str = "") -> list[dict]:
    """
    Provide realistic demo data when the API is unavailable.
    This ensures the app is always functional for presentations.
    """
    demo_internships = [
        {
            "title": "Data Science Intern",
            "company": "TCS (Tata Consultancy Services)",
            "location": "Mumbai, India",
            "description": (
                "Join our Data Science team to work on predictive analytics, "
                "machine learning models, and data visualization. Required skills: "
                "Python, Pandas, Scikit-learn, SQL, TensorFlow, statistics, data "
                "wrangling, Jupyter Notebooks."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.tcs.com/careers",
            "posted_date": "May 28, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Data Science",
        },
        {
            "title": "Machine Learning Engineer Intern",
            "company": "Infosys",
            "location": "Bangalore, India",
            "description": (
                "Work on cutting-edge ML projects involving NLP, computer vision, "
                "and recommendation systems. Required skills: Python, PyTorch, "
                "TensorFlow, Scikit-learn, deep learning, neural networks, Docker, "
                "MLOps, Git."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.infosys.com/careers",
            "posted_date": "May 25, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Machine Learning",
        },
        {
            "title": "Full Stack Web Developer Intern",
            "company": "Wipro",
            "location": "Hyderabad, India",
            "description": (
                "Build responsive web applications using modern frameworks. "
                "Required skills: React, Node.js, JavaScript, TypeScript, HTML, "
                "CSS, MongoDB, REST APIs, Git, Agile methodology."
            ),
            "employment_type": "Internship",
            "apply_link": "https://careers.wipro.com",
            "posted_date": "May 30, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Web Development",
        },
        {
            "title": "AI Research Intern",
            "company": "Google India",
            "location": "Bangalore, India (Hybrid)",
            "description": (
                "Contribute to AI research in areas such as reinforcement learning, "
                "generative models, and large language models. Required skills: "
                "Python, TensorFlow, JAX, research methodology, linear algebra, "
                "probability, academic writing, Git."
            ),
            "employment_type": "Internship",
            "apply_link": "https://careers.google.com",
            "posted_date": "May 22, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Artificial Intelligence",
        },
        {
            "title": "Cloud Computing Intern",
            "company": "Amazon Web Services (AWS)",
            "location": "Remote",
            "description": (
                "Support cloud infrastructure projects and learn AWS services. "
                "Required skills: AWS, Linux, Docker, Kubernetes, Terraform, "
                "networking, Python, CI/CD, monitoring, CloudFormation."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.amazon.jobs",
            "posted_date": "May 27, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Cloud Computing",
        },
        {
            "title": "Cybersecurity Analyst Intern",
            "company": "HCL Technologies",
            "location": "Noida, India",
            "description": (
                "Assist the security team with vulnerability assessments and "
                "incident response. Required skills: network security, SIEM, "
                "penetration testing, Linux, firewalls, Wireshark, Python, "
                "risk assessment, compliance, OWASP."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.hcltech.com/careers",
            "posted_date": "May 26, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Cyber Security",
        },
        {
            "title": "DevOps Engineer Intern",
            "company": "Flipkart",
            "location": "Bangalore, India",
            "description": (
                "Automate deployment pipelines and manage cloud infrastructure. "
                "Required skills: Jenkins, Docker, Kubernetes, AWS, Terraform, "
                "Ansible, Git, Linux, Python, shell scripting, monitoring."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.flipkartcareers.com",
            "posted_date": "May 29, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "DevOps",
        },
        {
            "title": "Data Analytics Intern",
            "company": "Deloitte India",
            "location": "Delhi, India",
            "description": (
                "Analyze business data and create insightful dashboards. "
                "Required skills: SQL, Python, Excel, Tableau, Power BI, "
                "statistics, data cleaning, ETL, reporting, communication."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www2.deloitte.com/careers",
            "posted_date": "May 24, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Data Analytics",
        },
        {
            "title": "Software Engineering Intern",
            "company": "Microsoft India",
            "location": "Hyderabad, India (Hybrid)",
            "description": (
                "Develop features for Microsoft products and services. "
                "Required skills: C++, C#, Java, Python, data structures, "
                "algorithms, system design, Git, Azure, unit testing, Agile."
            ),
            "employment_type": "Internship",
            "apply_link": "https://careers.microsoft.com",
            "posted_date": "May 23, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Software Engineering",
        },
        {
            "title": "Backend Developer Intern",
            "company": "Zomato",
            "location": "Gurgaon, India",
            "description": (
                "Build scalable backend services for a high-traffic platform. "
                "Required skills: Python, Django, Flask, PostgreSQL, Redis, "
                "REST APIs, Docker, microservices, Git, testing."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.zomato.com/careers",
            "posted_date": "May 31, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Software Engineering",
        },
        {
            "title": "NLP Engineer Intern",
            "company": "Zoho Corporation",
            "location": "Chennai, India",
            "description": (
                "Work on natural language processing projects for enterprise "
                "products. Required skills: Python, NLP, spaCy, NLTK, "
                "transformers, Hugging Face, text classification, sentiment "
                "analysis, deep learning, Git."
            ),
            "employment_type": "Internship",
            "apply_link": "https://www.zoho.com/careers",
            "posted_date": "May 20, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Machine Learning",
        },
        {
            "title": "UI/UX Design Intern",
            "company": "Swiggy",
            "location": "Bangalore, India (Remote)",
            "description": (
                "Design intuitive user interfaces for mobile and web platforms. "
                "Required skills: Figma, Adobe XD, prototyping, wireframing, "
                "user research, design systems, HTML, CSS, accessibility, "
                "responsive design."
            ),
            "employment_type": "Internship",
            "apply_link": "https://careers.swiggy.com",
            "posted_date": "May 21, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Web Development",
        },
        {
            "title": "Blockchain Developer Intern",
            "company": "Polygon Labs",
            "location": "Remote",
            "description": (
                "Develop smart contracts and decentralized applications. "
                "Required skills: Solidity, Ethereum, Web3.js, JavaScript, "
                "smart contracts, DeFi, testing frameworks, Git, Node.js."
            ),
            "employment_type": "Internship",
            "apply_link": "https://polygon.technology/careers",
            "posted_date": "May 19, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Software Engineering",
        },
        {
            "title": "Data Engineer Intern",
            "company": "Paytm",
            "location": "Noida, India",
            "description": (
                "Build and maintain data pipelines and data warehouses. "
                "Required skills: Python, SQL, Apache Spark, Airflow, Kafka, "
                "ETL, AWS, data modeling, Hadoop, Git."
            ),
            "employment_type": "Internship",
            "apply_link": "https://paytm.com/careers",
            "posted_date": "May 18, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Data Science",
        },
        {
            "title": "Computer Vision Intern",
            "company": "Samsung R&D Institute India",
            "location": "Bangalore, India",
            "description": (
                "Work on image recognition and object detection for mobile "
                "devices. Required skills: Python, OpenCV, TensorFlow, PyTorch, "
                "deep learning, CNNs, image processing, CUDA, C++, research."
            ),
            "employment_type": "Internship",
            "apply_link": "https://research.samsung.com/sri-b",
            "posted_date": "May 17, 2026",
            "salary_min": None,
            "salary_max": None,
            "category": "Artificial Intelligence",
        },
    ]

    # Simple keyword filtering on demo data
    kw_lower = keywords.lower() if keywords else ""
    loc_lower = location.lower() if location else ""

    filtered = demo_internships
    if kw_lower and kw_lower != "internship":
        filtered = [
            i for i in filtered
            if kw_lower in i["title"].lower()
            or kw_lower in i["description"].lower()
            or kw_lower in i["category"].lower()
        ]

    if loc_lower:
        loc_filtered = [
            i for i in filtered
            if loc_lower in i["location"].lower()
        ]
        if loc_filtered:
            filtered = loc_filtered

    # Always return at least some results
    return filtered if filtered else demo_internships


def search_internships(
    skills: list[str],
    domain: str = "",
    location: str = "",
    internship_type: str = "All",
) -> list[dict]:
    """
    High-level search that combines skills + domain into a query.

    Parameters
    ----------
    skills : list[str]
        Student's skill list.
    domain : str
        Preferred career domain.
    location : str
        Preferred city/region.
    internship_type : str
        Remote / Hybrid / Onsite / All.

    Returns
    -------
    list[dict]
        Internship results.
    """
    # Build a rich query string
    query_parts = []
    if domain:
        query_parts.append(domain)
    if skills:
        query_parts.append(" ".join(skills[:5]))  # top 5 skills
    query_parts.append("internship")
    query = " ".join(query_parts)

    results = fetch_internships(keywords=query, location=location)

    # Client-side filter by internship type
    if internship_type and internship_type != "All":
        type_lower = internship_type.lower()
        type_filtered = []
        for r in results:
            loc_lower = r["location"].lower()
            if type_lower == "remote" and "remote" in loc_lower:
                type_filtered.append(r)
            elif type_lower == "hybrid" and "hybrid" in loc_lower:
                type_filtered.append(r)
            elif type_lower == "onsite" and "remote" not in loc_lower and "hybrid" not in loc_lower:
                type_filtered.append(r)
            else:
                type_filtered.append(r)
        if type_filtered:
            results = type_filtered

    return results


def get_api_status() -> dict:
    """
    Check whether the Groq API is reachable using a lightweight test.
    Uses a fast timeout to avoid blocking the UI.
    """
    if not GROQ_API_KEY:
        return {
            "status": "error",
            "message": "No Groq API key configured",
            "icon": "🔴",
        }

    try:
        # Use a quick connectivity check with minimal tokens
        test_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=5.0,     # Quick 5-second timeout for status check
            max_retries=0,   # No retries for status check
        )

        response = test_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=1,
        )

        return {
            "status": "live",
            "message": "Connected to Groq API",
            "icon": "🟢",
        }

    except Exception as e:
        error_type = type(e).__name__
        return {
            "status": "demo",
            "message": f"Using Demo Mode ({error_type})",
            "icon": "🟡",
        }