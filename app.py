"""
app.py — Internship Navigator AI 🎯
=====================================
A production-ready Streamlit application that uses live API data
and Scikit-learn to intelligently match students with internships.

Features:
  • Real-time internship fetching (Adzuna API + demo fallback)
  • TF-IDF + Cosine Similarity recommendation engine
  • RandomForest domain prediction
  • Skill gap analysis
  • Resume parsing (PDF / DOCX / TXT)
  • Interactive Plotly dashboards
  • Downloadable CSV & PDF reports

Author: Internship Navigator AI Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from ai_assistant import get_career_advice

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab6"

# ─── Local Modules ────────────────────────────────────────────────────────────
from api_service import search_internships, get_api_status
from recommendation_engine import (
    build_recommendation_engine,
    get_average_match_score,
    get_top_match_score,
)
from skill_gap import analyze_skill_gap, get_top_missing_skills
from domain_predictor import get_predictor
from resume_parser import extract_skills_from_resume
from report_generator import generate_csv_report, generate_pdf_report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Internship Navigator AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS — Premium Dark Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
/* ─── Google Font ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ─── Global Text Color ────────────────────────────────────────────────── */
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li, label, .stText {
    color: white !important;
}

/* ─── Main Background ─────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* ─── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

/* ─── KPI Card ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.10));
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.25);
}
.kpi-icon { font-size: 32px; margin-bottom: 8px; }
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 4px 0;
}
.kpi-label {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ─── Internship Card ──────────────────────────────────────────────────── */
.intern-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.8), rgba(49,46,129,0.6));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.intern-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #3b82f6, #06b6d4);
}
.intern-card:hover {
    transform: translateY(-2px);
    border-color: rgba(139, 92, 246, 0.5);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.15);
}
.intern-title {
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.intern-company {
    font-size: 14px;
    color: #a78bfa;
    font-weight: 600;
    margin-bottom: 8px;
}
.intern-meta {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 12px;
}
.intern-desc {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 16px;
}

/* ─── Match Badge ──────────────────────────────────────────────────────── */
.match-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 700;
    color: white;
}
.match-high { background: linear-gradient(135deg, #059669, #10b981); }
.match-med  { background: linear-gradient(135deg, #d97706, #f59e0b); }
.match-low  { background: linear-gradient(135deg, #dc2626, #ef4444); }

/* ─── Skill Tags ───────────────────────────────────────────────────────── */
.skill-tag {
    display: inline-block;
    padding: 4px 12px;
    margin: 3px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 600;
}
.skill-have {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.skill-missing {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.skill-neutral {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* ─── Section Header ──────────────────────────────────────────────────── */
.section-header {
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    padding-top: 8px;
}
.section-sub {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* ─── Profile Card ─────────────────────────────────────────────────────── */
.profile-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.15));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
}
.profile-name {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
}
.profile-detail {
    font-size: 13px;
    color: #cbd5e1;
    padding: 4px 0;
}

/* ─── API Status Badge ─────────────────────────────────────────────────── */
.api-badge {
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    margin: 8px 0;
}
.api-live {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.api-demo {
    background: rgba(234, 179, 8, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

/* ─── Apply Button ─────────────────────────────────────────────────────── */
.apply-btn {
    display: inline-block;
    padding: 8px 24px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white !important;
    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.3s ease;
}
.apply-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
}

/* ─── Hero Banner ──────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.15), rgba(6,182,212,0.1));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 50%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd, #93c5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    position: relative;
    z-index: 1;
}
.hero-sub {
    font-size: 16px;
    color: #94a3b8;
    position: relative;
    z-index: 1;
}

/* ─── Readiness Gauge ──────────────────────────────────────────────────── */
.readiness-container {
    background: linear-gradient(135deg, rgba(30,27,75,0.9), rgba(49,46,129,0.7));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.readiness-score {
    font-size: 56px;
    font-weight: 800;
    margin: 16px 0 8px;
}
.readiness-label {
    font-size: 16px;
    color: #94a3b8;
    font-weight: 500;
}

/* ─── Tab Styling ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 27, 75, 0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
}

/* ─── Plotly chart background fix ──────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ─── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #1e1b4b; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #7c3aed, #3b82f6);
    border-radius: 4px;
}

/* ─── Expander Styling ─────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(30, 27, 75, 0.5) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE INITIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "internships" not in st.session_state:
    st.session_state.internships = []
if "skill_gap" not in st.session_state:
    st.session_state.skill_gap = None
if "domain_prediction" not in st.session_state:
    st.session_state.domain_prediction = None
if "readiness_score" not in st.session_state:
    st.session_state.readiness_score = 0.0
if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — Student Profile & Navigation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0;">
        <div style="font-size: 48px; margin-bottom: 8px;">🎯</div>
        <div style="font-size: 20px; font-weight: 800;
             background: linear-gradient(135deg, #c4b5fd, #93c5fd);
             -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">
            Internship Navigator AI
        </div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
            AI-Powered Career Matching
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── API Status (cached to avoid slow checks on every rerun) ─────────
    if "api_status" not in st.session_state:
        st.session_state.api_status = get_api_status()

    api_status = st.session_state.api_status
    status_class = "api-live" if api_status["status"] == "live" else "api-demo"
    st.markdown(
        f'<div class="api-badge {status_class}">'
        f'{api_status["icon"]} {api_status["message"]}</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh API Status", use_container_width=True):
        st.session_state.api_status = get_api_status()
        st.rerun()

    st.markdown("---")

    # ── Student Profile Form ──────────────────────────────────────────────
    st.markdown("### 👤 Student Profile")

    student_name = st.text_input(
        "Full Name",
        placeholder="e.g. Boomi Rao",
        key="student_name",
    )

    # Resume Upload
    st.markdown("#### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="resume_upload",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        extracted = extract_skills_from_resume(file_bytes, uploaded_file.name)
        if extracted:
            st.session_state.resume_skills = extracted
            st.success(f"✅ Extracted {len(extracted)} skills from resume!")
        else:
            st.warning("⚠️ Could not extract skills. Please enter manually.")

    # Skills input (pre-filled from resume if available)
    default_skills = ", ".join(st.session_state.resume_skills) if st.session_state.resume_skills else ""
    skills_input = st.text_area(
        "Your Skills (comma-separated)",
        value=default_skills,
        placeholder="e.g. Python, SQL, Machine Learning, Pandas",
        height=80,
        key="skills_input",
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "High School",
            "Undergraduate (1st Year)",
            "Undergraduate (2nd Year)",
            "Undergraduate (3rd Year)",
            "Undergraduate (Final Year)",
            "Postgraduate",
            "PhD",
        ],
        index=3,
        key="education",
    )

    preferred_domain = st.selectbox(
        "Preferred Domain",
        [
            "All Domains",
            "Data Science",
            "Machine Learning",
            "Artificial Intelligence",
            "Data Analytics",
            "Web Development",
            "Software Engineering",
            "Cyber Security",
            "Cloud Computing",
            "DevOps",
        ],
        key="domain",
    )

    preferred_location = st.text_input(
        "Preferred Location",
        placeholder="e.g. Bangalore",
        key="location",
    )

    internship_type = st.selectbox(
        "Internship Type",
        ["All", "Remote", "Hybrid", "Onsite"],
        key="intern_type",
    )

    st.markdown("---")

    # ── Launch Button ─────────────────────────────────────────────────────
    find_button = st.button(
        "🚀 Find My Internships",
        use_container_width=True,
        type="primary",
    )

    # ── Profile Summary Card ──────────────────────────────────────────────
    if student_name:
        skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
        domain_text = preferred_domain if preferred_domain != "All Domains" else "Exploring"
        skills_preview = ", ".join(skills_list[:5])
        if len(skills_list) > 5:
            skills_preview += f" +{len(skills_list) - 5} more"

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-name">👤 {student_name}</div>
            <div class="profile-detail">🎓 {education_level}</div>
            <div class="profile-detail">🎯 {domain_text}</div>
            <div class="profile-detail">📍 {preferred_location or 'Any Location'}</div>
            <div class="profile-detail">🛠️ {skills_preview or 'No skills entered'}</div>
            <div class="profile-detail">💼 {internship_type}</div>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CONTENT — Process and Display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Hero Banner ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎯 Internship Navigator AI</div>
    <div class="hero-sub">
        Powered by TF-IDF Similarity & Machine Learning — Find your perfect internship match
    </div>
</div>
""", unsafe_allow_html=True)

# ── Process when button is clicked ────────────────────────────────────────
skills_list = [s.strip() for s in skills_input.split(",") if s.strip()] if skills_input else []

if find_button:
    if not skills_list:
        st.error("⚠️ Please enter at least one skill to get recommendations.")
    else:
        with st.spinner("🔍 Fetching internships and running AI analysis..."):
            # 1. Fetch internships
            domain_query = preferred_domain if preferred_domain != "All Domains" else ""
            internships = search_internships(
                skills=skills_list,
                domain=domain_query,
                location=preferred_location,
                internship_type=internship_type,
            )
            st.session_state.internships = internships

            # 2. Generate recommendations
            recommendations = build_recommendation_engine(
                user_skills=skills_list,
                internships=internships,
                top_n=10,
            )
            st.session_state.recommendations = recommendations

            # 3. Skill gap analysis
            gap = analyze_skill_gap(
                student_skills=skills_list,
                internships=recommendations,
                top_n=5,
            )
            st.session_state.skill_gap = gap

            # 4. Domain prediction
            predictor = get_predictor()
            prediction = predictor.predict(skills_list)
            st.session_state.domain_prediction = prediction

            # 5. Readiness score
            avg_match = get_average_match_score(recommendations)
            domain_confidence = prediction.get("confidence", 0)
            readiness = round(avg_match * 0.7 + domain_confidence * 0.3, 1)
            readiness = min(readiness, 99.0)
            st.session_state.readiness_score = readiness

        st.success("✅ Analysis complete! Scroll down to explore your results.")


# ── Display results if available ──────────────────────────────────────────
recommendations = st.session_state.recommendations
internships = st.session_state.internships
skill_gap = st.session_state.skill_gap
domain_prediction = st.session_state.domain_prediction
readiness_score = st.session_state.readiness_score

if recommendations:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KPI ROW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{len(internships)}</div>
            <div class="kpi-label">Internships Found</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        top_score = get_top_match_score(recommendations)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{top_score}%</div>
            <div class="kpi-label">Best Match</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        pred_domain = domain_prediction.get("predicted_domain", "N/A") if domain_prediction else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value" style="font-size:18px;">{pred_domain}</div>
            <div class="kpi-label">Predicted Domain</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        score_color = "#10b981" if readiness_score >= 70 else "#f59e0b" if readiness_score >= 40 else "#ef4444"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🚀</div>
            <div class="kpi-value" style="background: linear-gradient(135deg, {score_color}, {score_color}dd);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {readiness_score}%
            </div>
            <div class="kpi-label">Readiness Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Recommendations",
        "📊 Dashboard",
        "🔍 Skill Gap",
        "🧠 Domain Prediction",
        "🚀 Readiness Score",
        "🤖 AI Career Coach",
        "📥 Reports",
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 1: RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            '<div class="section-header">🎯 Top Internship Recommendations</div>'
            '<div class="section-sub">Ranked by AI-powered TF-IDF similarity matching</div>',
            unsafe_allow_html=True,
        )

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_min_score = st.slider(
                "Minimum Match Score", 0, 100, 0, key="filter_score"
            )
        

        # Apply filters
        filtered_recs = recommendations
        if filter_min_score > 0:
            filtered_recs = [r for r in filtered_recs if r.get("match_score", 0) >= filter_min_score]
        

        if not filtered_recs:
            st.info("No internships match the current filters. Try adjusting them.")
        else:
            for rec in filtered_recs:
                score = rec.get("match_score", 0)
                badge_class = "match-high" if score >= 70 else "match-med" if score >= 40 else "match-low"

                st.markdown(f"""
                <div class="intern-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
                        <div style="flex:1; min-width:250px;">
                            <div class="intern-title">#{rec['rank']} {rec['title']}</div>
                            <div class="intern-company">🏢 {rec['company']}</div>
                            <div class="intern-meta">
                                📍 {rec['location']} &nbsp;•&nbsp; 📅 {rec.get('posted_date', 'Recently')}
                                &nbsp;•&nbsp; 💼 {rec.get('employment_type', 'Internship')}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div class="match-badge {badge_class}">{score}% Match</div>
                        </div>
                    </div>
                    <div class="intern-desc">{rec.get('description', '')[:250]}...</div>
                    <a href="{rec.get('apply_link', '#')}" target="_blank" class="apply-btn">
                        Apply Now →
                    </a>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2: DASHBOARD ANALYTICS
    # ──────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            '<div class="section-header">📊 Dashboard Analytics</div>'
            '<div class="section-sub">Visual insights from your internship search</div>',
            unsafe_allow_html=True,
        )

        # Prepare data
        df = pd.DataFrame(recommendations)

        chart_colors = ["#8b5cf6", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b",
                        "#ef4444", "#ec4899", "#6366f1", "#14b8a6", "#f97316"]

        d1, d2 = st.columns(2)

        # ── Chart 1: Match Scores (Interactive ranking) ───────────────────
        with d1:
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Bar(
                x=df["match_score"],
                y=[f"#{r['rank']} {r['title'][:30]}" for _, r in df.iterrows()],
                orientation="h",
                marker=dict(
                    color=df["match_score"],
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                    line=dict(width=0),
                ),
                text=[f"{s}%" for s in df["match_score"]],
                textposition="auto",
                textfont=dict(color="white", size=11, family="Inter"),
            ))
            fig_scores.update_layout(
                title=dict(text="Match Score Ranking", font=dict(color="#e2e8f0", size=16, family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(title="Match Score (%)", gridcolor="rgba(139,92,246,0.1)"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(139,92,246,0.1)"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # ── Chart 2: Internship Domains (Bar chart) ───────────────────────
        with d2:
            domain_counts = df["category"].value_counts().reset_index()
            domain_counts.columns = ["Domain", "Count"]
            fig_domains = px.bar(
                domain_counts, x="Domain", y="Count",
                color="Domain",
                color_discrete_sequence=chart_colors,
            )
            fig_domains.update_layout(
                title=dict(text="Internships by Domain", font=dict(color="#e2e8f0", size=16, family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                showlegend=False,
                xaxis=dict(gridcolor="rgba(139,92,246,0.1)"),
                yaxis=dict(gridcolor="rgba(139,92,246,0.1)"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_domains, use_container_width=True)

        d3, d4 = st.columns(2)

        # ── Chart 3: Location Distribution (Pie chart) ────────────────────
        with d3:
            loc_counts = df["location"].value_counts().reset_index()
            loc_counts.columns = ["Location", "Count"]
            fig_loc = px.pie(
                loc_counts, names="Location", values="Count",
                color_discrete_sequence=chart_colors,
                hole=0,
            )
            fig_loc.update_layout(
                title=dict(text="Location Distribution", font=dict(color="#e2e8f0", size=16, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig_loc.update_traces(textfont_color="white")
            st.plotly_chart(fig_loc, use_container_width=True)

        # ── Chart 4: Internship Types (Donut chart) ───────────────────────
        with d4:
            type_counts = df["employment_type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig_type = px.pie(
                type_counts, names="Type", values="Count",
                color_discrete_sequence=chart_colors,
                hole=0.55,
            )
            fig_type.update_layout(
                title=dict(text="Internship Types", font=dict(color="#e2e8f0", size=16, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig_type.update_traces(textfont_color="white")
            st.plotly_chart(fig_type, use_container_width=True)

        # ── Chart 5: Most In-Demand Skills ────────────────────────────────
        if skill_gap:
            freq = skill_gap.get("skill_frequency", {})
            if freq:
                top_skills = dict(list(freq.items())[:15])
                fig_skills = go.Figure()
                fig_skills.add_trace(go.Bar(
                    x=list(top_skills.values()),
                    y=list(top_skills.keys()),
                    orientation="h",
                    marker=dict(
                        color=list(range(len(top_skills))),
                        colorscale=[[0, "#8b5cf6"], [1, "#06b6d4"]],
                    ),
                    text=list(top_skills.values()),
                    textposition="auto",
                    textfont=dict(color="white", size=11, family="Inter"),
                ))
                fig_skills.update_layout(
                    title=dict(text="Most In-Demand Skills", font=dict(color="#e2e8f0", size=16, family="Inter")),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="Inter"),
                    xaxis=dict(title="Frequency", gridcolor="rgba(139,92,246,0.1)"),
                    yaxis=dict(autorange="reversed", gridcolor="rgba(139,92,246,0.1)"),
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_skills, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3: SKILL GAP ANALYSIS
    # ──────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown(
            '<div class="section-header">🔍 Skill Gap Analysis</div>'
            '<div class="section-sub">Identify what you need to learn for your dream internship</div>',
            unsafe_allow_html=True,
        )

        if skill_gap:
            # Overview metrics
            sg1, sg2, sg3 = st.columns(3)
            with sg1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">✅</div>
                    <div class="kpi-value">{len(skill_gap['matching_skills'])}</div>
                    <div class="kpi-label">Skills You Have</div>
                </div>""", unsafe_allow_html=True)
            with sg2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">❌</div>
                    <div class="kpi-value">{len(skill_gap['missing_skills'])}</div>
                    <div class="kpi-label">Skills to Learn</div>
                </div>""", unsafe_allow_html=True)
            with sg3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">📈</div>
                    <div class="kpi-value">{skill_gap['match_percentage']}%</div>
                    <div class="kpi-label">Skill Coverage</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Progress bar for skill coverage
            st.progress(min(skill_gap["match_percentage"] / 100, 1.0))

            st.markdown("<br>", unsafe_allow_html=True)

            # Skills you have
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("#### ✅ Skills You Have")
                matching = skill_gap.get("matching_skills", set())
                if matching:
                    tags = "".join(
                        f'<span class="skill-tag skill-have">{s}</span>' for s in sorted(matching)
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.info("No matching skills detected.")

            with sc2:
                st.markdown("#### ❌ Missing Skills")
                missing = get_top_missing_skills(skill_gap, top_n=15)
                if missing:
                    tags = "".join(
                        f'<span class="skill-tag skill-missing">{s}</span>' for s in missing
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.success("🎉 You have all the required skills!")

            # Per-internship breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Per-Internship Breakdown")
            for analysis in skill_gap.get("per_internship", []):
                with st.expander(
                    f"📌 {analysis['title']} at {analysis['company']} — {analysis['coverage']}% coverage"
                ):
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.markdown("**✅ Matching:**")
                        if analysis["matching"]:
                            tags = "".join(
                                f'<span class="skill-tag skill-have">{s}</span>'
                                for s in sorted(analysis["matching"])
                            )
                            st.markdown(tags, unsafe_allow_html=True)
                        else:
                            st.caption("None")
                    with ac2:
                        st.markdown("**❌ Missing:**")
                        if analysis["missing"]:
                            tags = "".join(
                                f'<span class="skill-tag skill-missing">{s}</span>'
                                for s in sorted(analysis["missing"])
                            )
                            st.markdown(tags, unsafe_allow_html=True)
                        else:
                            st.caption("None")

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4: DOMAIN PREDICTION
    # ──────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown(
            '<div class="section-header">🧠 Career Domain Prediction</div>'
            '<div class="section-sub">RandomForest classifier predicts your ideal career path</div>',
            unsafe_allow_html=True,
        )

        if domain_prediction:
            dp1, dp2 = st.columns([1, 1])

            with dp1:
                confidence = domain_prediction.get("confidence", 0)
                conf_color = "#10b981" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#ef4444"

                st.markdown(f"""
                <div class="readiness-container">
                    <div style="font-size:14px; color:#94a3b8; text-transform:uppercase;
                         letter-spacing:1px; font-weight:600;">Predicted Domain</div>
                    <div style="font-size:32px; font-weight:800;
                         background: linear-gradient(135deg, #c4b5fd, #93c5fd);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         margin: 12px 0;">
                        {domain_prediction['predicted_domain']}
                    </div>
                    <div style="font-size:48px; font-weight:800; color:{conf_color};">
                        {confidence}%
                    </div>
                    <div style="font-size:14px; color:#94a3b8;">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

            with dp2:
                # Probability chart
                probs = domain_prediction.get("all_probabilities", {})
                if probs:
                    prob_df = pd.DataFrame(
                        list(probs.items()), columns=["Domain", "Probability"]
                    ).sort_values("Probability", ascending=True)

                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Bar(
                        x=prob_df["Probability"],
                        y=prob_df["Domain"],
                        orientation="h",
                        marker=dict(
                            color=prob_df["Probability"],
                            colorscale=[[0, "#312e81"], [0.5, "#7c3aed"], [1, "#06b6d4"]],
                        ),
                        text=[f"{p}%" for p in prob_df["Probability"]],
                        textposition="auto",
                        textfont=dict(color="white", size=11, family="Inter"),
                    ))
                    fig_prob.update_layout(
                        title=dict(text="Domain Probabilities", font=dict(color="#e2e8f0", size=16, family="Inter")),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", family="Inter"),
                        xaxis=dict(title="Probability (%)", gridcolor="rgba(139,92,246,0.1)"),
                        yaxis=dict(gridcolor="rgba(139,92,246,0.1)"),
                        height=400,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

            # Top 3 domains
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🏆 Top 3 Predicted Domains")
            top3 = domain_prediction.get("top_3", [])
            tc1, tc2, tc3 = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for col, (domain, prob), medal in zip([tc1, tc2, tc3], top3, medals):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">{medal}</div>
                        <div class="kpi-value" style="font-size:16px;">{domain}</div>
                        <div class="kpi-label">{prob}% probability</div>
                    </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5: READINESS SCORE
    # ──────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown(
            '<div class="section-header">🚀 Internship Readiness Score</div>'
            '<div class="section-sub">Combined metric: 70% Recommendation Match + 30% Domain Prediction</div>',
            unsafe_allow_html=True,
        )

        rs1, rs2 = st.columns([1, 1])

        with rs1:
            # Gauge chart
            gauge_color = "#10b981" if readiness_score >= 70 else "#f59e0b" if readiness_score >= 40 else "#ef4444"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=readiness_score,
                number=dict(
                    suffix="%",
                    font=dict(size=48, color="#e2e8f0", family="Inter"),
                ),
                title=dict(
                    text="Internship Readiness",
                    font=dict(size=18, color="#94a3b8", family="Inter"),
                ),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#94a3b8"),
                    bar=dict(color=gauge_color),
                    bgcolor="rgba(30,27,75,0.5)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                        dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                        dict(range=[70, 100], color="rgba(16,185,129,0.15)"),
                    ],
                    threshold=dict(
                        line=dict(color="#e2e8f0", width=3),
                        thickness=0.8,
                        value=readiness_score,
                    ),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                height=350,
                margin=dict(l=30, r=30, t=60, b=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with rs2:
            # Score breakdown
            avg_match = get_average_match_score(recommendations)
            domain_conf = domain_prediction.get("confidence", 0) if domain_prediction else 0

            st.markdown(f"""
            <div class="readiness-container">
                <div class="readiness-label">Your Score</div>
                <div class="readiness-score" style="color: {gauge_color};">{readiness_score}%</div>
                <div class="readiness-label">
                    {'🌟 Excellent! You\'re highly prepared.' if readiness_score >= 70
                     else '👍 Good progress! Keep building skills.' if readiness_score >= 40
                     else '💪 Just getting started — keep learning!'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Breakdown details
            st.markdown("#### 📐 Score Breakdown")
            st.markdown(f"""
            | Component | Score | Weight | Contribution |
            |-----------|-------|--------|-------------|
            | Recommendation Match | {avg_match}% | 70% | {round(avg_match * 0.7, 1)}% |
            | Domain Prediction | {domain_conf}% | 30% | {round(domain_conf * 0.3, 1)}% |
            | **Total Readiness** | | | **{readiness_score}%** |
            """)

            # Progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(readiness_score / 100, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # TAB 6: AI CAREER COACH
    # ──────────────────────────────────────────────────────────────────────
    
    with tab6:
        st.markdown("## 🤖 AI Career Coach")

        @st.fragment
        def ai_coach_fragment():
            if "ai_advice" not in st.session_state:
                st.session_state.ai_advice = None

            if st.button("Generate AI Advice", type="primary"):
                missing_skills = []

                if skill_gap:
                    missing_skills = list(
                        skill_gap.get("missing_skills", [])
                    )[:10]

                predicted_domain = (
                    domain_prediction.get("predicted_domain", "Unknown")
                    if domain_prediction else "Unknown"
                )

                with st.spinner("AI is analyzing your profile..."):
                    advice = get_career_advice(
                        skills_list,
                        predicted_domain,
                        missing_skills
                    )
                
                # Save advice in session state so it doesn't disappear
                st.session_state.ai_advice = advice

            # Display the advice if it exists
            if st.session_state.ai_advice:
                st.markdown(st.session_state.ai_advice)

        # Call the fragment
        ai_coach_fragment()

    # ──────────────────────────────────────────────────────────────────────
    # TAB 7: REPORTS
    # ──────────────────────────────────────────────────────────────────────
    with tab7:
        st.markdown(
            '<div class="section-header">📥 Download Reports</div>'
            '<div class="section-sub">Export your complete analysis as CSV or PDF</div>',
            unsafe_allow_html=True,
        )

        student_profile = {
            "name": student_name or "Student",
            "skills": skills_list,
            "education": education_level,
            "domain": preferred_domain,
            "location": preferred_location or "Any",
            "internship_type": internship_type,
        }

        rc1, rc2 = st.columns(2)

        with rc1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value" style="font-size:20px;">CSV Report</div>
                <div class="kpi-label">Spreadsheet format with all data</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            csv_data = generate_csv_report(
                student_profile, recommendations,
                skill_gap or {}, readiness_score,
                )
            
            st.download_button(
                label="📊 Download CSV Report",
                data=bytes(csv_data),   
                file_name=f"internship_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                )

        with rc2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-icon">📄</div>
                <div class="kpi-value" style="font-size:20px;">PDF Report</div>
                <div class="kpi-label">Professional formatted report</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            pdf_data = generate_pdf_report(
                student_profile, recommendations,
                skill_gap or {}, readiness_score,
                domain_prediction,
                )
            
            st.download_button(
                label="📄 Download PDF Report",
                data=bytes(pdf_data),   
                file_name=f"internship_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                )
        
        # Report preview
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("👁️ Preview Report Contents"):
            st.markdown("#### Student Profile")
            st.json(student_profile)

            st.markdown("#### Top 5 Recommendations")
            for rec in recommendations[:5]:
                st.markdown(
                    f"**#{rec['rank']}** {rec['title']} at {rec['company']} — "
                    f"**{rec['match_score']}% match**"
                )

            if skill_gap:
                st.markdown("#### Skill Gap Summary")
                st.markdown(f"- **Matching Skills:** {len(skill_gap.get('matching_skills', set()))}")
                st.markdown(f"- **Missing Skills:** {len(skill_gap.get('missing_skills', set()))}")
                st.markdown(f"- **Coverage:** {skill_gap.get('match_percentage', 0)}%")

            st.markdown(f"#### Readiness Score: **{readiness_score}%**")

else:
    # ── Welcome State — No results yet ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    steps = [
        ("1️⃣", "Enter Your Profile", "Add your skills, education level, and preferences in the sidebar."),
        ("2️⃣", "Upload Resume (Optional)", "Upload a PDF/DOCX/TXT resume to auto-extract your skills."),
        ("3️⃣", "Get AI Recommendations", "Click 'Find My Internships' to get personalized matches."),
    ]
    for col, (icon, title, desc) in zip([wc1, wc2, wc3], steps):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="min-height:180px;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value" style="font-size:16px;">{title}</div>
                <div class="kpi-label" style="text-transform:none; letter-spacing:0;
                     font-size:12px; margin-top:8px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature highlights
    st.markdown(
        '<div class="section-header" style="text-align:center;">✨ What This App Does</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🎯", "Smart Matching", "TF-IDF + Cosine Similarity"),
        ("🧠", "Domain Prediction", "RandomForest Class ifier"),
        ("🔍", "Skill Gap Analysis", "100+ skill database"),
        ("📊", "Visual Analytics", "Interactive Plotly charts"),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value" style="font-size:14px;">{title}</div>
                <div class="kpi-label" style="text-transform:none; letter-spacing:0;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:24px; border-top:1px solid rgba(139,92,246,0.2);">
    <div style="font-size:14px; color:#64748b;">
        Built with ❤️ using Streamlit, Scikit-learn & Plotly
    </div>
    <div style="font-size:12px; color:#475569; margin-top:4px;">
        Internship Navigator AI • Powered by TF-IDF Similarity & RandomForest ML
    </div>
</div>
""", unsafe_allow_html=True)
