# 🎯 Internship Navigator AI

**AI-Powered Internship Recommendation System** built with Streamlit, Scikit-learn, and Groq (Llama 3.3).

> Intelligently matches students with internship opportunities using TF-IDF similarity scoring, machine learning domain prediction, skill gap analysis, and personalized AI career coaching.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-orange?logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-black?logo=groq)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Smart Matching** | TF-IDF + Cosine Similarity recommendation engine |
| 🧠 **Domain Prediction** | RandomForest classifier predicts ideal career paths across 9 domains |
| 🤖 **AI Career Coach** | Llama-3.3-70b via Groq generates personalized career roadmaps |
| 🔍 **Skill Gap Analysis** | 100+ skill database identifies what you need to learn |
| 📊 **Interactive Dashboard** | 5 Plotly visualizations with real-time data |
| 📄 **Resume Parsing** | Auto-extract skills from PDF, DOCX, or TXT resumes |
| 📥 **Report Generation** | Downloadable CSV and PDF reports |
| 🌐 **Groq LLM Integration** | Generates realistic internship data with smart caching & demo fallback |
| 🎨 **Premium UI** | Dark glassmorphism theme with smooth animations |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API Key

Get a free API key from [Groq Console](https://console.groq.com/):

Then update the API key in two files:

```python
# ai_assistant.py  (line 7)
api_key="YOUR_GROQ_API_KEY"

# api_service.py   (line 19)
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

> **Note:** The app works without a valid Groq API key — it automatically falls back to **15 realistic demo internships** from top companies (TCS, Infosys, Google, Microsoft, AWS, etc.) so the app is always functional for demonstrations.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```text
Internship Navigator AI/
│
├── app.py                    # Main Streamlit application (UI + integration)
├── api_service.py            # Groq LLM API integration + caching + demo fallback
├── recommendation_engine.py  # TF-IDF + Cosine Similarity matching engine
├── skill_gap.py              # Skill extraction & gap analysis (100+ skills)
├── domain_predictor.py       # RandomForest career domain predictor (9 domains)
├── ai_assistant.py           # Groq-powered AI Career Coach (Llama 3.3)
├── resume_parser.py          # PDF/DOCX/TXT resume parsing
├── report_generator.py       # CSV & PDF report generation
├── test_fpdf.py              # FPDF2 library test script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🧠 How the AI Works

### 1. Recommendation Engine (TF-IDF + Cosine Similarity)

```text
Student Skills → TF-IDF Vector
Internship Descriptions → TF-IDF Vectors
Cosine Similarity Score → Ranked Recommendations
```

1. Combines user skills into a text profile
2. Combines each internship's title + description + category into documents
3. Vectorizes all documents using `TfidfVectorizer` (bigrams, 5000 features)
4. Calculates `cosine_similarity` between user and each internship
5. Returns top 10 ranked by match percentage (scaled ×1.5 for UX, capped at 99%)

### 2. Domain Predictor (RandomForest)

```text
Skills → TF-IDF Features → RandomForestClassifier → Domain + Confidence
```

- Trained on **54 synthetic skill-domain samples** across **9 career domains**:
  - Data Science, Machine Learning, Artificial Intelligence, Data Analytics, Web Development, Software Engineering, Cyber Security, Cloud Computing, DevOps
- 150 decision trees with balanced class weights, max depth 20
- TF-IDF vectorizer with 3000 features and bigrams
- Outputs prediction + probability distribution + top 3 domains

### 3. AI Career Coach (Groq + Llama 3.3)

```text
(Skills + Predicted Domain + Missing Skills) → Groq API → Personalized Roadmap
```

- Passes the user's analyzed profile to Llama-3.3-70b-versatile via the Groq API
- Generates career guidance, internship preparation tips, a structured learning roadmap, and certification recommendations
- Includes an intelligent fallback with offline advice if the API is unavailable

### 4. Internship Data (Groq LLM Generation)

```text
(Keywords + Location) → Groq LLM → Realistic Internship Listings
```

- Generates contextual internship data via Groq's Llama 3.3 model
- Results are cached for 10 minutes with MD5-based cache keys
- Falls back to 15 curated demo internships if the API is unreachable

### 5. Readiness Score

```text
Readiness = (Avg Match Score × 70%) + (Domain Confidence × 30%)
```

---

## 📊 Dashboard Visualizations

1. **Match Score Ranking** — Horizontal bar chart with color gradient (red → yellow → green)
2. **Internships by Domain** — Bar chart showing category distribution
3. **Location Distribution** — Pie chart of geographic spread
4. **Internship Types** — Donut chart (Remote/Hybrid/Onsite)
5. **Most In-Demand Skills** — Horizontal bar chart of skill frequency across internships

---

## 🔑 API Configuration

| Variable | Location | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | `api_service.py` | Groq API key for generating internship data |
| `api_key` | `ai_assistant.py` | Groq API key for the AI Career Coach |

Both use the same Groq API key. Get one for free at [console.groq.com](https://console.groq.com/).

> **Demo Mode:** When API keys are missing or invalid, the app automatically uses built-in demo data so it remains fully functional.

---

## 📋 Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Streamlit** | 1.45.1 | Web UI framework |
| **Pandas** | 2.2.3 | Data manipulation |
| **NumPy** | 2.2.6 | Numerical computing |
| **Plotly** | 6.1.2 | Interactive visualizations |
| **Scikit-learn** | 1.6.1 | TF-IDF, Cosine Similarity, RandomForest |
| **Groq** | 1.4.0 | Llama 3.3 API (internship generation + AI coach) |
| **Requests** | 2.32.3 | HTTP API calls |
| **PyPDF2** | 3.0.1 | PDF text extraction |
| **python-docx** | 1.1.2 | DOCX text extraction |
| **fpdf2** | 2.8.3 | PDF report generation |

---

## 🖥️ Application Tabs

| Tab | What It Shows |
|-----|---------------|
| 🎯 **Recommendations** | Top 10 internship matches with scores, cards, and apply links |
| 📊 **Dashboard** | 5 interactive Plotly charts analyzing your results |
| 🔍 **Skill Gap** | Skills you have vs. skills you need, per-internship breakdown |
| 🧠 **Domain Prediction** | RandomForest prediction with probability distribution chart |
| 🚀 **Readiness Score** | Gauge chart + weighted score breakdown table |
| 🤖 **AI Career Coach** | Personalized career guidance via Llama 3.3 |
| 📥 **Reports** | Download CSV and PDF reports of your full analysis |

---

## 🎓 For Presentations

This project demonstrates:

- ✅ **Generative AI** — Llama 3.3 for dynamic internship generation and career coaching
- ✅ **Machine Learning** — TF-IDF vectorization, Cosine Similarity, RandomForest classification
- ✅ **Skill Gap Analysis** — 100+ skill database with per-internship coverage breakdown
- ✅ **Interactive Visualization** — 5 Plotly charts (bar, pie, donut, gauge, horizontal bar)
- ✅ **File Processing** — Resume parsing for PDF, DOCX, and TXT formats
- ✅ **Report Generation** — Downloadable CSV and PDF reports
- ✅ **API Integration** — Groq LLM API with caching, error handling, and fallback
- ✅ **Premium UI/UX** — Dark glassmorphism theme with Inter font, animations, and hover effects

---

## 📸 Screenshots

After running the app, you'll see:

- **Hero banner** with animated gradient pulse effect
- **Sidebar** with student profile form and API status indicator
- **KPI cards** showing internships found, best match, predicted domain, and readiness score
- **Tabbed interface** with 7 feature tabs

---

Built with ❤️ using Python, Streamlit, Scikit-learn & Groq
