# 🎯 Internship Navigator AI

**AI-Powered Internship Recommendation System** built with Streamlit, Scikit-learn, Adzuna Jobs API, and Groq (Llama 3).

> Intelligently matches students with real-time internship opportunities using TF-IDF similarity scoring, machine learning domain prediction, and personalized AI career coaching.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-orange?logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-black?logo=groq)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Smart Matching** | TF-IDF + Cosine Similarity recommendation engine |
| 🧠 **Domain Prediction** | RandomForest classifier predicts ideal career paths |
| 🤖 **AI Career Coach** | Llama-3.3 powered by Groq generates personalized career roadmaps |
| 🔍 **Skill Gap Analysis** | 100+ skill database identifies what you need to learn |
| 📊 **Interactive Dashboard** | 5 Plotly visualizations with real-time data |
| 📄 **Resume Parsing** | Auto-extract skills from PDF, DOCX, or TXT resumes |
| 📥 **Report Generation** | Downloadable CSV and PDF reports |
| 🌐 **Live API Integration** | Adzuna Jobs API with smart caching & demo fallback |
| 🎨 **Premium UI** | Dark glassmorphism theme with smooth interactions |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure API Keys

For live internship data, get free API keys from [Adzuna Developer Portal](https://developer.adzuna.com/):

```bash
# Windows PowerShell
$env:ADZUNA_APP_ID = "your_app_id"
$env:ADZUNA_APP_KEY = "your_app_key"

# Linux/macOS
export ADZUNA_APP_ID="your_app_id"
export ADZUNA_APP_KEY="your_app_key"
```

> **Note:** The app works perfectly without Adzuna API keys — it uses 15 realistic demo internships from top tech companies (TCS, Infosys, Google, Microsoft, etc.) as fallback data.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```text
internship_navigator_ai/
│
├── app.py                    # Main Streamlit application (UI + integration)
├── api_service.py            # Adzuna API integration + caching + demo data
├── recommendation_engine.py  # TF-IDF + Cosine Similarity engine
├── skill_gap.py              # Skill extraction & gap analysis
├── domain_predictor.py       # RandomForest career domain predictor
├── ai_assistant.py           # Groq-powered AI Career Coach integration
├── resume_parser.py          # PDF/DOCX/TXT resume parsing
├── report_generator.py       # CSV & PDF report generation
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
2. Combines each internship's title + description into documents
3. Vectorizes all documents using `TfidfVectorizer` (bigrams, 5000 features)
4. Calculates `cosine_similarity` between user and each internship
5. Returns top 10 ranked by match percentage

### 2. Domain Predictor (RandomForest)

```text
Skills → TF-IDF Features → RandomForestClassifier → Domain + Confidence
```

- Trained on 54 synthetic skill-domain samples across 9 career domains
- 150 decision trees with balanced class weights
- Outputs prediction + probability distribution

### 3. AI Career Coach (Groq + Llama 3)
```text
(Skills + Predicted Domain + Missing Skills) → Groq API → Personalized Roadmap
```
- Passes the user's analyzed profile to Llama-3.3-70b via the blazing-fast Groq API.
- Generates actionable interview prep tips, a structured learning roadmap, and certification recommendations.

### 4. Readiness Score

```text
Readiness = (Avg Match Score × 70%) + (Domain Confidence × 30%)
```

---

## 📊 Dashboard Visualizations

1. **Match Score Ranking** — Horizontal bar chart with color gradient
2. **Internships by Domain** — Bar chart showing category distribution
3. **Location Distribution** — Pie chart of geographic spread
4. **Internship Types** — Donut chart (Remote/Hybrid/Onsite)
5. **Most In-Demand Skills** — Horizontal bar chart of skill frequency

---

## 🔑 API Configuration

| Variable | Description |
|----------|-------------|
| `ADZUNA_APP_ID` | Your Adzuna application ID |
| `ADZUNA_APP_KEY` | Your Adzuna API key |

*(Groq API key for the AI Career Coach is currently pre-configured for demonstration purposes).*

---

## 📋 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Web UI framework |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Plotly** | Interactive visualizations |
| **Scikit-learn** | TF-IDF, Cosine Similarity, RandomForest |
| **Groq API** | Llama 3 AI Career Coach |
| **Requests** | HTTP API calls |
| **PyPDF2** | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **fpdf2** | PDF report generation |

---

## 🎓 For Presentations

This project demonstrates:
- ✅ Real-time API data integration (Adzuna & Groq)
- ✅ Local Machine Learning (TF-IDF + RandomForest)
- ✅ Generative AI (Llama 3.3 for dynamic career coaching)
- ✅ Interactive data visualization
- ✅ File processing (resume parsing)
- ✅ Report generation (CSV + PDF)
- ✅ Professional UI/UX design with glassmorphism aesthetics

---

Built with ❤️ using Python, Streamlit, Scikit-learn & Groq
