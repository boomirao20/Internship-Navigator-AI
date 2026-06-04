"""
recommendation_engine.py — AI-Powered Internship Recommendation Engine
=======================================================================
Uses TF-IDF vectorization and cosine similarity from Scikit-learn
to match students with the most relevant internships.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_recommendation_engine(
    user_skills: list[str],
    internships: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """
    Recommend the top N internships using TF-IDF + Cosine Similarity.

    Step 1 — Combine user skills into a single text profile.
    Step 2 — Combine each internship's title + description into a document.
    Step 3 — Vectorize all documents with TfidfVectorizer.
    Step 4 — Compute cosine similarity between the user profile and every
             internship document.
    Step 5 — Rank by similarity and return the top N with match percentages.

    Parameters
    ----------
    user_skills : list[str]
        Skills the student has (e.g. ["Python", "SQL", "TensorFlow"]).
    internships : list[dict]
        List of internship records from the API.
    top_n : int
        Number of top recommendations to return.

    Returns
    -------
    list[dict]
        Internship records augmented with `match_score` and `rank`.
    """
    if not internships or not user_skills:
        return []

    # ── Step 1: Build the user profile text ───────────────────────────────
    user_profile = " ".join(user_skills).lower()

    # ── Step 2: Build internship documents ────────────────────────────────
    internship_docs = []
    for internship in internships:
        title = internship.get("title", "")
        description = internship.get("description", "")
        category = internship.get("category", "")
        doc = f"{title} {description} {category}".lower()
        internship_docs.append(doc)

    # ── Step 3: TF-IDF Vectorization ─────────────────────────────────────
    # Combine user profile + all internship docs into a single corpus.
    # The first element is the user profile; the rest are internships.
    corpus = [user_profile] + internship_docs

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),      # Capture bigrams like "machine learning"
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ── Step 4: Cosine Similarity ─────────────────────────────────────────
    user_vector = tfidf_matrix[0:1]            # shape (1, n_features)
    internship_vectors = tfidf_matrix[1:]       # shape (n_internships, n_features)
    similarities = cosine_similarity(user_vector, internship_vectors).flatten()

    # ── Step 5: Rank and return top N ─────────────────────────────────────
    # Sort indices by descending similarity
    ranked_indices = np.argsort(similarities)[::-1][:top_n]

    recommendations = []
    for rank, idx in enumerate(ranked_indices, start=1):
        internship = internships[idx].copy()
        raw_score = float(similarities[idx])
        # Convert to a friendly percentage (scale up for better UX)
        match_pct = min(round(raw_score * 100 * 1.5, 1), 99.0)
        match_pct = max(match_pct, 5.0)  # Floor at 5%
        internship["match_score"] = match_pct
        internship["raw_similarity"] = raw_score
        internship["rank"] = rank
        recommendations.append(internship)

    return recommendations


def get_average_match_score(recommendations: list[dict]) -> float:
    """
    Return the average match score across all recommendations.
    Useful for the readiness score calculation.
    """
    if not recommendations:
        return 0.0
    scores = [r.get("match_score", 0) for r in recommendations]
    return round(sum(scores) / len(scores), 1)


def get_top_match_score(recommendations: list[dict]) -> float:
    """Return the highest match score."""
    if not recommendations:
        return 0.0
    return max(r.get("match_score", 0) for r in recommendations)
