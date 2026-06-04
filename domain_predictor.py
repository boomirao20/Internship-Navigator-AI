"""
domain_predictor.py — Career Domain Prediction Engine
======================================================
Uses a RandomForestClassifier trained on a synthetic skill-domain
mapping dataset to predict the best career domain for a student.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

# ─── Domain Definitions ──────────────────────────────────────────────────────
DOMAINS = [
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Data Analytics",
    "Web Development",
    "Software Engineering",
    "Cyber Security",
    "Cloud Computing",
    "DevOps",
]

# ─── Synthetic Training Data ─────────────────────────────────────────────────
# Each entry maps a skill combination to a domain label.
# This acts as the training corpus for the classifier.
TRAINING_DATA = [
    # Data Science
    ("python pandas numpy scikit-learn statistics data wrangling jupyter sql visualization matplotlib seaborn", "Data Science"),
    ("python r statistics regression classification clustering data analysis exploratory data cleaning", "Data Science"),
    ("python sql data visualization pandas matplotlib scipy statistics hypothesis testing", "Data Science"),
    ("python numpy pandas scikit-learn feature engineering model evaluation cross validation data science", "Data Science"),
    ("python data wrangling sql pandas numpy statistics probability jupyter notebooks reporting", "Data Science"),
    ("r python statistics bayesian inference data mining sql excel visualization research", "Data Science"),

    # Machine Learning
    ("python tensorflow pytorch deep learning neural networks cnn rnn nlp computer vision", "Machine Learning"),
    ("python scikit-learn xgboost lightgbm model training hyperparameter tuning feature engineering", "Machine Learning"),
    ("python keras tensorflow deep learning image classification object detection transfer learning", "Machine Learning"),
    ("python pytorch transformers hugging face nlp text classification sentiment analysis bert gpt", "Machine Learning"),
    ("python machine learning reinforcement learning neural networks optimization gradient descent", "Machine Learning"),
    ("python mlops model deployment docker tensorflow serving flask api machine learning pipeline", "Machine Learning"),

    # Artificial Intelligence
    ("python tensorflow reinforcement learning generative models research linear algebra probability", "Artificial Intelligence"),
    ("python ai computer vision nlp deep learning research academic writing neural networks", "Artificial Intelligence"),
    ("python jax tensorflow research reinforcement learning generative adversarial networks optimization", "Artificial Intelligence"),
    ("python ai robotics computer vision deep learning planning reasoning knowledge representation", "Artificial Intelligence"),
    ("python research ai ethics natural language understanding reasoning transformers large language models", "Artificial Intelligence"),
    ("python ai opencv image processing deep learning cnns research cuda gpu computing", "Artificial Intelligence"),

    # Data Analytics
    ("sql excel tableau power bi data visualization reporting business intelligence analytics etl", "Data Analytics"),
    ("sql python excel data cleaning reporting tableau visualization dashboard business analytics", "Data Analytics"),
    ("excel power bi sql data analysis reporting statistics business intelligence visualization", "Data Analytics"),
    ("python sql tableau data analysis visualization reporting communication presentation business", "Data Analytics"),
    ("sql excel google analytics data visualization reporting kpi dashboard business intelligence", "Data Analytics"),
    ("python sql data analysis pandas visualization matplotlib reporting communication storytelling", "Data Analytics"),

    # Web Development
    ("html css javascript react node.js express mongodb rest apis responsive design", "Web Development"),
    ("html css javascript typescript angular vue.js webpack frontend responsive design", "Web Development"),
    ("javascript react next.js tailwind css node.js graphql rest apis web development", "Web Development"),
    ("python django flask html css javascript rest apis postgresql web development", "Web Development"),
    ("html css javascript php mysql wordpress responsive design frontend backend", "Web Development"),
    ("react typescript next.js node.js mongodb express graphql web development fullstack", "Web Development"),

    # Software Engineering
    ("java python c++ data structures algorithms system design git unit testing agile", "Software Engineering"),
    ("python java c# object oriented programming design patterns testing git ci cd", "Software Engineering"),
    ("c++ java python algorithms data structures system design databases git software engineering", "Software Engineering"),
    ("python java microservices rest apis docker testing git agile software development", "Software Engineering"),
    ("java spring boot python django rest apis postgresql docker git software engineering", "Software Engineering"),
    ("python go rust systems programming concurrency testing git linux software engineering", "Software Engineering"),

    # Cyber Security
    ("network security penetration testing linux firewalls wireshark siem owasp python", "Cyber Security"),
    ("cybersecurity vulnerability assessment incident response compliance risk assessment python", "Cyber Security"),
    ("penetration testing ethical hacking linux network security kali wireshark metasploit", "Cyber Security"),
    ("siem security monitoring incident response network security firewalls compliance python", "Cyber Security"),
    ("cybersecurity owasp web security penetration testing vulnerability scanning linux python", "Cyber Security"),
    ("network security cryptography python linux firewalls intrusion detection compliance", "Cyber Security"),

    # Cloud Computing
    ("aws azure gcp cloud infrastructure docker kubernetes terraform networking linux", "Cloud Computing"),
    ("aws ec2 s3 lambda cloudformation iam networking linux cloud architecture", "Cloud Computing"),
    ("azure cloud services virtual machines networking docker kubernetes terraform", "Cloud Computing"),
    ("gcp google cloud bigquery dataflow cloud functions kubernetes docker terraform", "Cloud Computing"),
    ("aws cloud computing networking linux security iam vpc load balancing auto scaling", "Cloud Computing"),
    ("cloud computing serverless aws lambda azure functions gcp cloud run docker", "Cloud Computing"),

    # DevOps
    ("docker kubernetes jenkins ci cd terraform ansible git linux automation monitoring", "DevOps"),
    ("devops jenkins pipeline docker kubernetes helm terraform aws automation python", "DevOps"),
    ("ci cd github actions docker kubernetes monitoring prometheus grafana linux devops", "DevOps"),
    ("ansible terraform docker kubernetes linux bash automation configuration management", "DevOps"),
    ("devops docker kubernetes aws terraform jenkins pipeline automation monitoring alerting", "DevOps"),
    ("gitlab ci docker kubernetes helm charts terraform ansible linux shell scripting devops", "DevOps"),
]


class DomainPredictor:
    """
    Predicts the best career domain based on a student's skill set
    using a RandomForestClassifier with TF-IDF features.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000,
            ngram_range=(1, 2),
        )
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            class_weight="balanced",
        )
        self._is_trained = False
        self._train()

    def _train(self):
        """Train the model on the synthetic dataset."""
        texts = [entry[0] for entry in TRAINING_DATA]
        labels = [entry[1] for entry in TRAINING_DATA]

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self._is_trained = True

    def predict(self, skills: list[str]) -> dict:
        """
        Predict the best career domain for the given skill set.

        Parameters
        ----------
        skills : list[str]
            Student's skills.

        Returns
        -------
        dict with keys:
            - predicted_domain: str
            - confidence: float (0–100)
            - all_probabilities: dict mapping each domain to its probability
            - top_3: list of (domain, probability) tuples
        """
        if not skills or not self._is_trained:
            return {
                "predicted_domain": "General",
                "confidence": 0.0,
                "all_probabilities": {},
                "top_3": [],
            }

        # Build input text from skills
        skill_text = " ".join(skills).lower()
        X = self.vectorizer.transform([skill_text])

        # Predict with probabilities
        predicted = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        # Build probability map
        prob_map = {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(classes, probabilities)
        }

        # Sort by probability
        sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        confidence = sorted_probs[0][1] if sorted_probs else 0.0

        return {
            "predicted_domain": predicted,
            "confidence": confidence,
            "all_probabilities": prob_map,
            "top_3": sorted_probs[:3],
        }

    def get_model_accuracy(self) -> float:
        """
        Estimate model accuracy via 3-fold cross-validation.
        """
        texts = [entry[0] for entry in TRAINING_DATA]
        labels = [entry[1] for entry in TRAINING_DATA]
        X = self.vectorizer.transform(texts)
        scores = cross_val_score(self.classifier, X, labels, cv=3)
        return round(float(scores.mean()) * 100, 1)


# Singleton instance — trained once and reused
_predictor_instance = None


def get_predictor() -> DomainPredictor:
    """Return a singleton DomainPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DomainPredictor()
    return _predictor_instance
