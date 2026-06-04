from groq import Groq
import os

def get_career_advice(skills, predicted_domain, missing_skills):

    client = Groq(
        api_key="YOUR_GROQ_API_KEY", # Replace with your actual API key
        timeout=20.0,        # 20-second timeout for longer responses
        max_retries=1,       # Retry once on transient errors
    )

    prompt = f"""
    Student Skills: {skills}

    Predicted Domain: {predicted_domain}

    Missing Skills: {missing_skills}

    Give:
    1. Career guidance
    2. Internship preparation tips
    3. Learning roadmap
    4. Recommended certifications
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        error_type = type(e).__name__

        # Provide helpful offline advice as fallback
        fallback = f"""
⚠️ **AI Career Coach is temporarily unavailable** ({error_type})

Here's a general guide based on your profile:

### 🎯 Your Skills: {', '.join(skills[:10]) if isinstance(skills, list) else skills}
### 🧠 Predicted Domain: {predicted_domain}

---

### 1. 📋 Career Guidance
- Focus on building projects in **{predicted_domain}** to strengthen your portfolio
- Contribute to open-source projects on GitHub
- Network with professionals in your domain via LinkedIn

### 2. 💡 Internship Preparation Tips
- Practice coding problems on LeetCode/HackerRank
- Build 2-3 strong portfolio projects
- Prepare a tailored resume for {predicted_domain} roles
- Practice common behavioral interview questions

### 3. 🗺️ Learning Roadmap
- **Skills to learn:** {', '.join(missing_skills[:5]) if isinstance(missing_skills, list) and missing_skills else 'Keep building on current skills'}
- Use free resources: Coursera, edX, Khan Academy, YouTube tutorials
- Join relevant communities and study groups

### 4. 🏆 Recommended Certifications
- Google Professional Certificates (Coursera)
- AWS/Azure/GCP Cloud Certifications
- Domain-specific certifications on platforms like Udemy or LinkedIn Learning

---
*Connect to the internet and try again for personalized AI-powered advice.*
"""
        return fallback