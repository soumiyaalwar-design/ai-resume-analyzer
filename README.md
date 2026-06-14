# AI Resume Analyzer

An AI-powered web application that analyzes resumes (PDF format), extracts skills using NLP, scores the resume, and matches it against job-specific skill requirements — providing actionable improvement suggestions.

Built with **Python**, **Streamlit**, **PyPDF2**, and **spaCy**.

---

##  Features

-  **Upload Resume (PDF)** — simple drag-and-drop file uploader
-  **Text Extraction & Cleaning** — extracts and preprocesses raw resume text
-  **NLP-Based Skill Extraction** — uses spaCy's `PhraseMatcher` to accurately detect single- and multi-word skills (e.g., "Machine Learning", "Deep Learning")
-  **Resume Score** — calculates an overall score based on detected skills
-  **Job Role Matching** — select a target role (Data Scientist, AI Engineer, etc.) and get a percentage match score
-  **Matched & Missing Skills** — clear visual breakdown of skill gaps
-  **Improvement Suggestions** — personalized, actionable tips to improve your resume
-  **Clean, Modern UI** — custom CSS styling for a professional look

---

##  Tech Stack

| Component        | Technology        |
|-------------------|-------------------|
| Frontend / UI     | Streamlit         |
| PDF Parsing       | PyPDF2            |
| NLP Engine        | spaCy (`en_core_web_sm`) |
| Language          | Python 3.9+       |
| Styling           | Custom CSS via Streamlit Markdown |

---

##  Project Structure

```
ai-resume-analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore               # Files/folders excluded from git
└── .streamlit/
    └── config.toml         # Streamlit theme & server settings
```
