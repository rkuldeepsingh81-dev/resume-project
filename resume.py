"""
AI Resume Ranker for Bulk Hiring  -  pure Python version (Streamlit UI)
Install : pip install streamlit scikit-learn pdfplumber python-docx pandas
Run     : streamlit run resume_ranker_app.py
"""
import io
import re
from pathlib import Path

import pandas as pd
import pdfplumber
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 1. RESUME PARSING =====================
SKILLS = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "node.js", "nodejs"],
    "typescript": ["typescript"], "c++": ["c++"], "c#": ["c#"], "sql": ["sql", "mysql", "postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"], "react": ["react", "react.js", "reactjs"], "flask": ["flask"],
    "django": ["django"], "fastapi": ["fastapi"], "html": ["html", "html5"], "css": ["css", "css3"],
    "aws": ["aws", "amazon web services"], "gcp": ["gcp", "google cloud"], "azure": ["azure"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "git": ["git", "github", "gitlab"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"], "pytorch": ["pytorch"], "pandas": ["pandas"], "numpy": ["numpy"],
    "spacy": ["spacy"], "data analysis": ["data analysis", "data analytics"],
    "rest api": ["rest api", "rest apis", "restful"], "linux": ["linux"], "agile": ["agile", "scrum"],
    "excel": ["excel"], "power bi": ["power bi"], "tableau": ["tableau"],
    "communication": ["communication"], "leadership": ["leadership", "team lead"],
}
EDU_LEVELS = [("phd", 4, ["phd", "ph.d", "doctorate"]),
              ("master", 3, ["master", "m.tech", "mtech", "m.sc", "msc", "mba", "m.e."]),
              ("bachelor", 2, ["bachelor", "b.tech", "btech", "b.e.", "b.sc", "bsc", "bca", "b.com"]),
              ("diploma", 1, ["diploma"])]


def extract_text(filename, data):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if ext == ".docx":
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


def has_term(text_lower, term):
    return re.search(r"(?<![a-z0-9+#])" + re.escape(term) + r"(?![a-z0-9+#])", text_lower) is not None


def extract_skills(text):
    t = text.lower()
    return {canon for canon, aliases in SKILLS.items() if any(has_term(t, a) for a in aliases)}


def extract_years_experience(text):
    """Counts only 'N years ... experience' phrases, so '24 years old' is never read as experience."""
    t = text.lower()
    nums = [float(m) for m in re.findall(
        r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:\w+\s+){0,2}?experience", t)]
    nums += [float(m) for m in re.findall(
        r"experience\s*(?:of|:)?\s*(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)", t)]
    return min(max(nums), 40.0) if nums else 0.0


def extract_education(text):
    t, best = text.lower(), ("none", 0)
    for name, level, keys in EDU_LEVELS:
        if any(has_term(t, k) for k in keys) and level > best[1]:
            best = (name, level)
    return best


def extract_contact(text):
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "Unknown")
    return {"name": first_line[:60], "email": email.group(0) if email else ""}


def parse_resume(filename, data):
    text = extract_text(filename, data)
    edu_name, edu_level = extract_education(text)
    return {"filename": filename, "text": text, **extract_contact(text),
            "skills": sorted(extract_skills(text)), "years": extract_years_experience(text),
            "education": edu_name, "edu_level": edu_level}


# ===================== 2. BIAS & FAIRNESS CHECK =====================
PROXY_TERMS = r"\b(he|she|his|her|mr|mrs|ms|male|female|married|unmarried|single|age|dob|date of birth|photo)\b"


def redact_pii(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)
    text = re.sub(r"(?:\+?\d[\d\s-]{8,13}\d)", " ", text)
    return re.sub(PROXY_TERMS, " ", text, flags=re.I)


def has_proxy_terms(text):
    return re.search(PROXY_TERMS, text, flags=re.I) is not None


# ===================== 3. RANKING ENGINE =====================
WEIGHTS = {"skills": 0.50, "text": 0.25, "experience": 0.15, "education": 0.10}


def parse_job(jd_text):
    _, edu_level = extract_education(jd_text)
    return {"text": jd_text, "skills": sorted(extract_skills(jd_text)),
            "years": extract_years_experience(jd_text), "edu_level": edu_level}


def rank_candidates(job, candidates):
    if not candidates:
        return []
    required = set(job["skills"])
    docs = [redact_pii(job["text"])] + [redact_pii(c["text"]) for c in candidates]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()

    results = []
    for c, sim in zip(candidates, sims):
        matched = sorted(required & set(c["skills"]))
        missing = sorted(required - set(c["skills"]))
        skill_score = len(matched) / len(required) if required else 0.0
        exp_score = min(c["years"] / job["years"], 1.0) if job["years"] else (1.0 if c["years"] else 0.5)
        edu_score = min(c["edu_level"] / job["edu_level"], 1.0) if job["edu_level"] else 0.5
        parts = {"skills": skill_score, "text": float(sim), "experience": exp_score, "education": edu_score}
        total = sum(WEIGHTS[k] * v for k, v in parts.items())
        results.append({
            "name": c["name"], "email": c["email"], "score": round(total * 100, 1),
            "years": c["years"], "education": c["education"],
            "matched_skills": matched, "missing_skills": missing,
            "proxy_flag": c.get("proxy_flag", False),
            "explanation": (f"Matched {len(matched)}/{len(required)} required skills; "
                            f"{c['years']:g} yrs experience; education: {c['education']}. "
                            f"Breakdown -> skills {skill_score:.0%}, text {float(sim):.0%}, "
                            f"experience {exp_score:.0%}, education {edu_score:.0%}."),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


# ===================== 4. RECRUITER DASHBOARD (Streamlit) =====================
def main():
    st.set_page_config(page_title="AI Resume Ranker", layout="wide")
    st.title("AI Resume Ranker for Bulk Hiring")
    st.caption("Upload resumes in bulk, get an explainable ranked shortlist.")

    jd = st.text_area("Job description", height=160,
                      placeholder="e.g. ML Engineer, 2+ years of experience. Python, NLP, Flask, SQL, Docker...")
    files = st.file_uploader("Resumes (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)

    if st.button("Rank resumes", type="primary"):
        if not jd.strip() or not files:
            st.error("Provide a job description and at least one resume.")
            return
        candidates = []
        for f in files:
            try:
                cand = parse_resume(f.name, f.read())
                cand["proxy_flag"] = has_proxy_terms(cand["text"])
                candidates.append(cand)
            except Exception as exc:                      # one bad file must not stop the batch
                st.warning(f"Skipped {f.name}: {exc}")

        job = parse_job(jd)
        results = rank_candidates(job, candidates)
        st.subheader("Ranked shortlist")
        st.write("**Required skills detected:** " + (", ".join(job["skills"]) or "none"))

        df = pd.DataFrame([{"Rank": r["rank"], "Name": r["name"], "Email": r["email"], "Fit score": r["score"],
                            "Years": r["years"], "Education": r["education"],
                            "Matched": ", ".join(r["matched_skills"]),
                            "Missing": ", ".join(r["missing_skills"])} for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Fit score": st.column_config.ProgressColumn(
                         "Fit score", min_value=0, max_value=100, format="%.1f")})
        st.download_button("Export shortlist (CSV)", df.to_csv(index=False).encode(),
                           "shortlist.csv", "text/csv")

        st.subheader("Why each candidate ranked here")
        for r in results:
            with st.expander(f"#{r['rank']}  {r['name']}  -  {r['score']}"):
                st.write(r["explanation"])
                st.write("Missing skills: " + (", ".join(r["missing_skills"]) or "none"))
                if r["proxy_flag"]:
                    st.caption("Contains personal/protected terms - these were ignored in scoring.")


if __name__ == "__main__":
    main()