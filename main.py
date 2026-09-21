import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="SkillSync Analytics", layout="centered")

st.title("📊 SkillSync Analytics")
st.caption("Academic Resume Skill-Gap & Benchmark Scoring Engine")

# ---------------------------------------------------------
# 2. BENCHMARK DATASET (PANDAS)
# ---------------------------------------------------------
# ---------------------------------------------------------
# BENCHMARK DATASET: 10 INDUSTRY JOB ROLES (PANDAS)
# ---------------------------------------------------------
benchmark_dict = {
    "Role": [
        "Data Analyst",
        "Java SDE",
        "Cloud Engineer (AWS)",
        "DevOps Engineer",
        "Python Full Stack Developer",
        "Machine Learning Engineer",
        "Cybersecurity Analyst",
        "Data Engineer",
        "Frontend Developer (React)",
        "Backend Developer (Node.js)"
    ],
    "Required_Skills": [
        "python, sql, excel, power bi, statistics, data visualization",
        "java, spring boot, sql, git, data structures, oops, microservices",
        "aws, linux, docker, python, terraform, networking, bash",
        "linux, docker, kubernetes, jenkins, git, ci/cd, terraform, aws",
        "python, django, rest api, html, css, javascript, postgresql, git",
        "python, pandas, numpy, scikit-learn, deep learning, sql, git",
        "networking, linux, cryptography, firewall, security, wireshark, python",
        "sql, python, spark, hadoop, aws, etl, data warehousing, pandas",
        "javascript, react, html, css, git, rest api, redux, responsive design",
        "javascript, node.js, express, mongodb, sql, rest api, docker, git"
    ]
}
df_benchmark = pd.DataFrame(benchmark_dict)# ---------------------------------------------------------
# 3. USER INPUT INTERFACE
# ---------------------------------------------------------
st.subheader("1. Candidate Details")
target_role = st.selectbox("Select Target Job Role", df_benchmark["Role"].unique())

resume_text = st.text_area(
    "Paste Resume Content / Technical Summary",
    placeholder="e.g. Skilled in Python, SQL, Excel. Completed AWS and Coursera certifications..."
).lower()

col1, col2 = st.columns(2)
with col1:
    projects_count = st.number_input("Number of Projects Built", min_value=0, max_value=20, value=2)
with col2:
    cgpa = st.number_input("Academic CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)

# ---------------------------------------------------------
# 4. CORE PROCESSING (PANDAS & NUMPY)
# ---------------------------------------------------------
if st.button("Analyze Readiness"):
    if not resume_text.strip():
        st.warning("Please enter resume content to analyze.")
    else:
        # Extract required skills from Pandas DataFrame
        role_row = df_benchmark[df_benchmark["Role"] == target_role]
        raw_skills = role_row["Required_Skills"].values[0]
        required_skills = [s.strip() for s in raw_skills.split(",")]

        # Skill matching binary vector
        matched_skills = []
        missing_skills = []
        binary_vector = []

        for skill in required_skills:
            if skill in resume_text:
                matched_skills.append(skill)
                binary_vector.append(1)
            else:
                missing_skills.append(skill)
                binary_vector.append(0)

        # NumPy Vectorized Scoring Model
        # Weights: Skills (35%), Projects (40%), Certs (15%), CGPA (10%)
        skill_ratio = np.mean(np.array(binary_vector)) if len(binary_vector) > 0 else 0.0
        score_skills = skill_ratio * 35.0

        score_projects = 40.0 if projects_count >= 2 else (projects_count * 20.0)

        cert_keywords = ["aws", "nptel", "coursera", "oracle", "cisco", "microsoft"]
        cert_found = sum(1 for c in cert_keywords if c in resume_text)
        score_certs = 15.0 if cert_found >= 2 else (cert_found * 7.5)

        score_cgpa = 10.0 if cgpa >= 7.5 else 5.0

        # Vector Summation using NumPy
        score_array = np.array([score_skills, score_projects, score_certs, score_cgpa])
        final_readiness_score = float(np.round(np.sum(score_array), 2))

        # ---------------------------------------------------------
        # 5. OUTPUT DISPLAY & METRICS
        # ---------------------------------------------------------
        st.divider()
        st.subheader("2. Evaluation Summary")

        st.metric(label="Final Readiness Score", value=f"{final_readiness_score} / 100")

        # Pandas Score Breakdown Table
        breakdown_df = pd.DataFrame({
            "Component": ["Technical Skills Match", "Projects Evaluation", "Certifications", "Academic CGPA"],
            "Weightage": [35, 40, 15, 10],
            "Score Awarded": np.round(score_array, 2)
        })
        st.table(breakdown_df)

        st.write(f"✅ **Matched Skills ({len(matched_skills)}):** {', '.join(matched_skills) if matched_skills else 'None'}")
        st.write(f"⚠️ **Missing Skills / Skill Gap ({len(missing_skills)}):** {', '.join(missing_skills) if missing_skills else 'None'}")

        # ---------------------------------------------------------
        # 6. VISUALIZATION (MATPLOTLIB)
        # ---------------------------------------------------------
        st.subheader("3. Visual Skill-Gap Breakdown")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        categories = ["Skills", "Projects", "Certs", "CGPA"]
        awarded = score_array
        max_weights = np.array([35, 40, 15, 10])

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, max_weights, width, label="Max Weightage", color="#D3D3D3")
        ax.bar(x + width/2, awarded, width, label="Your Score", color="#1F77B4")

        ax.set_ylabel("Points")
        ax.set_title(f"Score Allocation by Domain ({target_role})")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        st.pyplot(fig)