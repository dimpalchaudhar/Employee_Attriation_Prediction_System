from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"


# ---------- Load artefacts once ----------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model = joblib.load(MODEL_DIR / "model.joblib")
    return preprocessor, model


# ---------- Same feature engineering as in the notebook ----------
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["Income per Year at Company"] = d["Monthly Income"] / (d["Years at Company"] + 1)
    d["Promotions per Year"] = d["Number of Promotions"] / (d["Years at Company"] + 1)
    d["Income per Dependent"] = d["Monthly Income"] / (d["Number of Dependents"] + 1)
    d["Age at Joining"] = d["Age"] - d["Years at Company"]
    return d


try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Details below help find the problem.")
    st.write("Looking for the model in:", str(MODEL_DIR))
    st.write("`model/` folder exists:", MODEL_DIR.exists())
    st.write("Files in model/:", [f.name for f in MODEL_DIR.iterdir()] if MODEL_DIR.exists() else [])
    st.write("Files next to app.py:", sorted(f.name for f in MODEL_DIR.parent.iterdir()))
    st.stop()
except Exception as e:  # e.g. scikit-learn version mismatch
    st.error(f"The model files were found but could not be loaded: {type(e).__name__}: {e}")
    st.stop()

# ---------- Header ----------
st.title("👥 Employee Attrition Prediction")
st.caption("Enter an employee's details to estimate the probability that they will leave the company.")

# ---------- Input form ----------
with st.form("employee_form"):
    st.subheader("Personal details")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.slider("Age", 18, 59, 35)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    marital = c3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = c4.slider("Number of Dependents", 0, 6, 1)

    c1, c2, c3 = st.columns(3)
    education = c1.selectbox("Education Level", [
        "High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"], index=2)
    distance = c2.slider("Distance from Home (km)", 1, 99, 25)
    remote = c3.selectbox("Remote Work", ["No", "Yes"])

    st.subheader("Job details")
    c1, c2, c3, c4 = st.columns(4)
    job_role = c1.selectbox("Job Role", ["Education", "Media", "Healthcare", "Technology", "Finance"])
    job_level = c2.selectbox("Job Level", ["Entry", "Mid", "Senior"], index=1)
    income = c3.number_input("Monthly Income", 1000, 20000, 7000, step=100)
    overtime = c4.selectbox("Overtime", ["No", "Yes"])

    c1, c2, c3, c4 = st.columns(4)
    years = c1.slider("Years at Company", 1, 51, 5)
    promotions = c2.slider("Number of Promotions", 0, 4, 1)
    performance = c3.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"], index=2)
    leadership = c4.selectbox("Leadership Opportunities", ["No", "Yes"])

    st.subheader("Satisfaction & company")
    c1, c2, c3, c4 = st.columns(4)
    wlb = c1.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"], index=2)
    satisfaction = c2.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"], index=1)
    recognition = c3.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"], index=1)
    innovation = c4.selectbox("Innovation Opportunities", ["No", "Yes"])

    c1, c2, c3 = st.columns(3)
    company_size = c1.selectbox("Company Size", ["Small", "Medium", "Large"], index=1)
    reputation = c2.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"], index=2)
    company_tenure = c3.slider("Company Tenure (months)", 2, 128, 56)

    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

# ---------- Prediction ----------
if submitted:
    employee = pd.DataFrame([{
        "Age": age, "Gender": gender, "Years at Company": years, "Job Role": job_role,
        "Monthly Income": income, "Work-Life Balance": wlb, "Job Satisfaction": satisfaction,
        "Performance Rating": performance, "Number of Promotions": promotions,
        "Overtime": overtime, "Distance from Home": distance, "Education Level": education,
        "Marital Status": marital, "Number of Dependents": dependents, "Job Level": job_level,
        "Company Size": company_size, "Company Tenure": company_tenure, "Remote Work": remote,
        "Leadership Opportunities": leadership, "Innovation Opportunities": innovation,
        "Company Reputation": reputation, "Employee Recognition": recognition,
    }])

    X = preprocessor.transform(add_features(employee))
    prob = float(model.predict_proba(X)[0, 1])

    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 2])
    left.metric("Probability of leaving", f"{prob:.1%}")

    if prob >= 0.70:
        right.error("🔴 **High risk** – this employee is likely to leave.")
    elif prob >= 0.50:
        right.warning("🟠 **Moderate risk** – this employee may leave.")
    else:
        right.success("🟢 **Low risk** – this employee is likely to stay.")
    right.progress(min(max(prob, 0.0), 1.0))

    with st.expander("See the input data sent to the model"):
        st.dataframe(employee.T.astype(str).rename(columns={0: "Value"}), width="stretch")

st.caption("Model: Gradient Boosting Classifier · Test accuracy ≈ 75.6% · ROC-AUC ≈ 0.85")
