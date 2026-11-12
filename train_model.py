"""Train the model and save it for the Streamlit app.

Run once from the attrition-app folder:
    python train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASE = Path(__file__).parent
CSV = BASE / "Employee Attrition.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

if not CSV.exists():
    raise SystemExit(f"Cannot find {CSV.name} in {BASE}. Put the CSV in the same folder as this script.")

df = pd.read_csv(CSV)
df["Attrition"] = df["Attrition"].replace({"Left": 1, "Stayed": 0}).astype(int)

X = df.drop(columns=["Attrition", "Employee ID"])
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)


def add_features(data):
    d = data.copy()
    d["Income per Year at Company"] = d["Monthly Income"] / (d["Years at Company"] + 1)
    d["Promotions per Year"] = d["Number of Promotions"] / (d["Years at Company"] + 1)
    d["Income per Dependent"] = d["Monthly Income"] / (d["Number of Dependents"] + 1)
    d["Age at Joining"] = d["Age"] - d["Years at Company"]
    return d


X_train_fe, X_test_fe = add_features(X_train), add_features(X_test)

ordinal_map = {
    "Work-Life Balance": ["Poor", "Fair", "Good", "Excellent"],
    "Job Satisfaction": ["Low", "Medium", "High", "Very High"],
    "Performance Rating": ["Low", "Below Average", "Average", "High"],
    "Education Level": ["High School", "Associate Degree", "Bachelor’s Degree",
                        "Master’s Degree", "PhD"],
    "Job Level": ["Entry", "Mid", "Senior"],
    "Company Size": ["Small", "Medium", "Large"],
    "Company Reputation": ["Poor", "Fair", "Good", "Excellent"],
    "Employee Recognition": ["Low", "Medium", "High", "Very High"],
}
ordinal_cols = list(ordinal_map)
binary_cols = ["Overtime", "Remote Work", "Leadership Opportunities",
               "Innovation Opportunities", "Gender"]
nominal_cols = ["Job Role", "Marital Status"]
numeric_cols = [c for c in X_train_fe.columns
                if c not in ordinal_cols + binary_cols + nominal_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("ord", Pipeline([
        ("enc", OrdinalEncoder(categories=[ordinal_map[c] for c in ordinal_cols])),
        ("scale", StandardScaler())]), ordinal_cols),
    ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_cols),
    ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
])

X_train_proc = preprocessor.fit_transform(X_train_fe)
X_test_proc = preprocessor.transform(X_test_fe)

# Best parameters found by GridSearchCV in the notebook
model = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train_proc, y_train)

proba = model.predict_proba(X_test_proc)[:, 1]
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test_proc)):.4f}")
print(f"Test ROC-AUC : {roc_auc_score(y_test, proba):.4f}")

joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
joblib.dump(model, MODEL_DIR / "model.joblib")
print(f"\nSaved to {MODEL_DIR}")
print("scikit-learn version used:", sklearn.__version__)
