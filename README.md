# Employee Attrition Prediction System

A machine-learning app that predicts whether an employee is likely to leave a company.
Built with **scikit-learn (Gradient Boosting)** and deployed with **Streamlit**.

## Features
- Interactive form for 22 employee attributes
- Probability of leaving + Low / Moderate / High risk label
- Same preprocessing and feature engineering as the training notebook

## Model performance (20% held-out test set)
| Metric | Score |
|---|---|
| Accuracy | 0.756 |
| Precision | 0.745 |
| Recall | 0.741 |
| F1 | 0.743 |
| ROC-AUC | 0.851 |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
├── app.py                  # Streamlit app
├── model/
│   ├── preprocessor.joblib # fitted ColumnTransformer
│   └── model.joblib        # tuned Gradient Boosting model
├── Employee_Attriation_Prediction_System.ipynb
├── requirements.txt
└── README.md
```

## Live demo
<!-- paste your Streamlit link here -->
