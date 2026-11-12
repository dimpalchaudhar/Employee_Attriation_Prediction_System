# Paste this in a NEW cell at the END of your notebook and run it
import joblib, os, sklearn

os.makedirs("model", exist_ok=True)
joblib.dump(preprocessor, "model/preprocessor.joblib")
joblib.dump(best_model,   "model/model.joblib")

print("Saved! scikit-learn version used:", sklearn.__version__)

