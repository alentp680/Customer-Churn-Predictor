import pickle
# --------------------------
# Load Model
# --------------------------
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
    
def predict_churn(input_data):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        return prediction,probability