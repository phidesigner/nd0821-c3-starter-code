from pathlib import Path
import joblib

MODEL_DIR = Path("starter/model")

try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    lb = joblib.load(MODEL_DIR / "lb.pkl")
    print("Artifacts loaded successfully")
except Exception as e:
    print(f"Failed to load artifacts: {e}")
