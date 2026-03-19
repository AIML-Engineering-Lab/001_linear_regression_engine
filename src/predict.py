"""
Inference for Linear Regression Engine.
Load trained model and run predictions on new data.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    """Load model and predict on input DataFrame (features only, no target column)."""
    if model_path is None:
        model_path = str(MODEL_DIR / "ridge_fmax.pkl")

    pipe = joblib.load(model_path)
    X = data.select_dtypes(include="number")
    preds = pipe.predict(X)
    return preds.tolist()


if __name__ == "__main__":
    # Demo: Cheese predictions
    df_cheese = pd.read_csv(ROOT / "data" / "artisan_cheese_fermentation_data.csv")
    features = df_cheese.drop(columns=[df_cheese.columns[-1]]).head(5)
    preds = predict(features, str(MODEL_DIR / "ridge_cheese.pkl"))
    print(f"Cheese predictions: {preds}")

    # Demo: Fmax predictions
    df_fmax = pd.read_csv(ROOT / "data" / "silicon_fmax_validation_data.csv")
    features = df_fmax.drop(columns=[df_fmax.columns[-1]]).head(5)
    preds = predict(features, str(MODEL_DIR / "ridge_fmax.pkl"))
    print(f"Fmax predictions: {preds}")
