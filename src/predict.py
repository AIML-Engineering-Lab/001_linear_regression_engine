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
        model_path = str(MODEL_DIR / "ridge_pipeline.pkl")

    pipe = joblib.load(model_path)
    X = data.select_dtypes(include="number")
    preds = pipe.predict(X)
    return preds.tolist()


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "artisan_cheese_fermentation_data.csv")
    # Drop target column (last) to pass only features
    features = df.drop(columns=[df.columns[-1]]).head(5)
    preds = predict(features)
    print(f"Predictions: {preds}")
