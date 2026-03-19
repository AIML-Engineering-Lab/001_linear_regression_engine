"""
Train pipeline for Linear Regression Engine.
Trains Ridge model for each dataset and saves fitted pipelines.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "cheese": {
        "file": "artisan_cheese_fermentation_data.csv",
        "model": "ridge_cheese.pkl",
    },
    "fmax": {
        "file": "silicon_fmax_validation_data.csv",
        "model": "ridge_fmax.pkl",
    },
}


def train(name: str):
    """Train model pipeline and save to models/."""
    cfg = DATASETS[name]
    print(f"Loading {cfg['file']}...")
    df = pd.read_csv(DATA_DIR / cfg["file"])

    target_col = df.columns[-1]
    X = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Ridge ({name})...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}")

    model_path = MODEL_DIR / cfg["model"]
    joblib.dump(pipe, model_path)
    print(f"  Model saved to {model_path}")
    return pipe


if __name__ == "__main__":
    for name in DATASETS:
        train(name)
