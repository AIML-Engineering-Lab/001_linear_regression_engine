"""
Train pipeline for Linear Regression Engine.
Trains Ridge model, saves fitted pipeline.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train(dataset: str = "artisan_cheese_fermentation_data.csv"):
    """Train model pipeline and save to models/."""
    print(f"Loading {dataset}...")
    df = pd.read_csv(DATA_DIR / dataset)

    # Separate features and target (last column)
    target_col = df.columns[-1]
    X = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Ridge...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    pipe.fit(X_train, y_train)

    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")

    model_path = MODEL_DIR / "ridge_pipeline.pkl"
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")
    return pipe


if __name__ == "__main__":
    train()
