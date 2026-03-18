"""Tests for Linear Regression Engine model."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_model_file_exists():
    assert (ROOT / "models" / "ridge_pipeline.pkl").exists(), "Trained model not found. Run src/train.py first."


def test_prediction_output():
    from predict import predict
    model_path = ROOT / "models" / "ridge_pipeline.pkl"
    if not model_path.exists():
        return
    df = pd.read_csv(ROOT / "data" / "artisan_cheese_fermentation_data.csv")
    features = df.drop(columns=[df.columns[-1]]).head(3)
    preds = predict(features)
    assert len(preds) == 3


if __name__ == "__main__":
    test_model_file_exists()
    test_prediction_output()
    print("All tests passed.")
