# Project 001: The Linear Regression Engine

**The AI Engineering Lab** | Post 1 of the Progressive AIML Series

---

## Overview

This project builds a complete, production-quality linear regression pipeline from mathematical first principles. It covers Ordinary Least Squares (OLS), Ridge (L2), Lasso (L1), and Elastic Net regularization — applied to two independent datasets from entirely different domains to demonstrate the universality of the technique.

The goal is not just to fit a line. It is to understand *why* the math works, *what* regularization actually does geometrically, and *how* to make principled decisions about model selection and hyperparameter tuning.

---

## Datasets

### Dataset A: Artisan Cheese Fermentation Time Prediction

A synthetic dataset of 2,000 artisan cheese batches. The task is to predict the optimal fermentation time (in hours) based on biochemical and environmental conditions including milk fat percentage, starter culture pH, ambient temperature, humidity, salt concentration, curd cut size, aging room airflow, and bacterial strain type.

This dataset is designed to be scientifically grounded in fermentation biochemistry while being completely novel as a machine learning benchmark.

### Dataset B: Silicon Fmax Prediction (Post-Silicon Validation)

A synthetic dataset of 2,000 silicon characterization measurements. The task is to predict the maximum stable clock frequency (Fmax in MHz) based on voltage, temperature, leakage current, ring oscillator speed, thermal resistance, IR drop estimate, and silicon lot ID.

This represents a real post-silicon validation use case: predicting chip performance at untested voltage and temperature corners to reduce physical characterization time.

---

## What This Project Covers

| Concept | Description |
|:---|:---|
| **Ordinary Least Squares (OLS)** | Closed-form Normal Equation and iterative Gradient Descent implementations from scratch |
| **Cost Function** | MSE as a convex bowl in coefficient space; 3D visualization |
| **Gradient Descent** | Learning rate analysis, convergence behavior, update rule derivation |
| **Ridge (L2) Regularization** | Sphere constraint, coefficient shrinkage, no feature elimination |
| **Lasso (L1) Regularization** | Diamond constraint, automatic feature selection, sparsity |
| **Elastic Net** | Combined L1+L2 penalty for correlated feature sets |
| **Cross-Validation** | K-fold CV for hyperparameter selection without test set leakage |
| **Residual Analysis** | Checking model assumptions through residual diagnostics |
| **Feature Preprocessing** | StandardScaler, OneHotEncoder, ColumnTransformer pipelines |

---

## Repository Structure

```
001_linear_regression_engine/
├── data/
│   ├── artisan_cheese_fermentation_data.csv
│   └── silicon_fmax_validation_data.csv
├── notebooks/
│   ├── 01_linear_regression_cheese.ipynb
│   └── 02_linear_regression_fmax.ipynb
├── src/
│   ├── data_generator.py
│   └── generate_visuals.py
├── assets/
│   ├── fig1_cost_surface_3d.png
│   ├── fig2_l1_l2_contours.png
│   ├── fig3_coefficient_paths.png
│   ├── fig4_model_comparison.png
│   └── fig5_gd_convergence.png
├── PRD.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Key Visualizations

### 3D MSE Cost Function Surface
The MSE cost function forms a convex bowl in coefficient space. Gradient Descent follows the gradient of this surface downward to the global minimum. This visualization shows the path taken by the optimizer across 80 iterations.

### L1 vs L2 Geometric Interpretation
The fundamental reason Lasso creates sparsity while Ridge does not is geometric. The L2 constraint is a sphere; the L1 constraint is a diamond. The diamond has corners on the axes, making it likely that the constrained solution lands exactly on an axis — driving one coefficient to zero.

### Coefficient Shrinkage Paths
As regularization strength increases, coefficients are penalized more heavily. Ridge shrinks all coefficients gradually but never to zero. Lasso eliminates features entirely beyond a threshold. These paths are shown for both datasets.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/AIML-Engineering-Lab/001_linear_regression_engine.git
cd 001_linear_regression_engine

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python3 src/data_generator.py

# Generate all visualizations
python3 src/generate_visuals.py

# Open notebooks
jupyter notebook notebooks/
```

---

## Tech Stack

| Tool | Version | Purpose |
|:---|:---|:---|
| Python | 3.11+ | Core language |
| NumPy | 1.24+ | Linear algebra, Normal Equation |
| Pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | Ridge, Lasso, ElasticNet, CV |
| Matplotlib | 3.7+ | All visualizations including 3D |
| Seaborn | 0.12+ | Statistical plots |

---

## Series Context

This project is **Post 1** in the AI Engineering Lab series — a progressive, 50+ project curriculum covering the full AIML stack from Classic ML through Deep Learning, Reinforcement Learning, Generative AI, Agentic AI, and MLOps.

Each project applies the same technique to two datasets: one from a novel general domain and one from post-silicon validation, demonstrating that the same mathematical principles generalize across completely different engineering problems.

**Next:** Post 2 — The Classification Engine (Logistic Regression, Decision Boundaries, ROC/AUC)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
