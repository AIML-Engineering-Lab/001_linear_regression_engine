# Product Requirements Document (PRD)

## Project: 001_linear_regression_engine
**Series:** The AI Engineering Lab (Project 1 of 52)  
**Status:** Draft for Review  

---

## 1. Project Overview

**What we are building:**  
A comprehensive Linear Regression Engine implementing Ordinary Least Squares (OLS), Ridge, Lasso, and Elastic Net regression models. The project will build these models from scratch (to demonstrate mathematical first principles) and via industry-standard libraries (for production readiness).

**Engineering Objective:**  
Linear regression is the foundational algorithm for understanding supervised learning, cost functions, and optimization. Mastering these techniques—specifically L1 and L2 regularization—enables engineers to build robust predictive models in noisy, high-dimensional, real-world scenarios. This project establishes the baseline architecture, data handling protocols, and visualization standards for all subsequent AI Engineering Lab projects.

---

## 2. Dataset Strategy (Dual Dataset Approach)

To maximize educational value and demonstrate the universal applicability of these algorithms, this project will utilize two distinct synthetic datasets: one novel general dataset and one domain-specific post-silicon validation (Posiva) dataset.

### Dataset A: The Novel General Dataset
**Concept:** *Predicting the Optimal Fermentation Time of Artisan Cheese*  
A scientifically grounded, continuous target variable perfect for regression analysis, avoiding overused benchmarks.

**Features (Predictors):**  

| Feature Name | Description | Type | Units |
| :--- | :--- | :--- | :--- |
| `milk_fat_percentage` | Fat content percentage in raw milk | Continuous | % |
| `starter_culture_ph` | pH level of the starter culture before fermentation | Continuous | pH |
| `ambient_temperature` | Ambient temperature during fermentation | Continuous | °C |
| `fermentation_humidity` | Relative humidity in the fermentation chamber | Continuous | % |
| `salt_concentration` | Salt concentration added to the curd | Continuous | % w/w |
| `curd_cut_size` | Average size of curd pieces after cutting | Continuous | cm |
| `aging_room_airflow` | Airflow rate in the aging room | Continuous | m/s |
| `bacterial_strain_type` | Categorical representation of the bacterial strain used | Categorical | N/A |

**Target Variable:**  
- `optimal_fermentation_time`: The ideal fermentation duration to achieve desired texture and flavor (Continuous, Hours).

### Dataset B: The Post-Silicon Validation (Posiva) Dataset
**Concept:** *Predicting Maximum Stable Clock Frequency (Fmax) Across Voltage/Temperature Corners*  
A highly relevant engineering dataset that models the non-linear relationship between voltage, temperature, silicon process variations, and the maximum frequency a chip can sustain before failing.

**Features (Predictors):**  

| Feature Name | Description | Type | Units |
| :--- | :--- | :--- | :--- |
| `vdd_core` | Core supply voltage | Continuous | V |
| `junction_temp` | Silicon junction temperature | Continuous | °C |
| `leakage_current` | Static leakage current measured at test | Continuous | mA |
| `ring_oscillator_speed` | Speed of on-die ring oscillator (process proxy) | Continuous | MHz |
| `thermal_resistance` | Package thermal resistance (Theta-JA) | Continuous | °C/W |
| `ir_drop_estimate` | Estimated voltage drop across the power grid | Continuous | mV |
| `silicon_lot_id` | Manufacturing lot identifier | Categorical | N/A |

**Target Variable:**  
- `fmax_mhz`: The maximum stable operating frequency before timing failure (Continuous, MHz).

---

## 3. Technical Scope

**Models to Implement:**  
1. **Ordinary Least Squares (OLS):** Closed-form mathematical solution and iterative Gradient Descent approach.
2. **Ridge Regression (L2):** Handling multicollinearity by penalizing large coefficients.
3. **Lasso Regression (L1):** Feature selection via sparse coefficient vectors.
4. **Elastic Net:** Combining L1 and L2 penalties for robust feature selection.

**Core Concepts to Document:**  
- **Cost Functions:** Mean Squared Error (MSE) and regularized loss functions.
- **Gradient Descent:** The optimization mechanism for minimizing cost.
- **Data Leakage:** Definition, risks, and prevention during preprocessing.
- **Data Partitioning:** Proper Train/Validation/Test splitting strategies.

---

## 4. Architecture & Block Diagram

```text
[Raw Data Generation (Cheese & Fmax)] --> [Exploratory Data Analysis (EDA)]
                                                |
                                                v
[Data Preprocessing] <--- (Handling Missing Values, Scaling, Encoding)
                                                |
                                                v
[Data Partitioning] ----> (Train / Validation / Test Split)
                                                |
                                                v
[Model Training Engine]
    |-- OLS (Closed Form)
    |-- OLS (Gradient Descent)
    |-- Ridge (L2)
    |-- Lasso (L1)
    |-- Elastic Net
                                                |
                                                v
[Hyperparameter Tuning] -> (Cross-Validation for Alpha/Lambda)
                                                |
                                                v
[Model Evaluation] ------> (MSE, RMSE, MAE, R-squared)
                                                |
                                                v
[Visualization Engine] --> (3D Cost Surfaces, 2D Contour Plots, Residuals)
```

---

## 5. Tech Stack

- **Language:** Python 3.11+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **2D Visualization:** `matplotlib`, `seaborn`
- **3D Visualization:** `plotly` (for interactive surfaces) or `matplotlib.pyplot` (for static renders)
- **Environment:** Jupyter Notebook (`.ipynb`)

---

## 6. Repository Structure

```text
001_linear_regression_engine/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── artisan_cheese_fermentation_data.csv
│   └── silicon_fmax_validation_data.csv
├── notebooks/
│   ├── 01_linear_regression_cheese.ipynb
│   └── 02_linear_regression_fmax.ipynb
├── src/
│   ├── data_generator.py       # Script to generate both synthetic datasets
│   └── visualizer.py           # Helper functions for complex 3D plots
└── assets/
    ├── fig1_cost_surface_3d.png
    ├── fig2_l1_l2_contours.png
    └── fig3_coefficient_paths.png
```

---

## 7. Visualizations

**Signature 3D Visualization:**  
- **3D Cost Function Surface:** A 3D plot showing the Mean Squared Error (MSE) bowl landscape plotted against two model coefficients. A distinct path (e.g., red arrows) will overlay the surface, illustrating the step-by-step convergence of the Gradient Descent algorithm to the global minimum.

**2D Visualizations:**  
- **L1 vs L2 Regularization Contours:** Side-by-side contour plots showing the geometric difference between the L2 "sphere" constraint and the L1 "diamond" constraint, visually explaining why Lasso drives coefficients to exactly zero.
- **Coefficient Shrinkage Paths:** A line plot showing how the magnitude of different feature coefficients shrinks as the regularization penalty (alpha) increases.
- **Residual Analysis:** A scatter plot of residuals vs. predicted values to verify homoscedasticity.

---

## 8. Implementation Prompt (For AI Agent / Developer)

> **Role:** You are a Principal Data Scientist.
> **Task:** Implement the `001_linear_regression_engine` project exactly as specified in this PRD.
> 
> **Steps:**
> 1. Create the repository structure.
> 2. Write `src/data_generator.py` to generate 2,000 rows for both the artisan cheese dataset and the silicon Fmax dataset with realistic correlations. Save them to `data/`.
> 3. Develop the two notebooks (`01_linear_regression_cheese.ipynb` and `02_linear_regression_fmax.ipynb`). The notebooks must be highly educational, using markdown cells to explain the math behind OLS, Ridge, Lasso, and Elastic Net.
> 4. Implement OLS from scratch using numpy (both closed-form and gradient descent).
> 5. Implement Ridge, Lasso, and Elastic Net using `scikit-learn`.
> 6. Generate the signature 3D cost function surface plot and the 2D contour plots. Save these high-quality images to the `assets/` folder.
> 7. Ensure all code is PEP8 compliant, well-commented, and professional. Do not include any references to social media or external posting platforms.
