"""
Visualization Generator for Project 001: Linear Regression Engine
Generates all standalone publication-quality figures for the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data"

# ─────────────────────────────────────────────
# Load and preprocess Fmax dataset
# ─────────────────────────────────────────────
df_fmax = pd.read_csv(DATA_DIR / "silicon_fmax_validation_data.csv")
NUMERIC_F = ['vdd_core', 'junction_temp', 'leakage_current',
             'ring_oscillator_speed', 'thermal_resistance', 'ir_drop_estimate']
CAT_F = ['silicon_lot_id']
TARGET_F = 'fmax_mhz'

X_f = df_fmax[NUMERIC_F + CAT_F]
y_f = df_fmax[TARGET_F]
X_tr, X_te, y_tr, y_te = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
prep_f = ColumnTransformer([('num', StandardScaler(), NUMERIC_F),
                             ('cat', OneHotEncoder(drop='first', sparse_output=False), CAT_F)])
X_tr_p = prep_f.fit_transform(X_tr)
X_te_p = prep_f.transform(X_te)

# ─────────────────────────────────────────────
# Load and preprocess Cheese dataset
# ─────────────────────────────────────────────
df_ch = pd.read_csv(DATA_DIR / "artisan_cheese_fermentation_data.csv")
NUMERIC_C = ['milk_fat_percentage', 'starter_culture_ph', 'ambient_temperature',
             'fermentation_humidity', 'salt_concentration', 'curd_cut_size', 'aging_room_airflow']
CAT_C = ['bacterial_strain_type']
TARGET_C = 'optimal_fermentation_time'

X_c = df_ch[NUMERIC_C + CAT_C]
y_c = df_ch[TARGET_C]
X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
prep_c = ColumnTransformer([('num', StandardScaler(), NUMERIC_C),
                             ('cat', OneHotEncoder(drop='first', sparse_output=False), CAT_C)])
X_ctr_p = prep_c.fit_transform(X_ctr)
X_cte_p = prep_c.transform(X_cte)


# ─────────────────────────────────────────────
# FIGURE 1: 3D Cost Function Surface (Signature Visual)
# ─────────────────────────────────────────────
def fig1_cost_surface():
    print("Generating Fig 1: 3D Cost Surface...")
    X_2f = X_tr_p[:, :2]
    y_arr = y_tr.values
    ols_2f = LinearRegression(fit_intercept=True).fit(X_2f, y_arr)
    w0_opt, w1_opt = ols_2f.coef_[0], ols_2f.coef_[1]

    w0_range = np.linspace(w0_opt - 400, w0_opt + 400, 60)
    w1_range = np.linspace(w1_opt - 150, w1_opt + 150, 60)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    COST = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            y_hat = X_2f @ np.array([W0[i,j], W1[i,j]]) + ols_2f.intercept_
            COST[i,j] = np.mean((y_arr - y_hat)**2)

    # Gradient descent path
    lr, w = 0.06, np.array([w0_opt - 350.0, w1_opt - 120.0])
    path, path_cost = [w.copy()], []
    for _ in range(80):
        y_hat = X_2f @ w + ols_2f.intercept_
        grad = -2 * X_2f.T @ (y_arr - y_hat) / len(y_arr)
        w = w - lr * grad
        path.append(w.copy())
    path = np.array(path)
    for p in path:
        y_hat = X_2f @ p + ols_2f.intercept_
        path_cost.append(np.mean((y_arr - y_hat)**2))

    min_cost = np.mean((y_arr - ols_2f.predict(X_2f))**2)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W0, W1, COST, cmap='viridis', alpha=0.55, linewidth=0, antialiased=True)
    ax.plot(path[:,0], path[:,1], path_cost, 'r.-', markersize=4, linewidth=2, label='Gradient Descent Path', zorder=5)
    ax.scatter([w0_opt], [w1_opt], [min_cost], color='gold', s=150, zorder=10, label='Global Minimum (OLS Solution)')
    ax.set_xlabel('w₀  (VDD coefficient)', fontsize=9, labelpad=10)
    ax.set_ylabel('w₁  (Temp coefficient)', fontsize=9, labelpad=10)
    ax.set_zlabel('MSE Cost', fontsize=9, labelpad=10)
    ax.set_title('MSE Cost Function Surface — Silicon Fmax Prediction\nGradient Descent Converging to the Global Minimum',
                 fontsize=12, fontweight='bold', pad=18)
    ax.legend(loc='upper right', fontsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.35, aspect=12, label='MSE', pad=0.1)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig1_cost_surface_3d.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: fig1_cost_surface_3d.png")


# ─────────────────────────────────────────────
# FIGURE 2: L1 vs L2 Geometric Interpretation
# ─────────────────────────────────────────────
def fig2_l1_l2_contours():
    print("Generating Fig 2: L1 vs L2 Contours...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    w_ols = np.array([1.5, 0.9])
    w1g = np.linspace(-2.2, 2.5, 300)
    w2g = np.linspace(-1.8, 2.2, 300)
    W1g, W2g = np.meshgrid(w1g, w2g)
    COST_G = 0.7*(W1g - w_ols[0])**2 + 2.2*(W2g - w_ols[1])**2

    for ax, title, color, is_lasso in zip(
        axes,
        ['Ridge (L2): Sphere Constraint\nCoefficients shrink but never reach zero',
         'Lasso (L1): Diamond Constraint\nCoefficients can be driven to exactly zero'],
        ['#4CAF50', '#FF9800'], [False, True]
    ):
        cs = ax.contour(W1g, W2g, COST_G, levels=10, cmap='Blues', alpha=0.75)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('w₁ (Feature 1 coefficient)', fontsize=10)
        ax.set_ylabel('w₂ (Feature 2 coefficient)', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.scatter(*w_ols, color='#2196F3', s=100, zorder=5, label='OLS Minimum')
        ax.annotate('OLS Min', w_ols + 0.05, fontsize=9, color='#2196F3')
        ax.grid(True, alpha=0.25)

        r = 0.95
        if not is_lasso:
            theta = np.linspace(0, 2*np.pi, 300)
            ax.fill(r*np.cos(theta), r*np.sin(theta), alpha=0.15, color=color)
            ax.plot(r*np.cos(theta), r*np.sin(theta), color=color, linewidth=2.5, label='L2 Constraint Region')
            # Constrained solution: closest point on sphere to OLS min
            ax.scatter([0.88], [0.45], color=color, s=120, zorder=6, marker='*', label='Constrained Solution')
        else:
            dx = [r, 0, -r, 0, r]
            dy = [0, r, 0, -r, 0]
            ax.fill(dx, dy, alpha=0.15, color=color)
            ax.plot(dx, dy, color=color, linewidth=2.5, label='L1 Constraint Region')
            ax.scatter([0.0], [0.95], color=color, s=120, zorder=6, marker='*', label='Constrained Solution (on axis = zero!)')
            ax.annotate('w₁ = 0 here!', xy=(0.0, 0.95), xytext=(0.3, 1.3),
                       arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(-2.2, 2.5)
        ax.set_ylim(-1.8, 2.2)

    plt.suptitle('Why Lasso Creates Sparsity: The Geometric Intuition Behind L1 vs L2 Regularization',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig2_l1_l2_contours.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: fig2_l1_l2_contours.png")


# ─────────────────────────────────────────────
# FIGURE 3: Coefficient Shrinkage Paths (Both Datasets)
# ─────────────────────────────────────────────
def fig3_coefficient_paths():
    print("Generating Fig 3: Coefficient Shrinkage Paths...")
    alphas = np.logspace(-3, 3, 80)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    datasets = [
        (X_tr_p, y_tr.values, NUMERIC_F + ['lot_1','lot_2','lot_3','lot_4'], 'Fmax (Silicon)'),
        (X_ctr_p, y_ctr.values, NUMERIC_C + ['strain_1','strain_2','strain_3'], 'Fermentation (Cheese)'),
    ]

    cmap = plt.cm.tab10
    for row, (X_d, y_d, feat_names, ds_name) in enumerate(datasets):
        ridge_coefs = np.array([Ridge(alpha=a).fit(X_d, y_d).coef_ for a in alphas])
        lasso_coefs = np.array([Lasso(alpha=a, max_iter=5000).fit(X_d, y_d).coef_ for a in alphas])

        for i, coef in enumerate(ridge_coefs.T):
            axes[row,0].plot(alphas, coef, linewidth=1.5, color=cmap(i % 10),
                            label=feat_names[i] if i < len(feat_names) else f'f{i}')
        axes[row,0].set_xscale('log')
        axes[row,0].set_title(f'Ridge (L2) Coefficient Paths — {ds_name}', fontsize=11, fontweight='bold')
        axes[row,0].set_xlabel('Alpha (Regularization Strength)')
        axes[row,0].set_ylabel('Coefficient Value')
        axes[row,0].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[row,0].grid(True, alpha=0.25)
        axes[row,0].legend(fontsize=7, loc='upper right', ncol=2)

        for i, coef in enumerate(lasso_coefs.T):
            axes[row,1].plot(alphas, coef, linewidth=1.5, color=cmap(i % 10),
                            label=feat_names[i] if i < len(feat_names) else f'f{i}')
        axes[row,1].set_xscale('log')
        axes[row,1].set_title(f'Lasso (L1) Coefficient Paths — {ds_name}', fontsize=11, fontweight='bold')
        axes[row,1].set_xlabel('Alpha (Regularization Strength)')
        axes[row,1].set_ylabel('Coefficient Value')
        axes[row,1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[row,1].grid(True, alpha=0.25)
        axes[row,1].legend(fontsize=7, loc='upper right', ncol=2)

    plt.suptitle('Coefficient Shrinkage Paths: Ridge vs Lasso\nBoth Datasets Side by Side',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig3_coefficient_paths.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: fig3_coefficient_paths.png")


# ─────────────────────────────────────────────
# FIGURE 4: Model Comparison Dashboard
# ─────────────────────────────────────────────
def fig4_model_comparison():
    print("Generating Fig 4: Model Comparison Dashboard...")
    alphas_cv = np.logspace(-3, 3, 80)

    # Fmax models
    ols_f   = LinearRegression().fit(X_tr_p, y_tr)
    ridge_f = RidgeCV(alphas=alphas_cv, cv=5).fit(X_tr_p, y_tr)
    lasso_f = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000).fit(X_tr_p, y_tr)

    # Cheese models
    ols_c   = LinearRegression().fit(X_ctr_p, y_ctr)
    ridge_c = RidgeCV(alphas=alphas_cv, cv=5).fit(X_ctr_p, y_ctr)
    lasso_c = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000).fit(X_ctr_p, y_ctr)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row, (X_te_d, y_te_d, models_d, title_prefix) in enumerate([
        (X_te_p,  y_te,  {'OLS': ols_f, 'Ridge': ridge_f, 'Lasso': lasso_f}, 'Fmax (Silicon)'),
        (X_cte_p, y_cte, {'OLS': ols_c, 'Ridge': ridge_c, 'Lasso': lasso_c}, 'Fermentation (Cheese)'),
    ]):
        colors_m = {'OLS': '#2196F3', 'Ridge': '#4CAF50', 'Lasso': '#FF9800'}
        rmse_vals, r2_vals = {}, {}
        for name, model in models_d.items():
            y_pred = model.predict(X_te_d)
            rmse_vals[name] = np.sqrt(mean_squared_error(y_te_d, y_pred))
            r2_vals[name]   = r2_score(y_te_d, y_pred)

        # Bar: RMSE
        bars = axes[row,0].bar(rmse_vals.keys(), rmse_vals.values(),
                               color=[colors_m[k] for k in rmse_vals], edgecolor='white', width=0.5)
        axes[row,0].set_title(f'Test RMSE Comparison — {title_prefix}', fontsize=11, fontweight='bold')
        axes[row,0].set_ylabel('RMSE')
        axes[row,0].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, rmse_vals.values()):
            axes[row,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Predicted vs Actual for best model (Ridge)
        y_pred_best = models_d['Ridge'].predict(X_te_d)
        axes[row,1].scatter(y_te_d, y_pred_best, alpha=0.3, s=12, color='#4CAF50')
        lims = [min(y_te_d.min(), y_pred_best.min()), max(y_te_d.max(), y_pred_best.max())]
        axes[row,1].plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        axes[row,1].set_title(f'Ridge: Predicted vs Actual — {title_prefix}', fontsize=11, fontweight='bold')
        axes[row,1].set_xlabel('Actual Value')
        axes[row,1].set_ylabel('Predicted Value')
        axes[row,1].legend(fontsize=9)
        axes[row,1].grid(True, alpha=0.3)
        r2_best = r2_score(y_te_d, y_pred_best)
        axes[row,1].text(0.05, 0.92, f'R² = {r2_best:.4f}', transform=axes[row,1].transAxes,
                        fontsize=11, fontweight='bold', color='#4CAF50',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Model Performance Dashboard — Linear Regression Engine\nBoth Datasets: Silicon Fmax & Artisan Cheese Fermentation',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig4_model_comparison.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: fig4_model_comparison.png")


# ─────────────────────────────────────────────
# FIGURE 5: Gradient Descent Convergence
# ─────────────────────────────────────────────
def fig5_gd_convergence():
    print("Generating Fig 5: Gradient Descent Convergence...")
    X_2f = X_tr_p[:, :2]
    y_arr = y_tr.values
    ols_2f = LinearRegression(fit_intercept=True).fit(X_2f, y_arr)

    def run_gd(lr, n_iter=500):
        w = np.zeros(2)
        b = 0.0
        costs = []
        for _ in range(n_iter):
            y_hat = X_2f @ w + b
            error = y_hat - y_arr
            dw = 2 * X_2f.T @ error / len(y_arr)
            db = 2 * np.sum(error) / len(y_arr)
            w -= lr * dw
            b -= lr * db
            costs.append(np.mean(error**2))
        return costs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lrs = [0.001, 0.01, 0.05, 0.1]
    colors_lr = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
    for lr, color in zip(lrs, colors_lr):
        costs = run_gd(lr)
        axes[0].plot(costs, color=color, linewidth=2, label=f'lr = {lr}')
    axes[0].set_yscale('log')
    axes[0].set_title('Effect of Learning Rate on GD Convergence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('MSE Cost (log scale)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Best convergence
    best_costs = run_gd(0.05, n_iter=500)
    axes[1].plot(best_costs, color='#4CAF50', linewidth=2.5)
    axes[1].axhline(best_costs[-1], color='red', linestyle='--', linewidth=1.5, label=f'Final MSE: {best_costs[-1]:.1f}')
    axes[1].set_title('Gradient Descent Convergence (lr=0.05)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('MSE Cost')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Gradient Descent: Learning Rate Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig5_gd_convergence.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: fig5_gd_convergence.png")


if __name__ == "__main__":
    fig1_cost_surface()
    fig2_l1_l2_contours()
    fig3_coefficient_paths()
    fig4_model_comparison()
    fig5_gd_convergence()
    print("\nAll visualizations generated successfully.")
    print(f"Saved to: {ASSETS_DIR}")
