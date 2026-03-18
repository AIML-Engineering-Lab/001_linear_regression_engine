"""
Data Generator for Project 001: Linear Regression Engine
Generates two synthetic datasets:
  1. Artisan Cheese Fermentation Time (General/Universal)
  2. Silicon Fmax Prediction (Post-Silicon Validation)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 2000
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_cheese_dataset(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """
    Generates a synthetic dataset for predicting optimal artisan cheese
    fermentation time based on biochemical and environmental factors.
    """
    rng = np.random.default_rng(seed)

    milk_fat = rng.uniform(2.5, 9.0, n)               # %
    starter_ph = rng.uniform(4.2, 6.8, n)             # pH
    ambient_temp = rng.uniform(10.0, 28.0, n)         # Celsius
    humidity = rng.uniform(60.0, 95.0, n)             # %
    salt_conc = rng.uniform(0.5, 3.5, n)              # % w/w
    curd_cut_size = rng.uniform(0.3, 2.5, n)          # cm
    airflow = rng.uniform(0.1, 1.2, n)                # m/s
    strain_type = rng.choice([0, 1, 2, 3], n)        # encoded: 0=Lactobacillus, 1=Streptococcus, 2=Mesophilic, 3=Thermophilic

    # Realistic relationships:
    # Higher fat -> longer fermentation
    # Higher pH -> longer fermentation (less acidic starter)
    # Higher temp -> shorter fermentation (faster bacterial activity)
    # Higher humidity -> slightly shorter
    # Higher salt -> longer (inhibits bacteria)
    # Larger curd -> longer (less surface area)
    # Higher airflow -> slightly shorter (better gas exchange)
    # Strain type affects base fermentation time
    strain_effect = np.array([0.0, -2.0, 1.5, -3.0])[strain_type]

    fermentation_time = (
        20.0
        + 3.5 * milk_fat
        + 4.0 * starter_ph
        - 1.2 * ambient_temp
        - 0.15 * humidity
        + 5.0 * salt_conc
        + 3.0 * curd_cut_size
        - 2.5 * airflow
        + strain_effect
        + rng.normal(0, 2.5, n)  # noise
    )
    fermentation_time = np.clip(fermentation_time, 8.0, 120.0)

    df = pd.DataFrame({
        "milk_fat_percentage": np.round(milk_fat, 2),
        "starter_culture_ph": np.round(starter_ph, 2),
        "ambient_temperature": np.round(ambient_temp, 1),
        "fermentation_humidity": np.round(humidity, 1),
        "salt_concentration": np.round(salt_conc, 2),
        "curd_cut_size": np.round(curd_cut_size, 2),
        "aging_room_airflow": np.round(airflow, 2),
        "bacterial_strain_type": strain_type,
        "optimal_fermentation_time": np.round(fermentation_time, 1),
    })
    return df


def generate_fmax_dataset(n: int = N, seed: int = SEED + 1) -> pd.DataFrame:
    """
    Generates a synthetic dataset for predicting maximum stable clock frequency
    (Fmax in MHz) based on voltage, temperature, and silicon process parameters.
    This represents a post-silicon validation use case.
    """
    rng = np.random.default_rng(seed)

    vdd_core = rng.uniform(0.70, 1.10, n)             # Volts
    junction_temp = rng.uniform(25.0, 125.0, n)       # Celsius
    leakage_current = rng.uniform(5.0, 80.0, n)       # mA (proxy for process speed)
    ring_osc_speed = rng.uniform(800.0, 1400.0, n)    # MHz (on-die process monitor)
    thermal_resistance = rng.uniform(10.0, 35.0, n)   # deg C/W
    ir_drop = rng.uniform(5.0, 60.0, n)               # mV (voltage droop)
    lot_id = rng.choice([0, 1, 2, 3, 4], n)           # Silicon lot (process corner proxy)

    # Realistic relationships:
    # Higher VDD -> higher Fmax (more drive strength)
    # Higher temp -> lower Fmax (slower transistors)
    # Higher leakage -> higher Fmax (fast process corner)
    # Higher ring oscillator speed -> higher Fmax
    # Higher thermal resistance -> lower Fmax (thermal throttling risk)
    # Higher IR drop -> lower Fmax (effective VDD is lower)
    # Lot ID encodes process corner variation
    lot_effect = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])[lot_id]

    fmax = (
        -500.0
        + 1800.0 * vdd_core
        - 2.5 * junction_temp
        + 1.2 * leakage_current
        + 0.6 * ring_osc_speed
        - 3.0 * thermal_resistance
        - 1.5 * ir_drop
        + lot_effect
        + rng.normal(0, 15.0, n)  # measurement noise
    )
    fmax = np.clip(fmax, 200.0, 2500.0)

    df = pd.DataFrame({
        "vdd_core": np.round(vdd_core, 3),
        "junction_temp": np.round(junction_temp, 1),
        "leakage_current": np.round(leakage_current, 2),
        "ring_oscillator_speed": np.round(ring_osc_speed, 1),
        "thermal_resistance": np.round(thermal_resistance, 2),
        "ir_drop_estimate": np.round(ir_drop, 2),
        "silicon_lot_id": lot_id,
        "fmax_mhz": np.round(fmax, 1),
    })
    return df


if __name__ == "__main__":
    cheese_df = generate_cheese_dataset()
    cheese_path = DATA_DIR / "artisan_cheese_fermentation_data.csv"
    cheese_df.to_csv(cheese_path, index=False)
    print(f"Cheese dataset saved: {cheese_path} ({len(cheese_df)} rows)")

    fmax_df = generate_fmax_dataset()
    fmax_path = DATA_DIR / "silicon_fmax_validation_data.csv"
    fmax_df.to_csv(fmax_path, index=False)
    print(f"Fmax dataset saved:   {fmax_path} ({len(fmax_df)} rows)")
