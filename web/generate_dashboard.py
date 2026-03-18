"""
Production Fmax Validation Dashboard Generator
===============================================
Generates synthetic silicon lot data, runs Fmax predictions through the trained
Ridge regression pipeline, and produces an interactive HTML dashboard with
real post-silicon validation analytics.

Usage:
    python web/generate_dashboard.py
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "ridge_pipeline.pkl"
OUTPUT_HTML = ROOT / "web" / "dashboard.html"
DATA_OUT = ROOT / "web" / "dashboard_data.csv"

# ── Synthetic Lot Generation ────────────────────────────────────────────────
FMAX_TARGET = 1450.0  # MHz — product spec target
FMAX_GUARD  = 1250.0  # MHz — guardband for binning


def generate_production_lots(n_lots=50, wafers_per_lot=20, dies_per_wafer=8, seed=2024):
    """
    Generate realistic post-silicon validation data.
    Each lot comes from a fab run, with wafer-level and die-level variation.
    """
    rng = np.random.default_rng(seed)
    rows = []

    fab_start = datetime(2024, 6, 1)
    lot_names = [f"LOT-{2400 + i:04d}" for i in range(n_lots)]

    for lot_idx, lot_name in enumerate(lot_names):
        # Lot-level process variation (slow/typical/fast corner)
        lot_vdd_offset = rng.normal(0, 0.05)
        lot_leak_offset = rng.normal(0, 10.0)
        lot_date = fab_start + timedelta(days=int(lot_idx * 2 + rng.integers(0, 3)))

        for wafer_idx in range(wafers_per_lot):
            wafer_id = f"W{wafer_idx + 1:02d}"
            # Wafer-level variation (center-to-edge gradient)
            wafer_temp_offset = rng.normal(0, 5.0)

            for die_idx in range(dies_per_wafer):
                die_id = f"D{die_idx + 1:03d}"

                # Core silicon parameters with realistic variation
                vdd_core = np.clip(rng.normal(0.88 + lot_vdd_offset, 0.06), 0.70, 1.10)
                junction_temp = np.clip(rng.normal(80.0 + wafer_temp_offset, 20.0), 25, 125)
                leakage_current = np.clip(rng.normal(38.0 + lot_leak_offset, 15.0), 5, 80)
                ring_osc_speed = np.clip(rng.normal(1050.0, 120.0), 800, 1400)
                thermal_resistance = np.clip(rng.normal(24.0, 5.0), 10, 35)
                ir_drop = np.clip(rng.normal(35.0, 12.0), 5, 60)
                silicon_lot_id = lot_idx % 5  # Process corner proxy

                rows.append({
                    "lot_name": lot_name,
                    "lot_date": lot_date.strftime("%Y-%m-%d"),
                    "wafer_id": wafer_id,
                    "die_id": die_id,
                    "vdd_core": round(vdd_core, 3),
                    "junction_temp": round(junction_temp, 1),
                    "leakage_current": round(leakage_current, 2),
                    "ring_oscillator_speed": round(ring_osc_speed, 1),
                    "thermal_resistance": round(thermal_resistance, 2),
                    "ir_drop_estimate": round(ir_drop, 2),
                    "silicon_lot_id": silicon_lot_id,
                })

    return pd.DataFrame(rows)


def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Load trained Ridge pipeline and predict Fmax for all dies."""
    pipe = joblib.load(MODEL_PATH)
    feature_cols = ["vdd_core", "junction_temp", "leakage_current",
                    "ring_oscillator_speed", "thermal_resistance",
                    "ir_drop_estimate", "silicon_lot_id"]
    X = df[feature_cols]
    df = df.copy()
    df["fmax_predicted"] = pipe.predict(X)
    df["pass_fail"] = np.where(df["fmax_predicted"] >= FMAX_GUARD, "PASS", "FAIL")
    df["bin"] = pd.cut(
        df["fmax_predicted"],
        bins=[0, 1100, 1250, 1350, 1450, 1600, 3000],
        labels=["BIN5-Reject", "BIN4-Low", "BIN3-Below", "BIN2-Guardband", "BIN1-Target", "BIN0-Premium"]
    )
    return df


def generate_html_dashboard(df: pd.DataFrame) -> str:
    """Generate a self-contained interactive HTML dashboard."""

    # ── Aggregate metrics ──
    total_dies = len(df)
    total_lots = df["lot_name"].nunique()
    pass_count = (df["pass_fail"] == "PASS").sum()
    fail_count = (df["pass_fail"] == "FAIL").sum()
    overall_yield = pass_count / total_dies * 100
    mean_fmax = df["fmax_predicted"].mean()
    std_fmax = df["fmax_predicted"].std()
    min_fmax = df["fmax_predicted"].min()
    max_fmax = df["fmax_predicted"].max()

    # Lot-level yield
    lot_yield = df.groupby("lot_name").apply(
        lambda g: (g["pass_fail"] == "PASS").sum() / len(g) * 100
    ).reset_index()
    lot_yield.columns = ["lot_name", "yield_pct"]
    lot_yield = lot_yield.sort_values("lot_name")

    # Lot-level mean Fmax
    lot_fmax = df.groupby("lot_name")["fmax_predicted"].mean().reset_index()
    lot_fmax.columns = ["lot_name", "mean_fmax"]
    lot_fmax = lot_fmax.sort_values("lot_name")

    # Binning distribution
    bin_counts = df["bin"].value_counts().sort_index()
    bin_labels = bin_counts.index.tolist()
    bin_values = bin_counts.values.tolist()
    bin_colors = ["#f44336", "#ff9800", "#ffc107", "#8bc34a", "#4caf50", "#2196f3"]

    # VDD vs Fmax scatter (sample for performance)
    sample = df.sample(min(2000, len(df)), random_state=42)

    # Temperature vs Fmax shmoo
    temp_bins = pd.cut(df["junction_temp"], bins=10)
    vdd_bins = pd.cut(df["vdd_core"], bins=10)
    shmoo = df.groupby([temp_bins, vdd_bins])["fmax_predicted"].mean().unstack()

    # Wafer map for first lot (yield by wafer)
    first_lot = df[df["lot_name"] == df["lot_name"].iloc[0]]
    wafer_yield = first_lot.groupby("wafer_id").apply(
        lambda g: (g["pass_fail"] == "PASS").sum() / len(g) * 100
    ).reset_index()
    wafer_yield.columns = ["wafer_id", "yield_pct"]

    # Lot date trend
    lot_date_stats = df.groupby(["lot_name", "lot_date"]).agg(
        mean_fmax=("fmax_predicted", "mean"),
        yield_pct=("pass_fail", lambda x: (x == "PASS").sum() / len(x) * 100)
    ).reset_index()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fmax Validation Dashboard — Post-Silicon Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; }}
.header {{ background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); padding: 24px 40px; border-bottom: 1px solid #30363d; }}
.header h1 {{ font-size: 22px; color: #58a6ff; font-weight: 600; }}
.header p {{ color: #8b949e; font-size: 13px; margin-top: 4px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; padding: 20px 40px; }}
.kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
.kpi .value {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
.kpi .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
.kpi.pass .value {{ color: #3fb950; }}
.kpi.fail .value {{ color: #f85149; }}
.kpi.yield .value {{ color: {('#3fb950' if overall_yield >= 90 else '#f0883e' if overall_yield >= 75 else '#f85149')}; }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px 40px; }}
.chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; min-height: 380px; }}
.chart-box.full {{ grid-column: 1 / -1; }}
.chart-title {{ font-size: 14px; font-weight: 600; color: #c9d1d9; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
.footer {{ padding: 16px 40px; text-align: center; color: #484f58; font-size: 11px; border-top: 1px solid #21262d; margin-top: 20px; }}
.status-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
.status-pass {{ background: rgba(63, 185, 80, 0.15); color: #3fb950; }}
.status-fail {{ background: rgba(248, 81, 73, 0.15); color: #f85149; }}
</style>
</head>
<body>

<div class="header">
    <h1>Silicon Fmax Validation Dashboard</h1>
    <p>Post-Silicon Characterization &amp; Yield Analytics &mdash; Ridge Regression Model &mdash; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

<div class="kpi-row">
    <div class="kpi">
        <div class="value">{total_lots}</div>
        <div class="label">Total Lots</div>
    </div>
    <div class="kpi">
        <div class="value">{total_dies:,}</div>
        <div class="label">Total Dies Tested</div>
    </div>
    <div class="kpi yield">
        <div class="value">{overall_yield:.1f}%</div>
        <div class="label">Overall Yield</div>
    </div>
    <div class="kpi pass">
        <div class="value">{pass_count:,}</div>
        <div class="label">PASS (≥{FMAX_GUARD:.0f} MHz)</div>
    </div>
    <div class="kpi fail">
        <div class="value">{fail_count:,}</div>
        <div class="label">FAIL (&lt;{FMAX_GUARD:.0f} MHz)</div>
    </div>
    <div class="kpi">
        <div class="value">{mean_fmax:.0f}</div>
        <div class="label">Mean Fmax (MHz)</div>
    </div>
</div>

<div class="charts">

    <div class="chart-box">
        <div class="chart-title">Fmax Distribution — All Dies</div>
        <div id="fmax-hist"></div>
    </div>

    <div class="chart-box">
        <div class="chart-title">Bin Classification</div>
        <div id="bin-pie"></div>
    </div>

    <div class="chart-box full">
        <div class="chart-title">Lot-Level Yield Trend</div>
        <div id="lot-yield"></div>
    </div>

    <div class="chart-box">
        <div class="chart-title">VDD Core vs Predicted Fmax</div>
        <div id="vdd-scatter"></div>
    </div>

    <div class="chart-box">
        <div class="chart-title">Junction Temperature vs Predicted Fmax</div>
        <div id="temp-scatter"></div>
    </div>

    <div class="chart-box full">
        <div class="chart-title">Lot-Level Mean Fmax</div>
        <div id="lot-fmax-bar"></div>
    </div>

    <div class="chart-box">
        <div class="chart-title">Wafer Yield Map — {first_lot["lot_name"].iloc[0]}</div>
        <div id="wafer-map"></div>
    </div>

    <div class="chart-box">
        <div class="chart-title">Feature Correlation with Fmax</div>
        <div id="feature-corr"></div>
    </div>

</div>

<div class="footer">
    Linear Regression Engine — Post-Silicon Fmax Validation Dashboard &mdash; AIML Engineering Lab &mdash; Model: Ridge Regression (scikit-learn)
</div>

<script>
const plotBg = '#161b22';
const paperBg = '#161b22';
const gridColor = '#21262d';
const textColor = '#c9d1d9';
const layout_base = {{
    paper_bgcolor: paperBg, plot_bgcolor: plotBg, font: {{ color: textColor, size: 11 }},
    margin: {{ l: 50, r: 20, t: 10, b: 40 }},
    xaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }},
    yaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }}
}};

// 1. Fmax Histogram
Plotly.newPlot('fmax-hist', [{{
    x: {json.dumps(df["fmax_predicted"].round(1).tolist())},
    type: 'histogram', nbinsx: 60,
    marker: {{ color: '#58a6ff', line: {{ color: '#0d1117', width: 0.5 }} }},
    opacity: 0.85
}}, {{
    x: [{FMAX_TARGET}, {FMAX_TARGET}], y: [0, {total_dies // 10}], mode: 'lines',
    line: {{ color: '#3fb950', width: 2, dash: 'dash' }}, name: 'Target ({FMAX_TARGET:.0f} MHz)'
}}, {{
    x: [{FMAX_GUARD}, {FMAX_GUARD}], y: [0, {total_dies // 10}], mode: 'lines',
    line: {{ color: '#f0883e', width: 2, dash: 'dash' }}, name: 'Guardband ({FMAX_GUARD:.0f} MHz)'
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'Predicted Fmax (MHz)' }},
    yaxis: {{ ...layout_base.yaxis, title: 'Count' }},
    showlegend: true, legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0)' }},
    bargap: 0.02
}}, {{ responsive: true }});

// 2. Bin Pie Chart
Plotly.newPlot('bin-pie', [{{
    labels: {json.dumps(bin_labels)},
    values: {json.dumps(bin_values)},
    type: 'pie', hole: 0.45,
    marker: {{ colors: {json.dumps(bin_colors[:len(bin_labels)])} }},
    textinfo: 'label+percent', textfont: {{ size: 10 }},
    hovertemplate: '%{{label}}: %{{value}} dies (%{{percent}})<extra></extra>'
}}], {{
    paper_bgcolor: paperBg, font: {{ color: textColor, size: 11 }},
    margin: {{ l: 10, r: 10, t: 10, b: 10 }},
    showlegend: false
}}, {{ responsive: true }});

// 3. Lot Yield Trend
Plotly.newPlot('lot-yield', [{{
    x: {json.dumps(lot_yield["lot_name"].tolist())},
    y: {json.dumps(lot_yield["yield_pct"].round(1).tolist())},
    type: 'bar',
    marker: {{ color: {json.dumps(lot_yield["yield_pct"].apply(lambda v: '#3fb950' if v >= 90 else '#f0883e' if v >= 75 else '#f85149').tolist())} }}
}}, {{
    x: {json.dumps(lot_yield["lot_name"].tolist())},
    y: {json.dumps([90.0] * total_lots)},
    mode: 'lines', line: {{ color: '#f0883e', dash: 'dash', width: 1.5 }}, name: '90% Target'
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'Lot', tickangle: -45, tickfont: {{ size: 8 }} }},
    yaxis: {{ ...layout_base.yaxis, title: 'Yield %', range: [0, 105] }},
    showlegend: true, legend: {{ x: 0.85, y: 1.0, bgcolor: 'rgba(0,0,0,0)' }},
    bargap: 0.15, height: 300
}}, {{ responsive: true }});

// 4. VDD vs Fmax scatter
Plotly.newPlot('vdd-scatter', [{{
    x: {json.dumps(sample["vdd_core"].tolist())},
    y: {json.dumps(sample["fmax_predicted"].round(1).tolist())},
    mode: 'markers', type: 'scatter',
    marker: {{ size: 3, color: {json.dumps(sample["fmax_predicted"].round(1).tolist())}, colorscale: 'Viridis', opacity: 0.6 }}
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'VDD Core (V)' }},
    yaxis: {{ ...layout_base.yaxis, title: 'Predicted Fmax (MHz)' }}
}}, {{ responsive: true }});

// 5. Temp vs Fmax scatter
Plotly.newPlot('temp-scatter', [{{
    x: {json.dumps(sample["junction_temp"].tolist())},
    y: {json.dumps(sample["fmax_predicted"].round(1).tolist())},
    mode: 'markers', type: 'scatter',
    marker: {{ size: 3, color: {json.dumps(sample["fmax_predicted"].round(1).tolist())}, colorscale: 'RdYlGn', opacity: 0.6 }}
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'Junction Temperature (°C)' }},
    yaxis: {{ ...layout_base.yaxis, title: 'Predicted Fmax (MHz)' }}
}}, {{ responsive: true }});

// 6. Lot Mean Fmax bar
Plotly.newPlot('lot-fmax-bar', [{{
    x: {json.dumps(lot_fmax["lot_name"].tolist())},
    y: {json.dumps(lot_fmax["mean_fmax"].round(1).tolist())},
    type: 'bar',
    marker: {{ color: '#58a6ff' }}
}}, {{
    x: {json.dumps(lot_fmax["lot_name"].tolist())},
    y: {json.dumps([FMAX_TARGET] * total_lots)},
    mode: 'lines', line: {{ color: '#3fb950', dash: 'dash', width: 1.5 }}, name: 'Target'
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'Lot', tickangle: -45, tickfont: {{ size: 8 }} }},
    yaxis: {{ ...layout_base.yaxis, title: 'Mean Fmax (MHz)' }},
    showlegend: true, legend: {{ x: 0.85, y: 1.0, bgcolor: 'rgba(0,0,0,0)' }},
    bargap: 0.15, height: 300
}}, {{ responsive: true }});

// 7. Wafer yield map
Plotly.newPlot('wafer-map', [{{
    x: {json.dumps(wafer_yield["wafer_id"].tolist())},
    y: {json.dumps(wafer_yield["yield_pct"].round(1).tolist())},
    type: 'bar',
    marker: {{ color: {json.dumps(wafer_yield["yield_pct"].apply(lambda v: '#3fb950' if v >= 90 else '#f0883e' if v >= 75 else '#f85149').tolist())} }}
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, title: 'Wafer' }},
    yaxis: {{ ...layout_base.yaxis, title: 'Yield %', range: [0, 105] }},
    bargap: 0.2
}}, {{ responsive: true }});

// 8. Feature correlation
const features = ['vdd_core', 'junction_temp', 'leakage_current', 'ring_oscillator_speed', 'thermal_resistance', 'ir_drop_estimate'];
const corrs = {json.dumps([round(df[f].corr(df["fmax_predicted"]), 3) for f in ["vdd_core", "junction_temp", "leakage_current", "ring_oscillator_speed", "thermal_resistance", "ir_drop_estimate"]])};
Plotly.newPlot('feature-corr', [{{
    x: features, y: corrs, type: 'bar',
    marker: {{ color: corrs.map(v => v > 0 ? '#3fb950' : '#f85149') }}
}}], {{
    ...layout_base,
    xaxis: {{ ...layout_base.xaxis, tickangle: -30 }},
    yaxis: {{ ...layout_base.yaxis, title: 'Correlation with Fmax', range: [-1, 1] }},
    bargap: 0.3
}}, {{ responsive: true }});
</script>

</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("Fmax Production Dashboard Generator")
    print("=" * 60)

    print("\n[1/4] Generating synthetic lot data...")
    df = generate_production_lots(n_lots=50, wafers_per_lot=20, dies_per_wafer=8)
    print(f"      Generated {len(df):,} die records across {df['lot_name'].nunique()} lots")

    print("[2/4] Running Fmax predictions through Ridge pipeline...")
    df = run_predictions(df)
    pass_pct = (df["pass_fail"] == "PASS").sum() / len(df) * 100
    print(f"      Mean Fmax: {df['fmax_predicted'].mean():.1f} MHz | Yield: {pass_pct:.1f}%")

    print("[3/4] Saving prediction data...")
    df.to_csv(DATA_OUT, index=False)
    print(f"      Saved to {DATA_OUT}")

    print("[4/4] Generating interactive HTML dashboard...")
    html = generate_html_dashboard(df)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"      Dashboard saved to {OUTPUT_HTML}")
    print(f"      Open in browser: file://{OUTPUT_HTML}")
    print("\n✅ Dashboard generation complete!")


if __name__ == "__main__":
    main()
