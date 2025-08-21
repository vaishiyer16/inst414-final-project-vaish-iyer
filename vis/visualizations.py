import os
import logging
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_plot(outpath: str):
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def generate_visuals(processed_data_path: str,
                     outputs_dir: str = "data/outputs",
                     logger: logging.Logger | None = None) -> None:
    """
    Generate simple EDA/model-friendly visuals:
      - Class balance bar
      - Example: fraud rate by incident_severity (if present)
      - Numeric correlation heatmap (if enough numeric columns)
    """
    if logger: logger.info("[Vis] Generating visuals")
    _ensure_dir(outputs_dir)

    df = pd.read_csv(processed_data_path)

    # 1) Class balance
    if "fraud_reported" in df.columns:
        counts = df["fraud_reported"].value_counts().sort_index()
        counts.index = counts.index.map({0: "Not Fraud", 1: "Fraud"})
        counts.plot(kind="bar", title="Class Balance: Fraud vs Not Fraud")
        _save_plot(os.path.join(outputs_dir, "class_balance.png"))

    # 2) Fraud rate by incident_severity (if present)
    if "incident_severity" in df.columns and "fraud_reported" in df.columns:
        rate = df.groupby("incident_severity")["fraud_reported"].mean().sort_values(ascending=False)
        rate.plot(kind="bar", title="Fraud Rate by Incident Severity")
        _save_plot(os.path.join(outputs_dir, "fraud_rate_by_incident_severity.png"))

    # 3) Correlation heatmap for numeric columns
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, interpolation="nearest")
        ax.set_title("Correlation Heatmap (Numeric)")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_plot(os.path.join(outputs_dir, "correlation_heatmap.png"))

    if logger: logger.info(f"[Vis] Visualizations saved to {outputs_dir}")
