import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _make_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    """
    Numeric: StandardScaler (helps LR convergence).
    Categorical: OneHotEncoder(handle_unknown='ignore').
    """
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target_col]]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),  # with_mean=False keeps sparse compatibility
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def _plot_confusion_matrix(y_true, y_pred, outpath: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_roc(model: Pipeline, X_test, y_test, outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def train_and_compare_models(processed_data_path: str,
                             evaluations_dir: str = "data/evaluations",
                             logger=None) -> pd.DataFrame:
    """
    Train LogisticRegression and RandomForest with preprocessing.
    Save per-model JSON metrics, a metrics CSV summary, and plots (confusion + ROC).
    Return DataFrame of metrics.
    """
    os.makedirs(evaluations_dir, exist_ok=True)
    plots_dir = os.path.join(evaluations_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if logger: logger.info(f"[Model] Loading processed data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)

    target_col = "fraud_reported"
    if target_col not in df.columns:
        raise ValueError("Target column 'fraud_reported' not found in processed data.")

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    preprocessor = _make_preprocessor(df, target_col)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),
    }

    metrics_rows: List[Dict] = []

    # Single split for fairness
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    for name, clf in models.items():
        if logger: logger.info(f"[Model] Training {name}...")
        pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score) if len(np.unique(y_test)) == 2 else float("nan"),
        }
        metrics_rows.append(row)

        if logger:
            logger.info(
                f"[Model] {name} metrics: "
                f"acc={row['accuracy']:.3f}, prec={row['precision']:.3f}, "
                f"rec={row['recall']:.3f}, f1={row['f1']:.3f}, auc={row['roc_auc']:.3f}"
            )

        # Save plots
        _plot_confusion_matrix(
            y_test, y_pred,
            outpath=os.path.join(plots_dir, f"{name.lower()}_confusion.png"),
            title=f"{name} – Confusion Matrix"
        )
        _plot_roc(
            pipeline, X_test, y_test,
            outpath=os.path.join(plots_dir, f"{name.lower()}_roc.png"),
            title=f"{name} – ROC Curve"
        )
        if logger: logger.info(f"[Model] Saved plots for {name} → {plots_dir}")

        # Save per-model metrics JSON
        report_path = os.path.join(evaluations_dir, f"{name.lower()}_report.json")
        with open(report_path, "w") as f:
            json.dump(row, f, indent=2)
        if logger: logger.info(f"[Model] {name} metrics saved → {report_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(evaluations_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    if logger: logger.info(f"[Model] Metrics summary saved → {metrics_csv}")

    return metrics_df
