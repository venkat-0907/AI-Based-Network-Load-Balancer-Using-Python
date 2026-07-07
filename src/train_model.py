"""
train_model.py
---------------
Trains two AI models used by the load balancer:

1. LOAD CLASSIFIER (RandomForestClassifier)
   Predicts current server load bucket: Low / Medium / High
   Used to instantly flag an overloaded server.

2. RESPONSE-TIME REGRESSOR (GradientBoostingRegressor)
   Predicts expected response time (ms) for a server given its live
   telemetry. The load balancer routes each new request to the server
   with the LOWEST predicted response time -> this is the actual
   "smart routing" decision engine.

Produces:
  models/load_classifier.joblib
  models/response_time_regressor.joblib
  models/feature_scaler.joblib
  outputs/graphs/*.png   (confusion matrix, ROC curves, feature importance,
                          regression fit, residuals, class distribution,
                          correlation heatmap)
  outputs/logs/training_report.txt   (full metrics report)
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

sns.set_theme(style="whitegrid")

DATA_PATH = "data/network_load_dataset.csv"
MODEL_DIR = "models"
GRAPH_DIR = "outputs/graphs"
LOG_DIR = "outputs/logs"

FEATURE_COLS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "active_connections",
    "requests_per_sec",
    "network_latency_ms",
    "disk_io_mb_s",
    "bandwidth_mbps",
    "error_rate_pct",
]
CLASS_TARGET = "load_status"
REG_TARGET = "response_time_ms"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def train_classifier(df, report_lines):
    X = df[FEATURE_COLS].values
    y = df[CLASS_TARGET].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Multiclass ROC-AUC (one-vs-rest)
    lb = LabelBinarizer().fit(y_train)
    y_test_bin = lb.transform(y_test)
    try:
        auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    report_lines.append("=" * 70)
    report_lines.append("MODEL 1: LOAD CLASSIFIER (RandomForestClassifier)")
    report_lines.append("=" * 70)
    report_lines.append(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    report_lines.append(f"Training time: {train_time:.2f}s")
    report_lines.append(f"Accuracy       : {acc:.4f}")
    report_lines.append(f"Precision (wtd): {prec:.4f}")
    report_lines.append(f"Recall (wtd)   : {rec:.4f}")
    report_lines.append(f"F1-score (wtd) : {f1:.4f}")
    report_lines.append(f"ROC-AUC (OvR)  : {auc:.4f}")
    report_lines.append("\nPer-class classification report:")
    report_lines.append(classification_report(y_test, y_pred, target_names=le.classes_))

    # ---- Graph 1: Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Load Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- Graph 2: Feature Importance ----
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=importances[order], y=np.array(FEATURE_COLS)[order], ax=ax, palette="viridis")
    ax.set_title("Feature Importance - Load Classifier")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/feature_importance_classifier.png", dpi=150)
    plt.close(fig)

    # ---- Graph 3: ROC Curves (one-vs-rest, per class) ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, cls_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        class_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls_name} (AUC={class_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Load Classifier (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/roc_curves.png", dpi=150)
    plt.close(fig)

    # ---- Graph 4: Class distribution ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.countplot(x=CLASS_TARGET, data=df, order=["Low", "Medium", "High"],
                   palette={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}, ax=ax)
    ax.set_title("Dataset Class Distribution (Load Status)")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/class_distribution.png", dpi=150)
    plt.close(fig)

    joblib.dump(clf, f"{MODEL_DIR}/load_classifier.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/feature_scaler.joblib")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    return metrics


def train_regressor(df, report_lines):
    X = df[FEATURE_COLS].values
    y = df[REG_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42
    )
    t0 = time.time()
    reg.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    report_lines.append("=" * 70)
    report_lines.append("MODEL 2: RESPONSE-TIME REGRESSOR (GradientBoostingRegressor)")
    report_lines.append("=" * 70)
    report_lines.append(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    report_lines.append(f"Training time: {train_time:.2f}s")
    report_lines.append(f"MAE  : {mae:.3f} ms")
    report_lines.append(f"RMSE : {rmse:.3f} ms")
    report_lines.append(f"MAPE : {mape:.2f}%")
    report_lines.append(f"R^2  : {r2:.4f}")

    # ---- Graph 5: Predicted vs Actual ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.25, s=12, color="#3F51B5")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal fit (y = x)")
    ax.set_xlabel("Actual Response Time (ms)")
    ax.set_ylabel("Predicted Response Time (ms)")
    ax.set_title(f"Response-Time Regressor: Predicted vs Actual (R²={r2:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/regression_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # ---- Graph 6: Residuals ----
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(residuals, kde=True, ax=ax, color="#009688")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Residual Distribution - Response-Time Regressor")
    ax.set_xlabel("Residual (Actual - Predicted) ms")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/regression_residuals.png", dpi=150)
    plt.close(fig)

    # ---- Graph 7: Feature importance for regressor ----
    importances = reg.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=importances[order], y=np.array(FEATURE_COLS)[order], ax=ax, palette="magma")
    ax.set_title("Feature Importance - Response-Time Regressor")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/feature_importance_regressor.png", dpi=150)
    plt.close(fig)

    joblib.dump(reg, f"{MODEL_DIR}/response_time_regressor.joblib")

    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
    return metrics


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[FEATURE_COLS + [REG_TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    report_lines = [
        "AI LOAD BALANCER - MODEL TRAINING REPORT",
        f"Dataset: {DATA_PATH}  ({len(df):,} rows)",
        "",
    ]

    clf_metrics = train_classifier(df, report_lines)
    reg_metrics = train_regressor(df, report_lines)
    correlation_heatmap(df)

    report_lines.append("=" * 70)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append(json.dumps({"classifier": clf_metrics, "regressor": reg_metrics}, indent=2))

    report_text = "\n".join(report_lines)
    with open(f"{LOG_DIR}/training_report.txt", "w") as f:
        f.write(report_text)

    print(report_text)

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({"classifier": clf_metrics, "regressor": reg_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
