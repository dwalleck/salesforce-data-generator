"""Model evaluation utilities for churn prediction.

Provides metrics, visualization, and reporting for ML models.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    model_name: str = "Model",
) -> dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC/PR curves)
        model_name: Name of the model for reporting

    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC and PR-AUC require probabilities
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))

            # Calculate Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["pr_auc"] = float(auc(recall, precision))
        except ValueError as e:
            # Handle edge cases (e.g., only one class present)
            print(f"Warning: Could not calculate ROC/PR AUC for {model_name}: {e}")
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # Classification report (as dict for JSON serialization)
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path | str | None = None
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_path: Path to save plot (if None, displays only)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved confusion matrix: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba_dict: dict[str, np.ndarray],
    output_path: Path | str | None = None,
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true: Ground truth labels
        y_pred_proba_dict: Dict mapping model names to predicted probabilities
        output_path: Path to save plot (if None, displays only)
    """
    plt.figure(figsize=(10, 8))

    for model_name, y_pred_proba in y_pred_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved ROC curve: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba_dict: dict[str, np.ndarray],
    output_path: Path | str | None = None,
) -> None:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        y_true: Ground truth labels
        y_pred_proba_dict: Dict mapping model names to predicted probabilities
        output_path: Path to save plot (if None, displays only)
    """
    plt.figure(figsize=(10, 8))

    for model_name, y_pred_proba in y_pred_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", linewidth=2)

    # Baseline (proportion of positive class)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.3f})", linewidth=1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Model Comparison")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved PR curve: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    importance_scores: np.ndarray,
    model_name: str,
    top_n: int = 20,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot feature importance for tree-based models.

    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        model_name: Name of the model
        top_n: Number of top features to display
        output_path: Path to save plot (if None, displays only)
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_scores}).sort_values(
        "importance", ascending=False
    )

    # Select top N features
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, max(8, top_n * 0.4)))
    plt.barh(range(len(top_features)), top_features["importance"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved feature importance: {output_path}")
    else:
        plt.show()

    plt.close()


def save_metrics(metrics_dict: dict[str, dict[str, Any]], output_path: Path | str) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"  Saved metrics: {output_path}")


def print_evaluation_summary(metrics: dict[str, Any]) -> None:
    """
    Print formatted evaluation summary.

    Args:
        metrics: Evaluation metrics dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")

    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    if metrics.get("pr_auc") is not None:
        print(f"PR AUC:    {metrics['pr_auc']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN: {cm['tn']:>6}  FP: {cm['fp']:>6}")
    print(f"  FN: {cm['fn']:>6}  TP: {cm['tp']:>6}")
    print(f"{'=' * 60}\n")
