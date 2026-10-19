"""
Machine Learning Module for Climate Risk Classification.
Trains and compares Logistic Regression, Random Forest, and Gradient Boosting models
to predict 'High_Climate_Risk'. Evaluates models with Accuracy, Precision, Recall,
F1-Score, ROC-AUC, 5-Fold Stratified Cross-Validation, Confusion Matrices, and Feature Importance.
"""

import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


FEATURE_COLS = [
    "Year",
    "CO2_Emissions_MT",
    "CO2_per_Capita",
    "Cumulative_CO2_MT",
    "Sea_Level_Rise_mm",
    "Precipitation_mm",
    "Humidity_pct",
    "Population",
    "Country_Baseline_Temp",
    "Temp_Change_YoY"
]
TARGET_COL = "High_Climate_Risk"


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepares feature matrix X and target y, and performs stratified train/test split."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_models():
    """Returns dictionary of baseline and ensemble models."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=4, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
        )
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains all models, computes test metrics and 5-fold cross-validation scores.
    Returns metrics DataFrame and trained models dict.
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    trained_models = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Fit on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predictions on holdout test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "CV_Accuracy_Mean": round(cv_scores.mean(), 4),
            "CV_Accuracy_Std": round(cv_scores.std(), 4),
            "Test_Accuracy": round(acc, 4),
            "Test_Precision": round(prec, 4),
            "Test_Recall": round(rec, 4),
            "Test_F1_Score": round(f1, 4),
            "Test_ROC_AUC": round(auc, 4)
        })

    metrics_df = pd.DataFrame(results)
    return metrics_df, trained_models, predictions, probabilities


def plot_model_evaluations(y_test, probabilities, predictions, output_dir: str = "outputs/figures"):
    """Plots ROC curves and confusion matrices for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=150)
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2.2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    ax_roc.set_title("ROC Curves Comparison: Climate Risk Classifiers", fontsize=13, fontweight="bold", pad=12)
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    fig_roc.savefig(os.path.join(output_dir, "09_roc_curves.png"))

    # 2. Confusion Matrices (3 side-by-side)
    fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "High Risk"], yticklabels=["Normal", "High Risk"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("Actual Label", fontsize=9)

    fig_cm.suptitle("Confusion Matrices on Test Set", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_cm.savefig(os.path.join(output_dir, "10_confusion_matrices.png"))

    return fig_roc, fig_cm


def plot_feature_importance(trained_models, output_dir: str = "outputs/figures"):
    """Plots feature importances for tree-based ensemble models."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
        model = trained_models[name]
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        colors = sns.color_palette("viridis", n_colors=len(importances))
        importances.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"{name} Feature Importance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Relative Importance", fontsize=9)

    fig.suptitle("Feature Importance in Tree-Based Climate Risk Predictions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "11_feature_importance.png"))
    return fig


def save_models(trained_models, metrics_df, output_dir: str = "outputs/models", training_records: int = 2220):
    """Serializes trained classification models and saves metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib"
    }
    for name, model in trained_models.items():
        fname = filename_map.get(name)
        if fname:
            fpath = os.path.join(output_dir, fname)
            joblib.dump(model, fpath)
            print(f"Saved {name} to {fpath}")

    # Save model metadata
    metadata = {
        "version": "1.0.0",
        "training_records": training_records,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": metrics_df.to_dict(orient="records")
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {meta_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/climate_data_cleaned.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)
    metrics_df, trained_models, predictions, probabilities = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    print("\n--- Model Evaluation Summary ---")
    print(metrics_df.to_string(index=False))
    plot_model_evaluations(y_test, probabilities, predictions)
    plot_feature_importance(trained_models)
    save_models(trained_models, metrics_df, training_records=len(df))
