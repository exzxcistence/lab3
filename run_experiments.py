"""Повторяет эксперименты из REPORT.md и сохраняет измеренные метрики."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

ROOT = Path(__file__).resolve().parent
SEED = 42


def metric_dict(metrics):
    return {name: float(getattr(metrics, name)) for name in
            ("roc_auc", "accuracy", "precision", "recall", "f1")}


def main():
    # Student Performance: Yes/No кодируются как 1/0; цель не масштабируется.
    student = pd.read_csv(ROOT / "Student_Performance.csv")
    student["Extracurricular Activities"] = student["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    x = student.drop(columns="Performance Index").to_numpy(dtype=float)
    y = student["Performance Index"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    reg_input = RegressionInput(x_train, y_train, x_test, y_test)
    reg = train_linear_regression(reg_input)
    regression_mse = float(np.mean((y_test - reg.predictions) ** 2))

    # В german.csv метка 1 означает good credit, 2 — bad credit.
    german = pd.read_csv(ROOT / "german.csv", sep=";")
    gx = german.iloc[:, 1:].to_numpy(dtype=float)
    gy = (german.iloc[:, 0].to_numpy() == 1).astype(int)
    gx_train, gx_test, gy_train, gy_test = train_test_split(
        gx, gy, test_size=0.2, random_state=SEED, stratify=gy
    )
    clf_input = ClassificationInput(gx_train, gy_train, gx_test, gy_test)
    own = train_logistic_classifier(clf_input)
    comparisons = compare_classical_models(clf_input)
    own_metrics = {
        "roc_auc": float(roc_auc_score(gy_test, own.probabilities)),
        "accuracy": float(accuracy_score(gy_test, own.predictions)),
        "precision": float(precision_score(gy_test, own.predictions, zero_division=0)),
        "recall": float(recall_score(gy_test, own.predictions, zero_division=0)),
        "f1": float(f1_score(gy_test, own.predictions, zero_division=0)),
    }
    results = {
        "linear_regression": {"test_mse": regression_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": metric_dict(comparisons.logistic_regression),
            "decision_tree": metric_dict(comparisons.decision_tree),
            "knn": metric_dict(comparisons.knn),
        },
        "experiment": {
            "random_state": SEED,
            "test_size": 0.2,
            "regression_rows": {"train": len(y_train), "test": len(y_test)},
            "classification_rows": {"train": len(gy_train), "test": len(gy_test)},
            "regression_learning_rate": reg_input.learning_rate,
            "regression_epochs": reg_input.epochs,
            "classification_learning_rate": clf_input.learning_rate,
            "classification_epochs": clf_input.epochs,
            "regression_mse_first_epoch": reg.loss_history[0],
            "regression_mse_last_epoch": reg.loss_history[-1],
            "classification_bce_first_epoch": own.loss_history[0],
            "classification_bce_last_epoch": own.loss_history[-1],
        },
    }
    (ROOT / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
