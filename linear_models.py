"""Линейные модели, обучаемые пакетным градиентным спуском."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _arrays(features, target):
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be a 2D array aligned with target")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and target must contain only finite values")
    return x, y


def _scale_train_test(train_x, test_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучает линейную регрессию полным пакетным градиентным спуском."""
    x, y = _arrays(data.train_features, data.train_target)
    test_x, _ = _arrays(data.test_features, data.test_target)
    if x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test must have the same feature count")
    if data.learning_rate <= 0 or data.epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    xs, test_xs, mean, scale = _scale_train_test(x, test_x)
    w = np.zeros(xs.shape[1], dtype=float)
    b = 0.0
    losses = []
    n = len(y)
    for _ in range(data.epochs):
        error = xs @ w + b - y
        losses.append(float(np.mean(error**2)))
        w -= data.learning_rate * (2.0 / n) * (xs.T @ error)
        b -= data.learning_rate * 2.0 * float(error.mean())
    raw_w = w / scale
    raw_b = b - float(mean @ raw_w)
    predictions = test_x @ raw_w + raw_b
    return RegressionResult(raw_w, raw_b, losses, predictions)


def _fit_logistic(train_x, train_y, learning_rate, epochs):
    """Возвращает параметры логистической модели в исходных единицах признаков."""
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    classes = np.unique(train_y)
    if not np.isin(classes, [0.0, 1.0]).all():
        raise ValueError("binary targets must be encoded as 0 and 1")
    xs, _, mean, scale = _scale_train_test(train_x, train_x)
    w = np.zeros(xs.shape[1], dtype=float)
    b = 0.0
    losses = []
    n = len(train_y)
    for _ in range(epochs):
        logits = np.clip(xs @ w + b, -500, 500)
        probs = 1.0 / (1.0 + np.exp(-logits))
        # BCE с устойчивым logaddexp вычислением.
        losses.append(float(np.mean(np.logaddexp(0.0, logits) - train_y * logits)))
        residual = probs - train_y
        w -= learning_rate * (xs.T @ residual) / n
        b -= learning_rate * float(residual.mean())
    raw_w = w / scale
    raw_b = b - float(mean @ raw_w)
    return raw_w, raw_b, losses


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучает бинарную логистическую модель пакетным градиентным спуском."""
    x, y = _arrays(data.train_features, data.train_target)
    test_x, _ = _arrays(data.test_features, data.test_target)
    if x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test must have the same feature count")
    weights, bias, losses = _fit_logistic(x, y, data.learning_rate, data.epochs)
    logits = np.clip(test_x @ weights + bias, -500, 500)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(weights, bias, losses, probabilities, predictions)


def _metrics(target, probabilities, predictions):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Считает метрики трёх классификаторов на предоставленном одном split."""
    x, y = _arrays(data.train_features, data.train_target)
    test_x, test_y = _arrays(data.test_features, data.test_target)
    if not np.isin(np.unique(y), [0.0, 1.0]).all() or not np.isin(np.unique(test_y), [0.0, 1.0]).all():
        raise ValueError("binary targets must be encoded as 0 and 1")

    own = train_logistic_classifier(data)
    own_metrics = _metrics(test_y, np.asarray(own.probabilities), np.asarray(own.predictions))

    scaler = StandardScaler().fit(x)
    xs, test_xs = scaler.transform(x), scaler.transform(test_x)
    logreg = LogisticRegression(max_iter=2000, random_state=42).fit(xs, y)
    log_probs = logreg.predict_proba(test_xs)[:, 1]
    log_metrics = _metrics(test_y, log_probs, (log_probs >= 0.5).astype(int))

    tree = DecisionTreeClassifier(random_state=42).fit(x, y)
    tree_probs = tree.predict_proba(test_x)[:, 1]
    tree_metrics = _metrics(test_y, tree_probs, tree.predict(test_x))

    knn = KNeighborsClassifier(n_neighbors=5).fit(xs, y)
    knn_probs = knn.predict_proba(test_xs)[:, 1]
    knn_metrics = _metrics(test_y, knn_probs, (knn_probs >= 0.5).astype(int))

    return BaselineComparisonResult(log_metrics, tree_metrics, knn_metrics)
