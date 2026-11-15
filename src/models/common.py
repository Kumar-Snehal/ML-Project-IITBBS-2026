"""Helpers shared by the model training and evaluation steps."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from src.config import CLASS_LABELS, TEST_FILE, TRAIN_FILE, VAL_FILE

N_CLASSES = len(CLASS_LABELS)


def load_splits():
    """(train, validation, test) DataFrames written by src.features.split_data."""
    return tuple(pd.read_csv(path) for path in (TRAIN_FILE, VAL_FILE, TEST_FILE))


def ranked_probability_score(y_true, proba) -> float:
    """RPS for ordered outcomes Away < Draw < Home (the column order of CLASS_LABELS).
    Unlike log loss it gives partial credit for predicting a draw when the result was a
    narrow win. 0 = perfect; lower is better."""
    onehot = np.eye(N_CLASSES)[np.asarray(y_true)]
    cum_diff = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum(cum_diff[:, :-1] ** 2, axis=1) / (N_CLASSES - 1)))


def brier_score(y_true, proba) -> float:
    """Multi-class Brier score: mean squared error of the probability vector. Lower is better."""
    onehot = np.eye(N_CLASSES)[np.asarray(y_true)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def score(y_true, proba) -> dict:
    """All metrics used in the project. Accuracy alone hides that draws are almost never
    predicted, so probability-quality metrics and macro-F1 are reported as well."""
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=list(range(N_CLASSES))),
        "brier": brier_score(y_true, proba),
        "rps": ranked_probability_score(y_true, proba),
        "macro_f1": f1_score(y_true, pred, average="macro", labels=list(range(N_CLASSES)),
                             zero_division=0),
        "draws_predicted": int((pred == CLASS_LABELS.index("D")).sum()),
    }


def confusion_table(y_true, proba) -> pd.DataFrame:
    pred = np.asarray(proba).argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(N_CLASSES)))
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_LABELS],
                        columns=[f"pred_{c}" for c in CLASS_LABELS])
