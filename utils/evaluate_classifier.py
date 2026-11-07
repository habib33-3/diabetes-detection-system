import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifiers(
    models: dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Sensitivity (Recall)": recall_score(y_test, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "NPV": tn / (tn + fn) if (tn + fn) else 0.0,
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "Brier Score": brier_score_loss(y_test, y_prob),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "Cohen's Kappa": cohen_kappa_score(y_test, y_pred),
            "True Negative": tn,
            "False Positive": fp,
            "False Negative": fn,
            "True Positive": tp,
        }
        results[model_name] = metrics

    return pd.DataFrame(results).round(4)