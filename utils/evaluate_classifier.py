from pathlib import Path
from typing import Any

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


MODEL_RESULT_COLUMNS = [
    "model_name",
    "smote_applied",
    "test_size",
    "split_random_state",
    "stratified_split",
    "cv_folds",
    "cv_shuffle",
    "cv_random_state",
    "grid_search_scoring",
    "grid_search_best_f1",
    "grid_search_n_jobs",
    "classifier_random_state",
    "classifier_n_jobs",
    "classifier_penalty",
    "classifier_class_weight",
    "classifier_objective",
    "classifier_eval_metric",
    "classifier_tree_method",
    "classifier_device",
    "classifier_gpu_id",
    "classifier_reg_lambda",
    "smote_random_state",
    "smote_sampling_strategy",
    "smote_k_neighbors",
    "best_param_C",
    "best_param_solver",
    "best_param_max_iter",
    "best_param_n_estimators",
    "best_param_max_depth",
    "best_param_learning_rate",
    "best_param_subsample",
    "best_param_colsample_bytree",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]


def export_model_results(
    evaluation_results: pd.DataFrame,
    model_configs: dict[str, dict[str, Any]],
    output_path: Path,
    training_config: dict[str, Any],
) -> pd.DataFrame:
    """Upsert evaluated model configurations into a consistently structured CSV."""
    rows = []
    parameter_names = (
        "C",
        "solver",
        "max_iter",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    )
    classifier_config_names = (
        "random_state",
        "n_jobs",
        "penalty",
        "class_weight",
        "objective",
        "eval_metric",
        "tree_method",
        "device",
        "gpu_id",
        "reg_lambda",
    )
    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Sensitivity (Recall)",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    for result_name, config in model_configs.items():
        grid_search = config["grid_search"]
        pipeline = config["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        classifier_params = classifier.get_params(deep=False)
        cv = grid_search.cv
        smote = pipeline.named_steps.get("smote")

        row = {
            "model_name": config["model_name"],
            "smote_applied": smote is not None,
            "test_size": training_config["test_size"],
            "split_random_state": training_config["split_random_state"],
            "stratified_split": training_config["stratified_split"],
            "cv_folds": cv.get_n_splits() if hasattr(cv, "get_n_splits") else cv,
            "cv_shuffle": getattr(cv, "shuffle", None),
            "cv_random_state": getattr(cv, "random_state", None),
            "grid_search_scoring": grid_search.scoring,
            "grid_search_best_f1": grid_search.best_score_,
            "grid_search_n_jobs": grid_search.n_jobs,
            "smote_random_state": getattr(smote, "random_state", None),
            "smote_sampling_strategy": getattr(smote, "sampling_strategy", None),
            "smote_k_neighbors": getattr(smote, "k_neighbors", None),
        }
        row.update(
            {
                f"classifier_{name}": classifier_params.get(name)
                for name in classifier_config_names
            }
        )
        row.update(
            {
                f"best_param_{name}": grid_search.best_params_.get(
                    f"classifier__{name}"
                )
                for name in parameter_names
            }
        )
        row.update(
            {
                name: evaluation_results.at[metric, result_name]
                for name, metric in metric_names.items()
            }
        )
        rows.append(row)

    new_results = pd.DataFrame(rows, columns=MODEL_RESULT_COLUMNS)
    if output_path.exists():
        existing_results = pd.read_csv(output_path)
        if "model_name" not in existing_results.columns:
            raise ValueError(f"{output_path} does not contain a model_name column.")
        existing_results = existing_results[
            ~existing_results["model_name"].isin(
                {config["model_name"] for config in model_configs.values()}
            )
        ].reindex(columns=MODEL_RESULT_COLUMNS)

    else:
        existing_results = pd.DataFrame(columns=MODEL_RESULT_COLUMNS)

    combined_results = pd.concat(
        [existing_results, new_results], ignore_index=True
    ).sort_values(["model_name", "smote_applied"], ignore_index=True)
    combined_results.to_csv(output_path, index=False)
    return combined_results