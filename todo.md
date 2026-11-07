## XGBoost – Next Phase Task List

### Phase 1: Hyperparameter Optimization

- Implement `GridSearchCV` for XGBoost.
- Define a reasonable hyperparameter search space.
- Tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
- Evaluate the best parameters using stratified cross-validation.
- Justify hyperparameter choices with code comments.

### Phase 2: SMOTE Optimization

- Integrate SMOTE into an `imblearn` pipeline.
- Experiment with different `sampling_strategy` values.
- Experiment with different `k_neighbors` values.
- Compare XGBoost performance with and without SMOTE.
- Ensure SMOTE is applied only to training folds to prevent data leakage.

### Phase 3: Model Evaluation

- Evaluate the best model with and without SMOTE on the test set.
- Generate confusion matrices.
- Compare precision, recall, and F1-score for both classes.
- Compare accuracy, balanced accuracy, macro F1, and weighted F1.
- Compare ROC-AUC and PR-AUC.
- Analyze false positives and false negatives.
- Return evaluation results in a unified DataFrame.

### Phase 4: Classification Threshold Optimization

- Evaluate different classification thresholds.
- Plot precision-recall trade-offs.
- Identify thresholds that improve diabetes recall without excessive false positives.
- Compare default threshold (0.5) against the optimized threshold.
- Evaluate the selected threshold on the untouched test set.

### Phase 5: Final Model Comparison

- Compare Logistic Regression and XGBoost.
- Compare models with and without SMOTE.
- Identify the best-performing model based on minority-class performance.
- Document the trade-offs between recall and precision.
- Select the final candidate model.

### Important Constraints

- Keep the existing XGBoost training workflow consistent.
- Avoid data leakage during preprocessing, SMOTE, and cross-validation.
- Maintain consistent function signatures across evaluation utilities.
- Preserve reproducibility using fixed random states.
- Justify important training and optimization decisions through comments.

Expected outcome: A properly tuned XGBoost model with a measurable improvement in diabetes detection, supported by leakage-free evaluation and a comparison against Logistic Regression.