## XGBoost – Next Phase Task List

### Phase 1: Hyperparameter Optimization

- [x] Implement `GridSearchCV` for XGBoost.
- [x] Define a reasonable hyperparameter search space.
- [x] Tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
- [x] Evaluate the best parameters using stratified cross-validation.
- [ ] Justify hyperparameter choices with code comments (the search-space choices still need justification).

### Phase 2: SMOTE Optimization

- [x] Integrate SMOTE into an `imblearn` pipeline.
- [ ] Experiment with different `sampling_strategy` values (only the default strategy is used).
- [ ] Experiment with different `k_neighbors` values (only the default value is used).
- [x] Compare XGBoost performance with and without SMOTE.
- [x] Ensure SMOTE is applied only to training folds to prevent data leakage.

### Phase 3: Model Evaluation

- [x] Evaluate the best model with and without SMOTE on the test set.
- [x] Generate confusion matrices.
- [x] Compare precision, recall, and F1-score for both classes.
- [x] Compare accuracy, balanced accuracy, macro F1, and weighted F1.
- [x] Compare ROC-AUC and PR-AUC.
- [ ] Analyze false positives and false negatives (counts are reported, but their implications are not analyzed).
- [x] Return evaluation results in a unified DataFrame.

### Phase 4: Classification Threshold Optimization

- [x] Evaluate different classification thresholds.
- [x] Plot precision-recall trade-offs.
- [x] Identify thresholds that improve diabetes recall without excessive false positives.
- [x] Compare default threshold (0.5) against the optimized threshold.
- [x] Evaluate the selected threshold on the untouched test set.

### Phase 5: Final Model Comparison

- [ ] Compare Logistic Regression and XGBoost (results are in separate notebooks, with no side-by-side comparison).
- [ ] Compare models with and without SMOTE (the within-XGBoost comparison exists; a cross-model comparison is missing).
- [ ] Identify the best-performing model based on minority-class performance (XGBoost variants are evaluated, but not ranked against Logistic Regression).
- [ ] Document the trade-offs between recall and precision (the separate result tables show metrics, but no cross-model trade-off analysis is documented).
- [ ] Select the final candidate model.

### Phase 6: Separate Sampling-Strategy XGBoost Pipeline

- [x] Add a standalone local notebook using the existing cleaned dataset and stratified train-test split.
- [x] Compare the unsampled baseline, random over-sampling, random under-sampling, and SMOTE.
- [x] Keep preprocessing and sampling inside pipelines; evaluate only on the untouched test split.
- [x] Tune each strategy with a practical `GridSearchCV` grid, stratified 5-fold CV, and F1 scoring.
- [x] Compare the best sampling strategy against the baseline using the existing evaluation metrics.
- [ ] Run the notebook end-to-end locally (not verified; no Python runtime is available in this environment).

### Important Constraints

- [x] Keep the existing XGBoost training workflow consistent.
- [x] Avoid data leakage during preprocessing, SMOTE, and cross-validation.
- [x] Maintain consistent function signatures across evaluation utilities.
- [x] Preserve reproducibility using fixed random states.
- [ ] Justify important training and optimization decisions through comments (the search-space choices still need justification).

Expected outcome: A properly tuned XGBoost model with a measurable improvement in diabetes detection, supported by leakage-free evaluation and a comparison against Logistic Regression.