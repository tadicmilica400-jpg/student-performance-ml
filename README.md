# Student Performance Prediction

## Project overview

A supervised-learning coursework project that compares linear regression and k-nearest-neighbours regression for predicting an `Exam_Score` target from tabular student-related features.

## Academic context and data policy

The original dataset was supplied for university coursework. Its source and redistribution permissions are currently unknown, so it is intentionally **not included** in this repository. Do not commit a copy unless its provenance and licence are confirmed.

This project is an educational modelling exercise. Its predictions are not evidence of causal relationships and are not validated for grading, admissions or other decisions about students.

## Methodology

1. Read a CSV file and remove exact duplicate rows.
2. Validate the target and require at least 30 unique observations and one predictor.
3. Reserve 20% of observations as a holdout set (`random_state=42`).
4. Within a scikit-learn pipeline, median-impute and standardise numeric features; mode-impute and one-hot encode categorical features. Unknown categories are ignored at prediction time.
5. On the training partition only, use shuffled five-fold cross-validation to compare linear regression with KNN values of `k` allowed by the fold size (up to 20).
6. Select by mean absolute error, refit the selected pipeline on the training partition and evaluate once on the holdout set.

Keeping preprocessing inside the pipeline prevents fitted imputation, scaling and encoding statistics from leaking from validation folds or the holdout set into model fitting.

## Results and evaluation

The program reports the selected model and `k`, cross-validation MAE for every candidate, and holdout MAE, MSE and R². No real-data scores are published because the original dataset is unavailable for independent reproduction. The synthetic tests check pipeline behaviour, not predictive quality on students.

## Installation and usage

```bash
python -m pip install -r requirements.txt
python main.py --data /path/to/authorized/dataset.csv --output results/metrics.json
python -m unittest -v test_pipeline.py
```

### Compatible dataset contract

Authorized users should supply a CSV with a finite numeric target column named `Exam_Score`, at least one numeric or categorical predictor, and at least 30 unique rows after exact duplicates are removed.

Example coursework-style predictors include `Hours_Studied`, `Attendance`, `Previous_Scores`, `Teacher_Quality` and `Parental_Education_Level`; these are examples, not a guaranteed schema. Remove identifiers, free-text personal data and target-derived fields before training. Document the dataset source, licence, unit of observation and missing-value conventions separately.

## Repository structure

- `main.py` — validation, preprocessing, model selection and metric export
- `test_pipeline.py` — synthetic leakage/robustness and end-to-end checks
- `requirements.txt` — runtime dependencies

## Limitations and future improvements

- Random splitting assumes independent rows. Repeated students, classrooms, schools or time-ordered measurements require grouped or temporal validation.
- Hyperparameter search is limited to KNN neighbour count; it does not tune distance metrics, weighting or linear regularisation.
- No naive baseline, uncertainty intervals, subgroup diagnostics or fairness analysis are implemented.
- Tests could not be executed during this audit because Python was unavailable in the review environment.


