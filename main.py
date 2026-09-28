"""Select a regression pipeline by training CV, then evaluate one holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number))])
    return Pipeline([('preprocessing', preprocessing), ('model', LinearRegression())])


def evaluate(path):
    data = pd.read_csv(path).drop_duplicates()
    if 'Exam_Score' not in data or len(data) < 30:
        raise ValueError('Provide Exam_Score and at least 30 unique rows.')
    y = pd.to_numeric(data['Exam_Score'], errors='raise')
    if not np.isfinite(y).all():
        raise ValueError('Exam_Score must be finite and non-missing.')
    X = data.drop(columns='Exam_Score')
    if X.shape[1] == 0:
        raise ValueError('At least one predictor is required.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    max_k = min(20, len(X_train) - int(np.ceil(len(X_train) / 5)))
    search = GridSearchCV(build_pipeline(), [
        {'model': [LinearRegression()]},
        {'model': [KNeighborsRegressor()], 'model__n_neighbors': list(range(1, max_k + 1))}
    ], scoring='neg_mean_absolute_error', cv=KFold(5, shuffle=True, random_state=42),
       refit=True, n_jobs=1, error_score='raise')
    search.fit(X_train, y_train)
    prediction = search.predict(X_test)
    model = search.best_estimator_.named_steps['model']
    return {'seed': 42, 'train_rows': len(X_train), 'test_rows': len(X_test),
            'selected_model': type(model).__name__, 'selected_k': getattr(model, 'n_neighbors', None),
            'cv_mae': float(-search.best_score_),
            'holdout_mae': float(mean_absolute_error(y_test, prediction)),
            'holdout_mse': float(mean_squared_error(y_test, prediction)),
            'holdout_r2': float(r2_score(y_test, prediction)),
            'cv_candidates': [{'model': type(p['model']).__name__, 'k': p.get('model__n_neighbors'),
                               'mae': float(-s)} for p,s in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('dataset.csv'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Dataset missing. See README.md and pass --data PATH.')
    try:
        result = evaluate(args.data)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
