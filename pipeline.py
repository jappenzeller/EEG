import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from loader import load_subject_sequences, sorted_paths, load_segment
from features import extract_features
from viz import plot_gridsearch_heatmap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_kaggle_pipeline(n_estimators=3000, min_samples_split=2):
    """
    Construct a sklearn Pipeline with a scaler and RandomForestClassifier.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            bootstrap=False,
            random_state=0,
            n_jobs=-1
        ))
    ])


def make_grid_search(pipeline: Pipeline, param_grid: dict, cv_folds: int=5) -> GridSearchCV:
    """
    Wrap a pipeline in a GridSearchCV with StratifiedKFold.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )


def train_model(subject_dir: Path, fs: int, param_grid: dict,
                cv_folds: int=5, window_sec: float=0.1) -> tuple:
    """
    Load full concatenated ictal/interictal data, slide windows, extract features,
    perform GridSearchCV, save a heatmap of results, save and return best model and params.

    window_sec: window length in seconds (default 1.0)
    """
    # Compute window size and step in samples
    window_size = int(window_sec * fs)
    step = window_size

    # Load continuous sequences
    ictal, interictal = load_subject_sequences(str(subject_dir.parent), subject_dir.name)

    # Build feature matrix and labels
    X, y = [], []
    for start in range(0, ictal.shape[1] - window_size + 1, step):
        X.append(extract_features(ictal[:, start:start+window_size], fs)); y.append(1)
    for start in range(0, interictal.shape[1] - window_size + 1, step):
        X.append(extract_features(interictal[:, start:start+window_size], fs)); y.append(0)
    X = np.vstack(X)
    y = np.array(y)

    # Grid search
    pipeline = build_kaggle_pipeline()
    gs = make_grid_search(pipeline, param_grid, cv_folds)
    gs.fit(X, y)

    best_model = gs.best_estimator_
    best_params = gs.best_params_

    # Plot hyperparameter AUC heatmap
    heatmap_file = f"{subject_dir.name}_grid_heatmap.png"
    plot_gridsearch_heatmap(gs, param_grid, subject_dir.name, heatmap_file)

    # Save model
    model_file = f"{subject_dir.name}_model.pkl"
    joblib.dump(best_model, model_file)

    return best_model, best_params


def predict_test(subject_dir: Path, fs: int, model) -> pd.DataFrame:
    """
    Run model.predict_proba on test segments and return a DataFrame
    with columns ['clip','seizure'].
    """
    records = []
    subject = subject_dir.name
    for p in sorted_paths(subject_dir, 'test'):
        data = load_segment(p)
        feats = extract_features(data, fs).reshape(1, -1)
        prob = model.predict_proba(feats)[0, 1]
        records.append({
            'clip': f"{subject}_{p.stem}.mat",
            'seizure': prob
        })
    return pd.DataFrame(records)
