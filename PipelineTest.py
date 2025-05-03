import numpy as np
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Kaggle API for submission
from kaggle.api.kaggle_api_extended import KaggleApi

from SeizureDetectionPipelines import plot_gridsearch_heatmap,build_feature_label_matrix,build_kaggle_pipeline

if __name__ == '__main__':
    
    subject='Patient_1'
    data_root = 'H:/Data/PythonDNU/EEG/DataKaggle'

    full_path  = Path(f"{subject}_kaggle_full.pkl")
    grid_full_path = Path( f"{subject}_grid_full.pkl")

    early_path = Path(f"{subject}_kaggle_early.pkl")
 #   if full_path.exists():
 #       full_model = joblib.load(full_path)
 #       print(f"Loaded full-window model from {full_path}")

    param_grid = {
        'rf__n_estimators':     [1000, 2000, 3000, 4000,5000,7000],
        'rf__min_samples_split': [4,5,6,8,9,10]
    }
    
    if True:
        print(f"Training")
        X_full,  y_full  = build_feature_label_matrix(
            data_root, subject)
        grid_full = GridSearchCV(
            build_kaggle_pipeline(), param_grid,
            scoring='roc_auc', cv=5, n_jobs=-1,
            verbose=2
        )
        grid_full.fit(X_full, y_full)
        plot_gridsearch_heatmap(
            grid_full,
            'rf__n_estimators',
            'rf__min_samples_split'
        )
    if False:
    #if grid_full_path.exists():
        grid_full = joblib.load(grid_full_path)
        print(f"Loaded full-window grid from {grid_full_path}")
        plot_gridsearch_heatmap(
            grid_full,
            'rf__n_estimators',
            'rf__min_samples_split'
        )

    