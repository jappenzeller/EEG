import os
import glob
import argparse
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import re

# ---------------------------- Data Loading ----------------------------

def load_segment(mat_path: str) -> np.ndarray:
    """
    Load EEG segment from .mat (channels × samples).
    """
    mat = sio.loadmat(mat_path)
    var_keys = [k for k in mat if not k.startswith('__')]
    arr = mat[var_keys[0]]
    while isinstance(arr, np.ndarray) and arr.size == 1:
        arr = arr.squeeze()
    if hasattr(arr, 'dtype') and arr.dtype.names:
        return arr['data']
    if isinstance(arr, np.ndarray):
        return arr
    raise ValueError(f"Cannot parse data in {mat_path}")

# ------------------------- Feature Extraction -------------------------

def fft_features(data: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.fft.rfft(data, axis=1)[:, 1:48]))


def freq_corr_features(fft_data: np.ndarray) -> np.ndarray:
    scaled = scale(fft_data, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    corr_feats = corr[iu]
    eigs = np.sort(np.abs(np.linalg.eigvals(corr)))
    return np.concatenate([corr_feats, eigs])


def time_corr_features(data: np.ndarray, target_len: int = 400) -> np.ndarray:
    if data.shape[1] > target_len:
        data = signal.resample(data, target_len, axis=1)
    scaled = scale(data, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    corr_feats = corr[iu]
    eigs = np.sort(np.abs(np.linalg.eigvals(corr)))
    return np.concatenate([corr_feats, eigs])


def extract_features(data: np.ndarray) -> np.ndarray:
    fft_out = fft_features(data)
    return np.concatenate([fft_out.ravel(), freq_corr_features(fft_out), time_corr_features(data)])

# ----------------------------- Subject Dataset -----------------------------

def build_subject_train(root_dir: str, subject: str):
    subj_path = Path(root_dir) / subject
    X, y = [], []

    def extract_index(p: Path) -> int:
        # assumes filenames like Dog_1_ictal_segment_12.mat
        m = re.search(r'_(\d+)\.mat$', p.name)
        return int(m.group(1)) if m else 0

    for seg_type, label in [('ictal', 1), ('interictal', 0)]:
        # grab all matching .mat paths...
        all_paths = list(subj_path.glob(f'*{seg_type}*.mat'))
        # ...then sort them by the trailing number in the filename
        sorted_paths = sorted(all_paths, key=extract_index)

        for mat_path in sorted_paths:
            data = load_segment(mat_path)
            X.append(extract_features(data))
            y.append(label)

    return np.array(X), np.array(y)


def build_subject_test(root_dir: str, subject: str):
    subj_dir = os.path.join(root_dir, subject)
    X_test, ids = [], []
    pattern = os.path.normpath(os.path.join(subj_dir, '*test*.mat'))

    for fn in glob.glob(pattern):
        data = load_segment(fn)
        X_test.append(extract_features(data))
        clip = os.path.splitext(os.path.basename(fn))[0]
        ids.append(f"{subject}_{clip}")
    return np.array(X_test), ids

# ------------------------- Kaggle Integration -------------------------

def submit_to_kaggle(csv_path: str, message: str, competition: str):
    api = KaggleApi(); api.authenticate()
    api.competition_submit(file_name=csv_path, message=message, competition=competition)


def list_submissions(competition: str):
    api = KaggleApi(); api.authenticate()
    for s in api.competition_submissions(competition):
        print(f"{s.id} | {s.date} | {s.status} | pub: {s.public_score} | priv: {s.private_score} | msg: {s.description}")

# ----------------------------- Main Pipeline -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seizure detection for one subject")
    parser.add_argument('subject', nargs='?', default='Dog_1',
                        help="Subject folder name (default: Dog_1)")
    parser.add_argument('--data-root', default="H:/Data/PythonDNU/EEG/DataKaggle",
                        help="Root directory containing subject folders")
    parser.add_argument('--competition', default='seizure-detection',
                        help="Kaggle competition slug")
    args = parser.parse_args()
    subject = args.subject
    data_root = args.data_root
    competition = args.competition
    subj_dir = os.path.join(data_root, subject).replace('\\', '/')  # normalize to forward slashes
    if not os.path.isdir(subj_dir):
        print(f"Error: subject directory not found: {subj_dir}")
        sys.exit(1)
    print(f"Processing subject: {subject}")

    # Build training data
    X, y = build_subject_train(data_root, subject)
    if X.size == 0:
        print(f"No training segments found for subject {subject} in {subj_dir}")
        sys.exit(1)
    print(f"  Train data: X={X.shape}, y={y.shape}")

    # Cross-validation
    clf = RandomForestClassifier(n_estimators=3000, min_samples_split=2.0,
                                 bootstrap=False, random_state=0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit and save model
    clf.fit(X, y)
    model_path = f"{subject}_model.pkl"
    joblib.dump(clf, model_path)
    print(f"  Model saved to {model_path}")

    # Build test data and predict
    X_test, ids = build_subject_test(data_root, subject)
    if X_test.size == 0:
        print(f"No test segments found for subject {subject} in {subj_dir}")
        sys.exit(1)
    print(f"  Test data: X_test={X_test.shape}, clips={len(ids)}")
    preds = clf.predict_proba(X_test)[:,1]

    # Write submission
    submission_csv = f"{subject}_submission.csv"
    pd.DataFrame({'clip_id': ids, 'prediction': preds}).to_csv(submission_csv, index=False)
    print(f"  Submission file: {submission_csv}")

    # Kaggle submit
    submit_to_kaggle(submission_csv, f"Submission for {subject}", competition)
    print("Recent submissions:")
    list_submissions(competition)
