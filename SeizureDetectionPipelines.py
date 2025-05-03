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

def plot_fft_out(fft_out: np.ndarray):
    """
    Visualize FFT output for all channels.

    Parameters:
    - fft_out: np.ndarray of shape (n_channels, n_bins), log10 magnitudes for 1–47 Hz.
    """
    # Frequency bins 1–47 Hz
    freqs = np.arange(1, fft_out.shape[1] + 1)

    # Line plot per channel
    plt.figure(figsize=(10, 6))
    for ch_idx in range(fft_out.shape[0]):
        plt.plot(freqs, fft_out[ch_idx], label=f'Ch {ch_idx}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log10 Magnitude')
    plt.title('FFT Magnitudes per Channel')
    plt.legend(ncol=4, fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()

    # Heatmap of all channels
    plt.figure(figsize=(8, 6))
    plt.imshow(fft_out, aspect='auto', origin='lower',
               extent=[freqs[0], freqs[-1], 0, fft_out.shape[0]])
    plt.colorbar(label='Log10 Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel Index')
    plt.title('FFT Heatmap Across Channels')
    plt.tight_layout()
    plt.show()

# --- Data loading & concatenation ---
def load_segment(mat_path: str) -> np.ndarray:
    import scipy.io as sio
    mat = sio.loadmat(str(mat_path))
    var_keys = [k for k in mat if not k.startswith('__')]
    arr = mat[var_keys[0]]
    while hasattr(arr, 'shape') and arr.size == 1:
        arr = arr.squeeze()
    if hasattr(arr, 'dtype') and arr.dtype.names:
        return arr['data']
    if isinstance(arr, np.ndarray):
        return arr
    raise ValueError(f"Cannot parse data in {mat_path}")

def plot_gridsearch_heatmap(grid, param1, param2):
    """
    grid     : your fitted GridSearchCV instance
    param1   : string, name of first hyper-param (e.g. 'rf__n_estimators')
    param2   : string, name of second hyper-param (e.g. 'rf__min_samples_split')
    """
    results = grid.cv_results_
    # pull out the sorted, unique values
    p1_vals = sorted({p for p in results[f'param_{param1}']})
    p2_vals = sorted({p for p in results[f'param_{param2}']})

    # build a score matrix of shape (len(p2_vals), len(p1_vals))
    score_matrix = np.zeros((len(p2_vals), len(p1_vals)))
    for mean_score, v1, v2 in zip(results['mean_test_score'],
                                  results[f'param_{param1}'],
                                  results[f'param_{param2}']):
        i = p2_vals.index(v2)
        j = p1_vals.index(v1)
        score_matrix[i, j] = mean_score

    plt.figure(figsize=(6, 5))
    plt.imshow(score_matrix, aspect='auto')
    plt.xticks(range(len(p1_vals)), p1_vals)
    plt.yticks(range(len(p2_vals)), p2_vals)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('GridSearchCV mean_test_score')
    # annotate each cell with the score
    for (i, j), v in np.ndenumerate(score_matrix):
        plt.text(j, i, f"{v:.3f}", ha='center', va='center', fontsize='small')
    plt.tight_layout()
    plt.show()
  #  plt.show(block=False)    # don’t block
  #  plt.pause(0.1)  

def load_subject_sequences(root_dir: str, subject: str):
    subj_path = Path(root_dir) / subject
    def sorted_paths(seg_type: str):
        def idx(p: Path):
            m = re.search(r"_(\d+)\.mat$", p.name)
            return int(m.group(1)) if m else -1
        return sorted(subj_path.glob(f"*{seg_type}*.mat"), key=idx)

    ictal = np.concatenate([load_segment(p) for p in sorted_paths('ictal')], axis=1)
    interictal = np.concatenate([load_segment(p) for p in sorted_paths('interictal')], axis=1)
    return ictal, interictal

# --- Feature extraction ---
def fft_features(data: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.fft.rfft(data, axis=1)[:, 1:48]))

def freq_corr_feats(fft_data: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import scale
    scaled = scale(fft_data, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    return np.concatenate([corr[iu], np.sort(np.abs(np.linalg.eigvals(corr)))])

def time_corr_feats(data: np.ndarray, target_len: int = 400) -> np.ndarray:
    from scipy.signal import resample
    from sklearn.preprocessing import scale
    if data.shape[1] != target_len:
        data = resample(data, target_len, axis=1)
    scaled = scale(data, axis=0)
    corr = np.corrcoef(scaled)
    iu = np.triu_indices_from(corr, k=1)
    return np.concatenate([corr[iu], np.sort(np.abs(np.linalg.eigvals(corr)))])

def extract_features_matrix(data: np.ndarray) -> np.ndarray:
    fft_out = fft_features(data)
   # plot_fft_out(fft_out)
    return np.concatenate([
        fft_out.ravel(),
        freq_corr_feats(fft_out),
        time_corr_feats(data)
    ])

# --- Build datasets ---
def build_feature_label_matrix(root_dir: str, subject: str, window_size: int = 400, step: int = 400):
    ictal, interictal = load_subject_sequences(root_dir, subject)
    X, y = [], []
    for start in range(0, ictal.shape[1] - window_size + 1, step):
        X.append(extract_features_matrix(ictal[:, start:start+window_size])); y.append(1)
    for start in range(0, interictal.shape[1] - window_size + 1, step):
        X.append(extract_features_matrix(interictal[:, start:start+window_size])); y.append(0)
    return np.array(X), np.array(y)

def build_early_feature_label_matrix(root_dir: str, subject: str, early_seconds: int = 15, window_size: int = 400):
    ictal, interictal = load_subject_sequences(root_dir, subject)
    max_sample = min(ictal.shape[1], early_seconds * 400)
    X, y = [], []
    for start in range(0, max_sample - window_size + 1, window_size):
        X.append(extract_features_matrix(ictal[:, start:start+window_size])); y.append(1)
    for start in range(0, interictal.shape[1] - window_size + 1, window_size):
        X.append(extract_features_matrix(interictal[:, start:start+window_size])); y.append(0)
    return np.array(X), np.array(y)

# --- Test data loader ---
def load_subject_test(root_dir: str, subject: str):
    subj_path = Path(root_dir) / subject
    def idx(p: Path):
        m = re.search(r"_(\d+)\.mat$", p.name)
        return int(m.group(1)) if m else -1
    paths = sorted(subj_path.glob("*test*.mat"), key=idx)
    X_test, ids = [], []
    for p in paths:
        X_test.append(extract_features_matrix(load_segment(p)))
        ids.append(f"{subject}_{p.stem}")
    return np.array(X_test), ids

# --- Pipeline and grid search factory ---
def build_kaggle_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(bootstrap=False, random_state=0, n_jobs=-1))
    ])

# --- Main: train, tune, predict, and submit ---
if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import joblib
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi
    from sklearn.model_selection import GridSearchCV

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description='Train or load seizure-detection models and generate submission')
    parser.add_argument('--data-root', default='H:/Data/PythonDNU/EEG/DataKaggle',
                        help='Root directory containing subject subfolders')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='List of subjects (omit to auto-discover from data-root)')
    parser.add_argument('--early-seconds', type=int, default=15,
                        help='Window length in seconds for early detection')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the resulting CSV to Kaggle')
    parser.add_argument('--competition', default='seizure-detection',
                        help='Kaggle competition slug')
    args = parser.parse_args()

    # --- Resolve subject list ---
    subjects = args.subjects or [
        p.name for p in Path(args.data_root).iterdir() if p.is_dir()
    ]

    # --- Hyperparameter grid ---
    param_grid = {
        'rf__n_estimators':     [3000],
        'rf__min_samples_split': [2]
    }

    submissions = []

    # --- Per-subject train/load & predict loop ---
    for subject in subjects:
        print(f"=== Processing {subject} ===")

        # Build feature/label matrices
        X_full,  y_full  = build_feature_label_matrix(
                              args.data_root, subject)
        X_early, y_early = build_early_feature_label_matrix(
                              args.data_root, subject, args.early_seconds)

        # Define model file paths
        full_path  = Path(f"{subject}_kaggle_full.pkl")
        early_path = Path(f"{subject}_kaggle_early.pkl")

        # --- Full-window model ---
        if full_path.exists():
            full_model = joblib.load(full_path)

            print(f"Loaded full-window model from {full_path}")
        else:
            grid_full = GridSearchCV(
                build_kaggle_pipeline(), param_grid,
                scoring='roc_auc', cv=5, n_jobs=-1
            )
            grid_full.fit(X_full, y_full)
      #      plot_gridsearch_heatmap(
      #          grid_full,
      #          'rf__n_estimators',
      #          'rf__min_samples_split'
      #      )
            full_model = grid_full.best_estimator_
            joblib.dump(full_model, full_path)
            joblib.dump(grid_full, f"{subject}_grid_full.pkl")
            print(f"Trained & saved full-window model to {full_path} ({grid_full.best_params_})")

        # --- Early-window model ---
        if early_path.exists():
            early_model = joblib.load(early_path)
            print(f"Loaded early-window model from {early_path}")
        else:
            grid_early = GridSearchCV(
                build_kaggle_pipeline(), param_grid,
                scoring='roc_auc', cv=5, n_jobs=-1
            )
            grid_early.fit(X_early, y_early)
      #      plot_gridsearch_heatmap(
      #          grid_early,
      #          'rf__n_estimators',
      #          'rf__min_samples_split'
      #      )
            early_model = grid_early.best_estimator_
            joblib.dump(early_model, early_path)
            joblib.dump(grid_early, f"{subject}_grid_early.pkl")
            print(f"Trained & saved early-window model to {early_path} ({grid_early.best_params_})")

        # --- Generate predictions for test segments ---
#        X_test, ids = load_subject_test(args.data_root, subject)
#        full_preds  = full_model.predict_proba(X_test)[:, 1]
#        early_preds = early_model.predict_proba(X_test)[:, 1]

#        for cid, f, e in zip(ids, full_preds, early_preds):
#            submissions.append({
#                'clip':    f"{cid}.mat",
#                'seizure': f,
#                'early':   e
#            })

    # --- Write submission CSV ---
    df = pd.DataFrame(submissions, columns=['clip','seizure','early'])
    df.to_csv('submission.csv', index=False)
    print("Wrote submission.csv")

    # --- Optional Kaggle submit ---
    if args.submit:
        api = KaggleApi()
        api.authenticate()
        api.competition_submit(
            'submission.csv',
            'Automated full+early predictions',
            args.competition
        )
        print("Submitted to Kaggle")
