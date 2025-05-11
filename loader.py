import numpy as np
import scipy.io as sio
from pathlib import Path
import re

def load_segment(mat_path: Path) -> np.ndarray:
    """
    Load a single EEG segment from a .mat file.

    Returns:
        data: np.ndarray of shape (n_channels, n_samples)
    """
    mat = sio.loadmat(str(mat_path))
    # find the primary variable key
    var_keys = [k for k in mat.keys() if not k.startswith('__')]
    arr = mat[var_keys[0]]
    # unwrap singleton arrays
    while isinstance(arr, np.ndarray) and arr.size == 1:
        arr = arr.squeeze()
    # structured array data field
    if hasattr(arr, 'dtype') and arr.dtype.names:
        return arr['data']
    if isinstance(arr, np.ndarray):
        return arr
    raise ValueError(f"Cannot parse data in {mat_path}")


def sorted_paths(subj_dir: Path, seg_type: str) -> list[Path]:
    """
    Return sorted list of segment files matching exactly '_<seg_type>_' in their name.

    seg_type should be 'ictal', 'interictal', or 'test'.
    """
    pattern = f"*_{seg_type}_*.mat"
    paths = list(subj_dir.glob(pattern))
    def idx(p: Path):
        m = re.search(r"_(\d+)\.mat$", p.name)
        return int(m.group(1)) if m else -1
    return sorted(paths, key=idx)


def load_subject_sequences(root_dir: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate all segments of a given type for a subject into continuous arrays.

    Returns:
        ictal: np.ndarray of shape (n_channels, total_ictal_samples)
        interictal: np.ndarray of shape (n_channels, total_interictal_samples)
    """
    subj_path = Path(root_dir) / subject
    # load and concatenate per channel
    def concat_segments(seg_type: str) -> np.ndarray:
        files = sorted_paths(subj_path, seg_type)
        arrays = [load_segment(p) for p in files]
        if not arrays:
            return np.empty((0,0))
        return np.concatenate(arrays, axis=1)

    ictal = concat_segments('ictal')
    interictal = concat_segments('interictal')
    return ictal, interictal


if __name__ == '__main__':
    # Quick self-test
    from loader import load_subject_sequences
    root = 'H:/Data/PythonDNU/EEG/DataKaggle'
    for subj in ['patient_1', 'dog_1']:
        i, ii = load_subject_sequences(root, subj)
        print(f"{subj}: ictal={i.shape}, interictal={ii.shape}")
