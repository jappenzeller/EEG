from pathlib import Path
import re
import numpy as np
import SeizureDetectionPipeline

def load_subject_sequences(root_dir: str, subject: str):
    """
    Returns two arrays:
      ictal_seq       — shape (channels, total_ictal_samples)
      interictal_seq  — shape (channels, total_interictal_samples)

    Segments are sorted by the trailing segment number in their filename
    (e.g. Dog_1_ictal_segment_2.mat comes before …segment_10.mat).
    """
    subj_path = Path(root_dir) / subject
    if not subj_path.is_dir():
        raise FileNotFoundError(f"{subj_path} not found")

    def sorted_segment_paths(seg_type: str):
        # match filenames containing the seg_type and capture their segment index
        pattern = f"*{seg_type}*.mat"
        def idx(p: Path):
            m = re.search(r"_(\d+)\.mat$", p.name)
            return int(m.group(1)) if m else -1
        all_paths = list(subj_path.glob(pattern))
        if not all_paths:
            raise FileNotFoundError(f"No '{seg_type}' files found in {subj_path}")
        return sorted(all_paths, key=idx)

    sequences = {}
    for seg_type in ("ictal", "interictal"):
        paths = sorted_segment_paths(seg_type)
        # load and concatenate along the time/samples axis (axis=1)
        data_blocks = [SeizureDetectionPipeline.load_segment(p) for p in paths]
        sequences[seg_type] = np.concatenate(data_blocks, axis=1)

    return sequences["ictal"], sequences["interictal"]

if __name__ == "__main__":
    root = "H:/Data/PythonDNU/EEG/DataKaggle"
    subject = "Dog_1"

    # get two long arrays of raw EEG:
    ictal_stream, interictal_stream = load_subject_sequences(root, subject)

    print("Ictal stream shape:     ", ictal_stream.shape)
    print("Interictal stream shape:", interictal_stream.shape)

    # Example: feed a sliding window into your PNDNU:
    window_size = 400  # samples per evaluation
    step = 50          # overlap/step

    for start in range(0, ictal_stream.shape[1] - window_size + 1, step):
        window = ictal_stream[:, start : start + window_size]
        # pndnu.process(window)  <-- your real-time inference here

