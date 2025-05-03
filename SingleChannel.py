import os
import glob
import scipy.io as sio
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class PosNegDynamicUnit:
    # define EEG frequency bands
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  13),
        'beta':  (13, 30),
        'gamma': (30,100),
    }

    def __init__(self):
        self.fs = None   # sampling frequency (Hz) will be set when loading a segment

    def load_segment(self, mat_path: str) -> np.ndarray:
        """
        Load one EEG segment from a .mat file.
        Returns:
            data: 2D NumPy array (channels, samples)
        """
        mat = sio.loadmat(mat_path)
        key = next(k for k in mat if k.endswith('_segment_1'))
        seg = mat[key].squeeze()

        data = seg['data']
        self.fs = float(seg['sampling_frequency'].squeeze())
        return data

    def load_all_by_type(self, folder_path: str) -> dict:
        """
        Load all segments grouped by type ('ictal', 'interictal', 'test').
        Returns:
            segments: dict mapping type -> list of 2D data arrays
        """
        types = ['ictal', 'interictal', 'test']
        segments = {t: [] for t in types}

        for t in types:
            pattern = os.path.join(folder_path, f'*{t}*.mat')
            for fn in glob.glob(pattern):
                data = self.load_segment(fn)
                segments[t].append(data)
            if not segments[t]:
                print(f"⚠️ Warning: no '{t}' files found in {folder_path}")
        return segments

    def compute_band_powers(self, data: np.ndarray) -> dict:
        """
        Compute mean power in each EEG band for a single segment.
        Returns:
            band_powers: dict mapping band -> mean power across channels
        """
        freqs, psd = signal.welch(data, fs=self.fs, axis=-1, nperseg=1024)
        band_powers = {}
        for band, (f_low, f_high) in self.BANDS.items():
            idx = np.logical_and(freqs >= f_low, freqs <= f_high)
            powers = np.trapz(psd[:, idx], freqs[idx], axis=-1)
            band_powers[band] = float(np.mean(powers))
        return band_powers

    def aggregate_band_powers(self, segments: dict) -> dict:
        """
        Compute and average band powers for each segment type.
        Args:
            segments: dict mapping type -> list of data arrays
        Returns:
            avg_powers: dict mapping type -> list of mean powers in BANDS order
        """
        bands = list(self.BANDS.keys())
        avg_powers = {}
        for t, data_list in segments.items():
            if not data_list:
                continue
            # compute powers for each segment
            powers_list = [self.compute_band_powers(data) for data in data_list]
            # assemble into an array: (n_segments, n_bands)
            arr = np.array([[p[b] for b in bands] for p in powers_list])
            avg_powers[t] = arr.mean(axis=0)
        return avg_powers

    def plot_band_powers(self, avg_powers: dict):
        """
        Plot average band powers for each segment type.
        """
        bands = list(self.BANDS.keys())
        x = np.arange(len(bands))
        plt.figure()
        for t, powers in avg_powers.items():
            plt.plot(x, powers, marker='o', label=t)
        plt.xticks(x, bands)
        plt.xlabel('EEG Band')
        plt.ylabel('Mean Band Power')
        plt.title('Mean EEG Band Power by Segment Type')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_folder = r"H:\Data\PythonDNU\EEG\DataKaggle"
    unit = PosNegDynamicUnit()
    segments = unit.load_all_by_type(data_folder)
    avg_powers = unit.aggregate_band_powers(segments)
    if not avg_powers:
        raise RuntimeError(f"No valid segments found in {data_folder}")
    unit.plot_band_powers(avg_powers)
