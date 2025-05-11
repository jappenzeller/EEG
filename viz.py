import numpy as np
import matplotlib.pyplot as plt


def plot_fft_out(fft_out: np.ndarray):
    """
    Visualize FFT output for all channels.

    Parameters:
        fft_out: np.ndarray of shape (n_channels, n_bins)
    """
    freqs = np.arange(1, fft_out.shape[1] + 1)
    plt.figure(figsize=(10, 6))
    for ch in range(fft_out.shape[0]):
        plt.plot(freqs, fft_out[ch], label=f'Ch {ch}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log10 Magnitude')
    plt.title('FFT Magnitudes per Channel')
    plt.legend(ncol=4, fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(data: np.ndarray, fs: int, channel: int = 1):
    """
    Plot a spectrogram of a single channel.

    Parameters:
        data: np.ndarray of shape (n_channels, n_samples)
        fs: sampling rate (Hz)
        channel: channel index (1-based)
    """
    x = data[channel - 1]
    window = round(0.5 * fs)
    noverlap = round(0.4 * fs)
    nfft = max(512, 2 ** int(np.ceil(np.log2(window))))

    plt.figure()
    plt.specgram(x, NFFT=nfft, Fs=fs, noverlap=noverlap)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.title(f'Channel {channel} Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim((0, fs / 2))
    plt.tight_layout()
    plt.show()


def plot_gridsearch_heatmap(grid: any, param_grid: dict, subject_name: str, out_file: str):
    """
    Plot and save a heatmap of GridSearchCV mean_test_score results.

    Parameters:
        grid: fitted GridSearchCV object
        param_grid: dict of two hyperparameters and their values
        subject_name: name of the subject for the title
        out_file: path to save the heatmap PNG
    """
    # Expect exactly two hyperparameters
    params = list(param_grid.keys())
    if len(params) != 2:
        raise ValueError('plot_gridsearch_heatmap requires exactly two parameters')
    p1, p2 = params
    v1_list = param_grid[p1]
    v2_list = param_grid[p2]

    results = grid.cv_results_
    score_matrix = np.zeros((len(v2_list), len(v1_list)))
    for mean_score, v1, v2 in zip(results['mean_test_score'],
                                  results[f'param_{p1}'],
                                  results[f'param_{p2}']):
        i = v2_list.index(v2)
        j = v1_list.index(v1)
        score_matrix[i, j] = mean_score

    plt.figure(figsize=(6, 5))
    plt.imshow(score_matrix, aspect='auto', origin='lower')
    plt.xticks(range(len(v1_list)), v1_list)
    plt.yticks(range(len(v2_list)), v2_list)
    plt.xlabel(p1)
    plt.ylabel(p2)
    plt.title(f'{subject_name} GridSearch AUC')
    for (i, j), v in np.ndenumerate(score_matrix):
        plt.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize='small')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
