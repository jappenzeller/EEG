import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from SeizureDetectionPipelines import load_subject_sequences, fft_features


def main():
    parser = argparse.ArgumentParser(
        description="Simulated real-time sliding-window time-series + heatmap at 30 FPS"
    )
    parser.add_argument('--data-root', default='H:/Data/PythonDNU/EEG/DataKaggle',
                        help='Root directory with subject folders')
    parser.add_argument('--subject', default='Dog_1', help='Subject folder name')
    parser.add_argument('--window-size', type=int, default=400,
                        help='Samples per window (400 = 1s at 400Hz)')
    parser.add_argument('--sampling-rate', type=float, default=400.0,
                        help='Sampling rate in Hz (default: 400)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second for simulation (default: 30)')
    args = parser.parse_args()

    # Derived parameters
    sr = args.sampling_rate
    window = args.window_size
    fps = args.fps
    data_root = args.data_root
    subject = args.subject
    step = int(sr / fps)  # samples to advance per frame
    interval = 1000.0 / fps  # ms per frame


    # Load data
    ictal, _ = load_subject_sequences(data_root, subject)
    n_channels, n_samples = ictal.shape
    n_frames = (n_samples - window) // step

    # Setup figure
    fig, (ax_time, ax_heat) = plt.subplots(1, 2, figsize=(12, 6))
    time_lines = []
    im = None
    freq_axis = None

    def init():
        nonlocal time_lines, im, freq_axis
        sample = ictal[:, :window]
        fft_out = fft_features(sample)
        n_bins = fft_out.shape[1]
        freq_axis = np.arange(1, n_bins + 1)

        # Time-series plot
        ax_time.clear()
        for ch in range(n_channels):
            (ln,) = ax_time.plot([], [], lw=1)
            time_lines.append(ln)
        ax_time.set_xlim(0, window / sr)
        ax_time.set_ylim(ictal.min(), ictal.max())
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title('Time-Series Sliding Window')

        # Heatmap plot (update per second)
        ax_heat.clear()
        im = ax_heat.imshow(
            fft_out,
            aspect='auto', origin='lower',
            extent=[1, n_bins, 0, n_channels],
            vmin=fft_out.min(), vmax=fft_out.max()
        )
        ax_heat.set_xlabel('Frequency (Hz)')
        ax_heat.set_ylabel('Channel')
        ax_heat.set_title('FFT Heatmap (1 Hz updates)')

        return time_lines + [im]

    def update(frame_idx):
        start = frame_idx * step
        segment = ictal[:, start:start + window]
        # Update time-series
        time_axis = (np.arange(window) / sr) + (start / sr)
        for ch, ln in enumerate(time_lines):
            ln.set_data(time_axis, segment[ch])

        # Update heatmap once per second (every fps frames)
        if frame_idx % int(fps) == 0:
            sec_idx = frame_idx // int(fps)
            idx_start = sec_idx * window
            fft_data = fft_features(ictal[:, idx_start:idx_start + window])
            im.set_data(fft_data)

        return time_lines + [im]

    ani = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=interval, blit=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
