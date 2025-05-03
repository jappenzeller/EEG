import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Import necessary functions from your existing code
from SeizureDetectionPipelines import load_subject_sequences, fft_features


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side FFT line & heatmap animation"
    )
    parser.add_argument('--data-root', default='H:/Data/PythonDNU/EEG/DataKaggle',
                        help='Root directory with subject folders')
    parser.add_argument('--subject', default='Dog_1',
                        help='Subject folder name')
    parser.add_argument('--window-size', type=int, default=400,
                        help='Samples per segment (default: 400 for 1s at 400Hz)')
    parser.add_argument('--output', default='fft_animation.mp4',
                        help='Output video filename')
    parser.add_argument('--interval', type=int, default=1000,
                        help='Interval between frames in ms (default: 1000)')
    args = parser.parse_args()

    # Load continuous ictal data
    ictal, _ = load_subject_sequences(args.data_root, args.subject)
    n_channels, n_samples = ictal.shape
    n_frames = n_samples // args.window_size
    window = args.window_size

    # Prepare figure and axes
    fig, (ax_line, ax_heat) = plt.subplots(1, 2, figsize=(12, 6))

    # Storage for dynamic elements
    lines = []
    im = None
    freq_axis = None

    def init():
        nonlocal lines, im, freq_axis
        # Use first segment to initialize plots
        sample = ictal[:, :window]
        fft_out = fft_features(sample)
        n_bins = fft_out.shape[1]
        freq_axis = np.arange(1, n_bins + 1)

        # Line plot init
        ax_line.clear()
        lines = []
        for ch in range(n_channels):
            (line,) = ax_line.plot(freq_axis, fft_out[ch], lw=1)
            lines.append(line)
        ax_line.set_xlim(freq_axis[0], freq_axis[-1])
        ax_line.set_ylim(fft_out.min(), fft_out.max())
        ax_line.set_xlabel('Frequency (Hz)')
        ax_line.set_ylabel('Log10 Magnitude')
        ax_line.set_title('FFT Magnitudes per Channel')

        # Heatmap init with correct clim
        ax_heat.clear()
        im = ax_heat.imshow(
            fft_out,
            aspect='auto',
            origin='lower',
            extent=[freq_axis[0], freq_axis[-1], 0, n_channels],
            vmin=fft_out.min(),
            vmax=fft_out.max()
        )
        ax_heat.set_xlabel('Frequency (Hz)')
        ax_heat.set_ylabel('Channel')
        ax_heat.set_title('FFT Heatmap Across Channels')

        return lines + [im]

    def update(frame_idx):
        start = frame_idx * window
        segment = ictal[:, start:start + window]
        fft_out = fft_features(segment)
        # Update line plots
        for ch, line in enumerate(lines):
            line.set_data(freq_axis, fft_out[ch])
        # Update heatmap data
        im.set_data(fft_out)
        return lines + [im]

    # Build animation
    ani = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=args.interval
    )

    # Save to MP4
    writer = FFMpegWriter(fps=1000/args.interval)
    ani.save(args.output, writer=writer)
    print(f"Saved animation to {args.output}")


if __name__ == '__main__':
    main()
