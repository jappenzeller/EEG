#!/usr/bin/env python3
"""
chart_test.py

Standalone script to test visualization functions in viz.py using real or synthetic EEG data.
"""
import argparse
from pathlib import Path
import numpy as np

from loader import load_segment, load_subject_sequences
from features import fft_features, extract_features
from viz import plot_fft_out, plot_spectrogram

import matplotlib.pyplot as plt
#plt.ion()

def test_fft(segment_path: Path, eps: float = 1e-8):
    """
    Load a segment, compute FFT features, and plot them.
    """
    data = load_segment(segment_path)
    fft_out = fft_features(data, eps)
    plot_fft_out(fft_out)


def test_spectrogram(segment_path: Path, fs: int, channel: int = 1):
    """
    Load a segment and plot its spectrogram for a given channel.
    """
    data = load_segment(segment_path)
    plot_spectrogram(data, fs, channel)


def test_subject_continuous(subject_dir: Path, fs: int, window_sec: float = 0.1):
    """
    Load full concatenated sequences for ictal and interictal data, slice
    one window, and plot its FFT and spectrogram.

    Useful for end-to-end chart testing.
    """
    ictal, interictal = load_subject_sequences(subject_dir.parent, subject_dir.name)
    window_size = int(window_sec * fs)
    # Take first ictal window
    segment = ictal[:, :window_size]
    print(f"Testing FFT on first ictal window of {subject_dir.name}")
    plot_fft_out(fft_features(segment))
    print(f"Testing spectrogram on same window (channel=1)")
    plot_spectrogram(segment, fs, channel=1)


def main():
    parser = argparse.ArgumentParser(
        description='Chart testing script for EEG visualization'
    )
    sub = parser.add_subparsers(dest='command')
    parser.set_defaults(command='subject')

    # FFT test
    p_fft = sub.add_parser('fft', help='Plot FFT of one segment')
    p_fft.add_argument('segment', type=Path, help='Path to .mat EEG segment')
    p_fft.add_argument('--eps', type=float, default=1e-8, help='Epsilon for log FFT')

    # Spectrogram test
    p_spec = sub.add_parser('spec', help='Plot spectrogram of one segment')
    p_spec.add_argument('segment', type=Path, help='Path to .mat EEG segment')
    p_spec.add_argument('--fs', type=int, required=True, help='Sampling rate in Hz')
    p_spec.add_argument('--channel', type=int, default=1, help='Channel index (1-based)')

    # Subject test
    p_sub = sub.add_parser('subject', help='Test charts on continuous subject data')
    p_sub.add_argument('subject_dir', type=Path, help='Path to subject folder' ,default=Path(r'H:/Data/PythonDNU/EEG/DataKaggle/Patient_1'))
    p_sub.add_argument('--fs', type=int, required=True, default=500, help='Sampling rate in Hz')
    p_sub.add_argument('--window', type=float, default=1.0, help='Window length in seconds')

    args = parser.parse_args()

    
    # If no subcommand provided, ensure subject defaults exist
    if args.command == 'subject':
        if not hasattr(args, 'subject_dir'):
            args.subject_dir = Path(r'H:/Data/PythonDNU/EEG/DataKaggle/Patient_1')
        if not hasattr(args, 'fs'):
            # infer fs from subject_dir name
            subj_name = args.subject_dir.name.lower()
            args.fs = 500 if subj_name.startswith('patient') else 400
        if not hasattr(args, 'window'):
            args.window = 1.0

    if args.command == 'fft':
        test_fft(args.segment, eps=args.eps)
    elif args.command == 'spec':
        test_spectrogram(args.segment, fs=args.fs, channel=args.channel)
    elif args.command == 'subject':
        test_subject_continuous(args.subject_dir, fs=args.fs, window_sec=args.window)

if __name__ == '__main__':
    main()
