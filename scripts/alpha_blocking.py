"""Alpha blocking validation — post-session analysis.

Loads an HDF5 session with eyes-open/closed markers, segments by phase,
computes PSD on O1/O2, and reports alpha power ratios.

Usage:
    python scripts/alpha_blocking.py data/raw/scalp_test.h5

Pass criterion: closed_alpha_power > 2.0 * open_alpha_power on at least
one of O1/O2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch


# Marker IDs from eyes_minimal.py
MARKER_SESSION_START = 150
MARKER_EYES_OPEN = 151
MARKER_EYES_CLOSED = 152
MARKER_SESSION_END = 153

# Alpha band
ALPHA_LOW = 8.0
ALPHA_HIGH = 13.0

# Target channels (O1, O2) — indices depend on channel order
# Standard montage: O1=CH15 (index 14), O2=CH16 (index 15)


def load_session(path: str | Path) -> dict:
    """Load HDF5 session and extract EEG + markers."""
    with h5py.File(path, "r") as f:
        eeg = f["raw"][:]
        timestamps = f["timestamps"][:]
        sr = int(f.attrs["sample_rate"])
        n_ch = int(f.attrs["n_channels"])

        ch_names_raw = f.attrs.get("channel_names", None)
        if ch_names_raw is not None:
            ch_names = [s.decode("utf-8") if isinstance(s, bytes) else s
                        for s in ch_names_raw]
        else:
            ch_names = [f"CH{i+1}" for i in range(n_ch)]

        markers = []
        if "markers" in f and "sample_index" in f["markers"]:
            sample_idx = f["markers"]["sample_index"][:]
            marker_id = f["markers"]["marker_id"][:]
            labels = f["markers"]["label"][:]
            for i in range(len(sample_idx)):
                label = labels[i]
                if isinstance(label, bytes):
                    label = label.decode("utf-8")
                markers.append({
                    "sample_index": int(sample_idx[i]),
                    "marker_id": int(marker_id[i]),
                    "label": label,
                })

        notes = f.attrs.get("session_notes", "")
        sq_profile = f.attrs.get("sq_profile", "unknown")

    return {
        "eeg": eeg,
        "timestamps": timestamps,
        "sample_rate": sr,
        "channel_names": ch_names,
        "markers": markers,
        "session_notes": notes,
        "sq_profile": sq_profile,
    }


def segment_by_markers(session: dict) -> dict[str, tuple[int, int]]:
    """Extract sample ranges for each phase from markers."""
    markers = session["markers"]
    segments = {}

    for i, m in enumerate(markers):
        mid = m["marker_id"]
        start = m["sample_index"]

        # Find end: next marker's sample index, or end of data
        if i + 1 < len(markers):
            end = markers[i + 1]["sample_index"]
        else:
            end = session["eeg"].shape[1]

        if mid == MARKER_EYES_OPEN:
            label = m["label"]
            segments[label] = (start, end)
        elif mid == MARKER_EYES_CLOSED:
            segments["eyes_closed"] = (start, end)

    return segments


def compute_alpha_power(eeg_channel: np.ndarray, sr: int) -> dict:
    """Compute PSD and extract alpha band power for a single channel."""
    nperseg = min(2 * sr, len(eeg_channel))  # 2-second windows
    freqs, psd = welch(eeg_channel, fs=sr, nperseg=nperseg)

    alpha_mask = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    alpha_power = np.mean(psd[alpha_mask])
    total_power = np.mean(psd)

    # Find peak in alpha band
    alpha_psd = psd[alpha_mask]
    alpha_freqs = freqs[alpha_mask]
    peak_idx = np.argmax(alpha_psd)
    peak_freq = alpha_freqs[peak_idx]
    peak_power = alpha_psd[peak_idx]

    return {
        "alpha_power": float(alpha_power),
        "total_power": float(total_power),
        "alpha_ratio": float(alpha_power / total_power) if total_power > 0 else 0,
        "peak_freq": float(peak_freq),
        "peak_power": float(peak_power),
        "freqs": freqs,
        "psd": psd,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/alpha_blocking.py <session.h5>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...")
    session = load_session(path)

    sr = session["sample_rate"]
    ch_names = session["channel_names"]
    eeg = session["eeg"]
    n_ch, n_samp = eeg.shape

    print(f"  {n_ch} channels, {n_samp} samples ({n_samp/sr:.1f}s) @ {sr} Hz")
    print(f"  Profile: {session['sq_profile']}")
    if session["session_notes"]:
        print(f"  Notes: {session['session_notes']}")
    print(f"  Markers: {len(session['markers'])}")

    for m in session["markers"]:
        print(f"    [{m['marker_id']}] {m['label']} @ sample {m['sample_index']}")

    # Segment
    segments = segment_by_markers(session)
    if not segments:
        print("\nNo eyes-open/closed markers found. Run the Alpha Check paradigm first.")
        sys.exit(1)

    print(f"\nSegments:")
    for name, (start, end) in segments.items():
        print(f"  {name}: samples {start}-{end} ({(end-start)/sr:.1f}s)")

    # Find O1 and O2
    target_channels = {}
    for i, name in enumerate(ch_names):
        if name in ("O1", "O2"):
            target_channels[name] = i

    if not target_channels:
        print("\nO1/O2 not found in channel names. Available:", ch_names)
        sys.exit(1)

    print(f"\nTarget channels: {list(target_channels.keys())}")

    # Compute alpha power per segment per channel
    print("\n" + "=" * 60)
    print("ALPHA POWER ANALYSIS")
    print("=" * 60)

    results = {}
    for seg_name, (start, end) in segments.items():
        results[seg_name] = {}
        for ch_name, ch_idx in target_channels.items():
            data = eeg[ch_idx, start:end]
            # Convert to volts if in microvolts (check magnitude)
            r = compute_alpha_power(data, sr)
            results[seg_name][ch_name] = r
            print(f"\n  {seg_name} / {ch_name}:")
            print(f"    Alpha power (8-13 Hz): {r['alpha_power']:.4f}")
            print(f"    Total power:           {r['total_power']:.4f}")
            print(f"    Alpha ratio:           {r['alpha_ratio']:.3f}")
            print(f"    Peak frequency:        {r['peak_freq']:.1f} Hz")

    # Alpha blocking ratio
    print("\n" + "=" * 60)
    print("ALPHA BLOCKING RATIO (closed / open)")
    print("=" * 60)

    passed = False
    open_segments = [k for k in segments if "open" in k]
    closed_segments = [k for k in segments if "closed" in k]

    if not open_segments or not closed_segments:
        print("  Need both open and closed segments for ratio.")
    else:
        for ch_name in target_channels:
            open_power = np.mean([
                results[seg][ch_name]["alpha_power"] for seg in open_segments
            ])
            closed_power = np.mean([
                results[seg][ch_name]["alpha_power"] for seg in closed_segments
            ])
            ratio = closed_power / open_power if open_power > 0 else 0
            status = "PASS" if ratio > 2.0 else "FAIL"
            if ratio > 2.0:
                passed = True
            print(f"  {ch_name}: open={open_power:.4f}  closed={closed_power:.4f}  "
                  f"ratio={ratio:.2f}x  [{status}]")

    print(f"\n{'PASS' if passed else 'FAIL'}: "
          f"{'At least one channel shows alpha blocking > 2x' if passed else 'No channel reached 2x alpha blocking ratio'}")


if __name__ == "__main__":
    main()
