"""Generate triptych figures from a Lattice HDF5 recording.

Usage:
    python scripts/triptych_report.py <input.h5> \
        [--output-dir reports/triptych] \
        [--band 8 12] \
        [--windows-by-marker] \
        [--manual-windows 0,60 60,120 120,180]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openbci_eeg.realtime.analysis.ica import hdf5_to_mne as load_hdf5_to_mne
from openbci_eeg.realtime.analysis.triptych import render_triptych


def _parse_window_arg(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate triptych figures from an HDF5 recording",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./reports/triptych"),
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=[8.0, 12.0],
        metavar=("FMIN", "FMAX"),
    )
    parser.add_argument(
        "--windows-by-marker", action="store_true", default=True,
    )
    parser.add_argument(
        "--manual-windows", nargs="+", type=_parse_window_arg, default=None,
        help="tmin,tmax pairs (seconds)",
    )
    parser.add_argument("--psd-channels", nargs="+", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"[triptych] loading {args.input}")
    raw = load_hdf5_to_mne(args.input)
    print(
        f"[triptych] {raw.info['nchan']} channels, "
        f"{raw.times[-1]:.1f}s, {raw.info['sfreq']:.0f} Hz"
    )

    windows: list[tuple[float, float, str]] = []
    if args.manual_windows:
        for i, (tmin, tmax) in enumerate(args.manual_windows):
            windows.append(
                (tmin, tmax, f"window_{i:02d}_{tmin:.0f}_{tmax:.0f}")
            )
    elif args.windows_by_marker and len(raw.annotations) >= 2:
        skip = {"session_start", "session_end"}
        ann_list = [
            (a["onset"], a["description"])
            for a in raw.annotations
            if a["description"] not in skip
        ]
        for i, (onset, label) in enumerate(ann_list):
            end = (
                ann_list[i + 1][0] if i + 1 < len(ann_list) else raw.times[-1]
            )
            if end - onset >= 4.0:
                windows.append((onset, end, label))
    else:
        windows.append((0.0, min(raw.times[-1], 60.0), "first_60s"))

    if not windows:
        print(f"[triptych] no usable windows found in {args.input}")
        return

    band = tuple(args.band)
    for tmin, tmax, label in windows:
        out_png = (
            args.output_dir
            / f"{stem}__{label}__band{band[0]:.0f}-{band[1]:.0f}.png"
        )
        print(
            f"[triptych] {label}: t={tmin:.1f}-{tmax:.1f}s -> {out_png.name}"
        )
        render_triptych(
            raw, tmin=tmin, tmax=tmax, band=band,
            psd_channels=args.psd_channels,
            title=(
                f"{stem}  \u2022  {label}  \u2022  "
                f"{tmin:.0f}\u2013{tmax:.0f}s  \u2022  "
                f"{band[0]:.0f}\u2013{band[1]:.0f} Hz"
            ),
            output_path=out_png,
        )

    print(f"[triptych] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
