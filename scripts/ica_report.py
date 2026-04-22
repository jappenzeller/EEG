"""Generate an ICA report for a recording.

Usage:
    python scripts/ica_report.py <input.h5> [--output-dir ./reports]

Produces:
    <stem>_components.png      topography of each component
    <stem>_timeseries.png      time series of each component
    <stem>_labels.txt          classifier labels and confidences
    <stem>_cleaned.h5          HDF5 with ICA applied
    <stem>_before_after.png    PSD before vs after cleaning, per channel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openbci_eeg.realtime.analysis.ica import (
    load_session,
    fit_ica,
    classify_components,
    select_rejects,
    apply_ica,
    save_cleaned,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICA report generator")
    parser.add_argument("input", type=Path, help="Input HDF5 session file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./reports"),
        help="Output directory for report files",
    )
    parser.add_argument(
        "--n-components", type=int, default=15,
        help="Number of ICA components to fit",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Minimum confidence to reject a component",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ica] ERROR: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"[ica] loading {args.input}")
    raw = load_session(args.input)
    print(
        f"[ica] {raw.info['nchan']} channels, {raw.times[-1]:.1f}s, "
        f"{raw.info['sfreq']:.0f} Hz"
    )

    print(f"[ica] fitting {args.n_components}-component ICA (picard)")
    ica = fit_ica(raw, n_components=args.n_components)

    print("[ica] classifying components")
    labels = classify_components(raw, ica)
    rejects = select_rejects(labels, confidence=args.confidence)

    # Write labels file
    labels_path = args.output_dir / f"{stem}_labels.txt"
    with open(labels_path, "w") as f:
        f.write(f"Component classification for {args.input.name}\n")
        f.write(f"Confidence threshold for rejection: {args.confidence}\n\n")
        f.write(f"{'idx':>4}  {'label':<18}  {'prob':>6}  {'action':<8}\n")
        f.write("-" * 42 + "\n")
        for i, (lab, p) in enumerate(
            zip(labels["labels"], labels["y_pred_proba"])
        ):
            action = "REJECT" if i in rejects else "keep"
            f.write(f"{i:>4}  {lab:<18}  {p:>6.2f}  {action:<8}\n")
    print(f"[ica] labels -> {labels_path}")

    # Component topographies
    try:
        fig = ica.plot_components(show=False)
        if isinstance(fig, list):
            for i, f_ in enumerate(fig):
                p = args.output_dir / f"{stem}_components_{i}.png"
                f_.savefig(p, dpi=100, bbox_inches="tight")
                plt.close(f_)
        else:
            p = args.output_dir / f"{stem}_components.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
        print(f"[ica] components -> {stem}_components*.png")
    except Exception as e:
        print(f"[ica] WARNING: could not plot components: {e}")

    # Component time series — first minute only
    try:
        raw_first_min = raw.copy().crop(tmax=min(60, raw.times[-1]))
        fig = ica.plot_sources(raw_first_min, show=False)
        p = args.output_dir / f"{stem}_timeseries.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[ica] sources -> {stem}_timeseries.png")
    except Exception as e:
        print(f"[ica] WARNING: could not plot sources: {e}")

    # Apply and save cleaned
    ica.exclude = rejects
    raw_clean = apply_ica(raw, ica)
    excluded_labels = [labels["labels"][i] for i in rejects]
    cleaned_path = args.output_dir / f"{stem}_cleaned.h5"
    save_cleaned(
        raw_clean, cleaned_path, args.input,
        ica_info={
            "n_excluded": len(rejects),
            "excluded_labels": excluded_labels,
        },
    )
    print(f"[ica] cleaned -> {cleaned_path}")

    # Before/after PSD comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        raw.compute_psd(fmax=50).plot(axes=axes[0], show=False)
        axes[0].set_title("Before ICA")
        raw_clean.compute_psd(fmax=50).plot(axes=axes[1], show=False)
        axes[1].set_title(
            f"After ICA ({len(rejects)} components removed)"
        )
        fig.tight_layout()
        p = args.output_dir / f"{stem}_before_after.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[ica] psd comparison -> {stem}_before_after.png")
    except Exception as e:
        print(f"[ica] WARNING: could not plot PSD comparison: {e}")

    print(
        f"\n[ica] summary: {len(rejects)}/{args.n_components} "
        f"components rejected"
    )
    for i in rejects:
        print(
            f"  {i}: {labels['labels'][i]} "
            f"(p={labels['y_pred_proba'][i]:.2f})"
        )


if __name__ == "__main__":
    main()
