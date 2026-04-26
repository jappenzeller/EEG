"""Three-panel cross-modal visualization for a single time window.

Combines spatial (16-channel topomap), spectral (Welch PSD), and
projected-source (low-resolution dipole estimate) views.

Source localization caveats:
  - Uses MNE's fsaverage template head model. No subject MRI.
  - Minimum-norm estimate with default regularization.
  - With 16 channels, effective spatial resolution ~5 cm (lobe-level).
  - Cortical activation magnitudes are NOT calibrated to physical units.
    Treat as relative spatial pattern, not quantified neural activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

RESOLUTION_CAVEAT = (
    "Source estimate: 16 ch + template head, ~5 cm resolution. "
    "Spatial pattern is relative; values are not calibrated to neural current."
)


@dataclass
class Triptych:
    """A rendered triptych for one (raw, time-window, band) combination."""
    figure: plt.Figure
    band_powers: dict[str, float]
    psd_data: dict[str, tuple[np.ndarray, np.ndarray]]
    source_estimate: Optional[mne.SourceEstimate]


def render_triptych(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    band: tuple[float, float] = (8.0, 12.0),
    psd_channels: Optional[Sequence[str]] = None,
    title: str = "",
    output_path: Optional[Path | str] = None,
) -> Triptych:
    """Render the three-panel view for a window of a recording.

    Args:
        raw: MNE Raw object
        tmin, tmax: time window in seconds
        band: frequency band for the spatial heatmap (Hz, Hz)
        psd_channels: channels to plot in spectral panel (default: occipital)
        title: optional figure title
        output_path: if given, save PNG to this path

    Returns:
        Triptych dataclass with figure and computed quantities.
    """
    raw_win = raw.copy().crop(tmin=tmin, tmax=tmax)

    band_powers = _band_power_per_channel(raw_win, band)

    if psd_channels is None:
        psd_channels = _default_psd_channels(raw_win.ch_names)
    psd_data = _compute_psd(raw_win, psd_channels)

    source_estimate = _compute_source_estimate(raw_win, band)

    fig = _make_figure(
        raw=raw_win,
        band=band,
        band_powers=band_powers,
        psd_channels=psd_channels,
        psd_data=psd_data,
        source_estimate=source_estimate,
        title=title or f"Triptych: {tmin:.1f}\u2013{tmax:.1f} s, "
                       f"{band[0]:.0f}\u2013{band[1]:.0f} Hz",
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return Triptych(
        figure=fig,
        band_powers=band_powers,
        psd_data=psd_data,
        source_estimate=source_estimate,
    )


def _band_power_per_channel(
    raw: mne.io.BaseRaw,
    band: tuple[float, float],
) -> dict[str, float]:
    """Welch PSD per channel, integrated over the band."""
    fmin, fmax = band
    psd_obj = raw.compute_psd(
        method="welch",
        fmin=max(0.5, fmin - 2),
        fmax=fmax + 5,
        n_fft=min(int(raw.info["sfreq"] * 2), raw.n_times),
        verbose="ERROR",
    )
    psds, freqs = psd_obj.get_data(return_freqs=True)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return {ch: 0.0 for ch in raw.ch_names}
    band_power = psds[:, band_mask].mean(axis=1)
    return dict(zip(raw.ch_names, band_power.tolist()))


def _default_psd_channels(all_channels: list[str]) -> list[str]:
    """Pick sensible PSD channels, preferring occipital."""
    preferred = ["Oz", "O2", "O1", "Pz", "Cz"]
    chosen = [ch for ch in preferred if ch in all_channels][:3]
    if not chosen:
        chosen = all_channels[:3]
    return chosen


def _compute_psd(
    raw: mne.io.BaseRaw,
    channels: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Welch PSD for each named channel."""
    fs = raw.info["sfreq"]
    nperseg = min(int(fs * 2), raw.n_times)
    out = {}
    for ch in channels:
        if ch not in raw.ch_names:
            continue
        idx = raw.ch_names.index(ch)
        data = raw.get_data(picks=[idx])[0]
        freqs, psd = welch(data, fs=fs, nperseg=nperseg)
        out[ch] = (freqs, psd)
    return out


def _compute_source_estimate(
    raw: mne.io.BaseRaw,
    band: tuple[float, float],
) -> Optional[mne.SourceEstimate]:
    """Source localize via MNE inverse with fsaverage template.

    Returns None if any step fails. Failure is logged but doesn't crash.
    """
    try:
        ch_pos = (
            raw.get_montage().get_positions()["ch_pos"]
            if raw.get_montage()
            else {}
        )
        positioned = [ch for ch in raw.ch_names if ch in ch_pos]
        if len(positioned) < 8:
            return None

        raw_band = raw.copy().pick(positioned).filter(
            l_freq=band[0], h_freq=band[1], verbose="ERROR",
        )
        raw_band.set_eeg_reference(
            "average", projection=True, verbose="ERROR",
        )

        fs_dir = mne.datasets.fetch_fsaverage(verbose="ERROR")
        src_path = Path(fs_dir) / "bem" / "fsaverage-ico-5-src.fif"
        bem_path = (
            Path(fs_dir) / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        )
        trans = "fsaverage"

        fwd = mne.make_forward_solution(
            raw_band.info, trans=trans, src=str(src_path),
            bem=str(bem_path), eeg=True, meg=False, verbose="ERROR",
        )

        noise_cov = mne.make_ad_hoc_cov(raw_band.info, verbose="ERROR")
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw_band.info, fwd, noise_cov, loose=0.2, depth=0.8,
            verbose="ERROR",
        )

        analytic = raw_band.copy().apply_hilbert(
            envelope=True, verbose="ERROR",
        )
        evoked = mne.EvokedArray(
            analytic.get_data().mean(axis=1, keepdims=True),
            analytic.info, tmin=0,
        )
        stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator, lambda2=1.0 / 9.0, method="MNE",
            verbose="ERROR",
        )
        return stc

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "source estimate failed: %s", e,
        )
        return None


def _make_figure(
    raw: mne.io.BaseRaw,
    band: tuple[float, float],
    band_powers: dict[str, float],
    psd_channels: Sequence[str],
    psd_data: dict[str, tuple[np.ndarray, np.ndarray]],
    source_estimate: Optional[mne.SourceEstimate],
    title: str,
) -> plt.Figure:
    """Compose the three panels into one figure."""
    fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.0])

    ax_topo = fig.add_subplot(gs[0, 0])
    _draw_topomap(ax_topo, raw, band_powers, band)

    ax_psd = fig.add_subplot(gs[0, 1])
    _draw_psd(ax_psd, psd_channels, psd_data, band)

    ax_src = fig.add_subplot(gs[0, 2])
    _draw_source(ax_src, source_estimate, band)

    fig.suptitle(title, fontsize=12)
    fig.text(
        0.5, -0.02, RESOLUTION_CAVEAT,
        ha="center", va="top", fontsize=8, style="italic", color="#555",
    )
    return fig


def _draw_topomap(
    ax: plt.Axes,
    raw: mne.io.BaseRaw,
    band_powers: dict[str, float],
    band: tuple[float, float],
) -> None:
    values = np.array([band_powers.get(ch, 0.0) for ch in raw.ch_names])
    try:
        mne.viz.plot_topomap(
            values, raw.info, axes=ax, show=False, cmap="viridis",
            sensors=True, contours=4,
        )
    except Exception:
        ax.bar(raw.ch_names, values)
        ax.set_xticklabels(raw.ch_names, rotation=45, ha="right", fontsize=7)
    ax.set_title(
        f"{band[0]:.0f}\u2013{band[1]:.0f} Hz power\n(spatial)", fontsize=10,
    )


def _draw_psd(
    ax: plt.Axes,
    channels: Sequence[str],
    psd_data: dict[str, tuple[np.ndarray, np.ndarray]],
    band: tuple[float, float],
) -> None:
    for ch, (freqs, psd) in psd_data.items():
        ax.semilogy(freqs, psd, label=ch, linewidth=1.2)
    ax.axvspan(
        *band, alpha=0.15, color="green",
        label=f"{band[0]:.0f}\u2013{band[1]:.0f} Hz",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V\u00b2/Hz)")
    ax.set_xlim(0, min(50, ax.get_xlim()[1]))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Spectral content\n(temporal)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def _draw_source(
    ax: plt.Axes,
    stc: Optional[mne.SourceEstimate],
    band: tuple[float, float],
) -> None:
    """Render source estimate as 2D top-down projection."""
    if stc is None:
        ax.text(
            0.5, 0.5,
            "source estimate\nunavailable\n\n"
            "(insufficient channels\nor fsaverage not loaded)",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "Predicted spatial activation\n(source-localized)", fontsize=10,
        )
        return

    data = stc.data[:, 0]
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose="ERROR")
        src = mne.read_source_spaces(
            str(Path(fs_dir) / "bem" / "fsaverage-ico-5-src.fif"),
            verbose="ERROR",
        )
        lh_pos = src[0]["rr"][stc.vertices[0]]
        rh_pos = src[1]["rr"][stc.vertices[1]]
        positions = np.concatenate([lh_pos, rh_pos], axis=0)
    except Exception:
        ax.text(
            0.5, 0.5, "source positions unavailable",
            ha="center", va="center", transform=ax.transAxes, color="#888",
        )
        return

    sc = ax.scatter(
        positions[:, 0], -positions[:, 1],
        c=data, cmap="hot", s=4, alpha=0.7,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("\u2190 L          R \u2192", fontsize=8)
    ax.set_ylabel("\u2190 post       ant \u2192", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Predicted spatial activation\n(source-localized, ~5 cm res)",
        fontsize=10,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="MNE est.")
