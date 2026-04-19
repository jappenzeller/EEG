"""Signal quality session analysis — multi-dimensional channel comparison.

Parses --sq-log output and generates:
    1. Heatmaps (time x channel) for each metric
    2. Status timeline (red/yellow/green over time per channel)
    3. Per-metric time series overlay
    4. Channel correlation during touch events
    5. 3D scatter (std vs 60Hz vs rail, colored by status)
    6. Radar/polar summary per channel

Usage:
    python scripts/sq_analysis.py <sq-log-output-file> [--output-dir charts/]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np

# Match thresholds from signal_quality.py
RAIL_THRESHOLD = 500.0
GREEN_MAX_PCT = 5.0
YELLOW_MAX_PCT = 50.0
FLATLINE_STD = 1.0
LOW_ACTIVITY_STD = 5.0
HIGH_ACTIVITY_STD = 50.0
SATURATION_STD = 500.0
LINE_GREEN_MAX = 0.30
LINE_YELLOW_MAX = 0.85

CHANNELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
            "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
CH_BOARD = ["Cyton"] * 8 + ["Daisy"] * 8
WIRE_COLORS = {
    0: "#9AA4AE", 1: "#8A2BE2", 2: "#1E90FF", 3: "#2E8B57",
    4: "#FFD700", 5: "#FF8C00", 6: "#D9342B", 7: "#8B5A2B",
    8: "#9AA4AE", 9: "#8A2BE2", 10: "#1E90FF", 11: "#2E8B57",
    12: "#FFD700", 13: "#FF8C00", 14: "#D9342B", 15: "#8B5A2B",
}


def parse_sq_log(path: str) -> dict:
    """Parse sq-log output into structured arrays."""
    times = []
    data = {ch: {"rail": [], "std": [], "line": []} for ch in CHANNELS}

    with open(path, encoding="utf-8") as f:
        for line in f:
            tm = re.search(r"t=\s*([\d.]+)", line)
            if not tm:
                continue
            t = float(tm.group(1))
            row_complete = True
            for ch in CHANNELS:
                mx = re.search(
                    ch + r"\s+r=\s*(\S+)\s+s=\s*(\S+)\s+l=\s*(\S+)", line
                )
                if mx:
                    data[ch]["rail"].append(float(mx.group(1)))
                    data[ch]["std"].append(float(mx.group(2)))
                    data[ch]["line"].append(
                        float(mx.group(3).rstrip("%")) / 100.0
                    )
                else:
                    row_complete = False
            if row_complete:
                times.append(t)

    t = np.array(times)
    n = len(t)
    rail = np.array([data[ch]["rail"][:n] for ch in CHANNELS])
    std = np.array([data[ch]["std"][:n] for ch in CHANNELS])
    line = np.array([data[ch]["line"][:n] for ch in CHANNELS])

    # Compute status (0=green, 1=yellow, 2=red)
    status = np.zeros_like(rail, dtype=int)
    yellow = (
        (rail >= GREEN_MAX_PCT)
        | (std < LOW_ACTIVITY_STD)
        | (std > HIGH_ACTIVITY_STD)
        | (line > LINE_GREEN_MAX)
    )
    status[yellow] = 1
    red = (
        (rail >= YELLOW_MAX_PCT)
        | (std < FLATLINE_STD)
        | (std > SATURATION_STD)
        | (line > LINE_YELLOW_MAX)
    )
    status[red] = 2

    return {"t": t, "rail": rail, "std": std, "line": line, "status": status}


def plot_heatmaps(d: dict, out: Path):
    """Three heatmaps: rail%, std, 60Hz ratio — all channels x time."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle("Signal Quality Heatmaps (channel x time)", fontsize=16, y=0.98)

    metrics = [
        ("rail", d["rail"], "Rail %", "YlOrRd", 0, 100),
        ("std", np.clip(d["std"], 0, 2000), "Std (uV, clipped 2000)", "YlOrRd", 0, 2000),
        ("line", d["line"] * 100, "60 Hz Ratio %", "YlOrRd", 0, 100),
    ]

    for ax, (name, data, title, cmap, vmin, vmax) in zip(axes, metrics):
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[d["t"][0], d["t"][-1], len(CHANNELS) - 0.5, -0.5],
            interpolation="nearest",
        )
        ax.set_yticks(range(len(CHANNELS)))
        ax.set_yticklabels([f"CH{i+1} {ch}" for i, ch in enumerate(CHANNELS)], fontsize=8)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Board separator
        ax.axhline(7.5, color="white", linewidth=2, linestyle="--")
        ax.text(d["t"][0] + 1, 3.5, "CYTON", color="white", fontsize=8, fontweight="bold", va="center")
        ax.text(d["t"][0] + 1, 11.5, "DAISY", color="white", fontsize=8, fontweight="bold", va="center")

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / 'heatmaps.png'}")


def plot_status_timeline(d: dict, out: Path):
    """Status strip chart — green/yellow/red per channel over time."""
    status_cmap = mcolors.ListedColormap(["#34C759", "#FFCC00", "#FF453A"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, status_cmap.N)

    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(
        d["status"], aspect="auto", cmap=status_cmap, norm=norm,
        extent=[d["t"][0], d["t"][-1], len(CHANNELS) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels([f"CH{i+1} {ch}" for i, ch in enumerate(CHANNELS)], fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_title("Channel Status Over Time (green=ok, yellow=check, red=bad)", fontsize=14)
    ax.axhline(7.5, color="black", linewidth=2, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#34C759", label="GREEN (good)"),
        Patch(facecolor="#FFCC00", label="YELLOW (contact/noisy ref)"),
        Patch(facecolor="#FF453A", label="RED (disconnected/railed)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "status_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / 'status_timeline.png'}")


def plot_metric_timeseries(d: dict, out: Path):
    """Overlaid time series per metric, colored by wire color."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle("Per-Channel Metrics Over Time", fontsize=16, y=0.98)

    metrics = [
        ("rail", d["rail"], "Rail %"),
        ("std", np.clip(d["std"], 0, 2000), "Std (uV, clipped 2000)"),
        ("line", d["line"] * 100, "60 Hz %"),
    ]

    for ax, (name, data, ylabel) in zip(axes, metrics):
        for i, ch in enumerate(CHANNELS):
            alpha = 0.8 if i < 8 else 0.5
            ls = "-" if i < 8 else "--"
            ax.plot(d["t"], data[i], color=WIRE_COLORS[i], alpha=alpha,
                    linewidth=1.2, linestyle=ls, label=f"CH{i+1} {ch}")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.2)

    axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(out / "metric_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / 'metric_timeseries.png'}")


def plot_3d_scatter(d: dict, out: Path):
    """3D scatter: std vs 60Hz vs rail, one point per (channel, time), colored by status."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors_map = {0: "#34C759", 1: "#FFCC00", 2: "#FF453A"}
    status_flat = d["status"].flatten()
    colors = [colors_map[s] for s in status_flat]

    std_flat = np.clip(d["std"].flatten(), 0, 2000)
    line_flat = d["line"].flatten() * 100
    rail_flat = d["rail"].flatten()

    ax.scatter(std_flat, line_flat, rail_flat, c=colors, alpha=0.3, s=8, edgecolors="none")

    ax.set_xlabel("Std (uV)", fontsize=11)
    ax.set_ylabel("60 Hz %", fontsize=11)
    ax.set_zlabel("Rail %", fontsize=11)
    ax.set_title("Signal Quality Space (all channels x time)", fontsize=14)

    # Add threshold planes as wireframes
    # Green box boundaries
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # std=50 plane
    xx = [50, 50, 50, 50]
    yy = [0, 30, 30, 0]
    zz = [0, 0, 5, 5]
    verts = [list(zip(xx, yy, zz))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="green", edgecolor="green"))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#34C759", label="GREEN"),
        Patch(facecolor="#FFCC00", label="YELLOW"),
        Patch(facecolor="#FF453A", label="RED"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "3d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / '3d_scatter.png'}")


def plot_channel_radar(d: dict, out: Path):
    """Radar chart per channel showing median metrics and % time in each status."""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 8, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Per-Channel Summary (median metrics + status distribution)", fontsize=14, y=0.98)

    for i, ch in enumerate(CHANNELS):
        row, col = divmod(i, 8)
        ax = fig.add_subplot(gs[row, col])

        n_total = len(d["t"])
        pct_green = np.sum(d["status"][i] == 0) / n_total * 100
        pct_yellow = np.sum(d["status"][i] == 1) / n_total * 100
        pct_red = np.sum(d["status"][i] == 2) / n_total * 100

        # Stacked bar for status distribution
        ax.barh(0, pct_green, color="#34C759", height=0.6)
        ax.barh(0, pct_yellow, left=pct_green, color="#FFCC00", height=0.6)
        ax.barh(0, pct_red, left=pct_green + pct_yellow, color="#FF453A", height=0.6)

        med_std = np.median(d["std"][i])
        med_line = np.median(d["line"][i]) * 100
        med_rail = np.median(d["rail"][i])
        min_std = np.min(d["std"][i][d["std"][i] > 1]) if np.any(d["std"][i] > 1) else 0

        ax.set_xlim(0, 100)
        ax.set_ylim(-1, 1.5)
        ax.set_yticks([])
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"], fontsize=6)

        title_color = WIRE_COLORS[i]
        ax.set_title(f"CH{i+1} {ch}", fontsize=9, fontweight="bold", color=title_color)

        # Stats text
        ax.text(50, 0.9, f"std:{med_std:.0f} 60Hz:{med_line:.0f}% r:{med_rail:.0f}%",
                ha="center", va="bottom", fontsize=6, color="#666666")
        ax.text(50, -0.6, f"best std:{min_std:.0f}",
                ha="center", va="top", fontsize=6, color="#888888")

        ax.set_facecolor("#0a0e14")
        ax.tick_params(colors="#888888", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    fig.patch.set_facecolor("#0a0e14")
    fig.savefig(out / "channel_summary.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0e14")
    plt.close(fig)
    print(f"  saved {out / 'channel_summary.png'}")


def plot_touch_correlation(d: dict, out: Path):
    """Correlation matrix of std across channels — shows which channels move together."""
    # Use std as the primary signal
    std_clipped = np.clip(d["std"], 0, 2000)

    # Correlation matrix
    corr = np.corrcoef(std_clipped)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"CH{i+1}\n{ch}" for i, ch in enumerate(CHANNELS)], fontsize=7, rotation=45)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f"CH{i+1} {ch}" for i, ch in enumerate(CHANNELS)], fontsize=7)
    ax.set_title("Std Correlation (which channels move together)", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.axhline(7.5, color="black", linewidth=1)
    ax.axvline(7.5, color="black", linewidth=1)

    # Std change rate — derivative shows when touches happen
    ax2 = axes[1]
    std_diff = np.abs(np.diff(std_clipped, axis=1))
    total_change = std_diff.sum(axis=0)
    ax2.fill_between(d["t"][1:], total_change, alpha=0.6, color="#4a9eff")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Total std change rate (all channels)", fontsize=11)
    ax2.set_title("Touch Event Detection (spikes = contact changes)", fontsize=12)
    ax2.grid(True, alpha=0.2)

    # Mark likely touch events (top 10% of change rate)
    threshold = np.percentile(total_change, 90)
    touch_times = d["t"][1:][total_change > threshold]
    for tt in touch_times:
        ax2.axvline(tt, color="#FF453A", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out / "touch_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / 'touch_correlation.png'}")


def plot_cyton_vs_daisy(d: dict, out: Path):
    """Compare Cyton (CH1-8) vs Daisy (CH9-16) behavior over time."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle("Cyton vs Daisy Board Comparison", fontsize=14, y=0.98)

    cyton = slice(0, 8)
    daisy = slice(8, 16)

    metrics = [
        ("Median Std (uV)", np.clip(d["std"], 0, 2000)),
        ("Median 60Hz %", d["line"] * 100),
        ("Median Rail %", d["rail"]),
    ]

    for ax, (ylabel, data) in zip(axes, metrics):
        cyton_med = np.median(data[cyton], axis=0)
        daisy_med = np.median(data[daisy], axis=0)
        cyton_q25 = np.percentile(data[cyton], 25, axis=0)
        cyton_q75 = np.percentile(data[cyton], 75, axis=0)
        daisy_q25 = np.percentile(data[daisy], 25, axis=0)
        daisy_q75 = np.percentile(data[daisy], 75, axis=0)

        ax.plot(d["t"], cyton_med, color="#4a9eff", linewidth=2, label="Cyton (CH1-8)")
        ax.fill_between(d["t"], cyton_q25, cyton_q75, color="#4a9eff", alpha=0.2)
        ax.plot(d["t"], daisy_med, color="#ff6b6b", linewidth=2, label="Daisy (CH9-16)")
        ax.fill_between(d["t"], daisy_q25, daisy_q75, color="#ff6b6b", alpha=0.2)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "cyton_vs_daisy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / 'cyton_vs_daisy.png'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/sq_analysis.py <sq-log-file> [--output-dir charts/]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_dir = Path("charts")
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        out_dir = Path(sys.argv[idx + 1])

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {log_path}...")
    d = parse_sq_log(log_path)
    print(f"  {len(d['t'])} timepoints, {len(CHANNELS)} channels")
    print()

    print("Generating charts...")
    plot_heatmaps(d, out_dir)
    plot_status_timeline(d, out_dir)
    plot_metric_timeseries(d, out_dir)
    plot_3d_scatter(d, out_dir)
    plot_channel_radar(d, out_dir)
    plot_touch_correlation(d, out_dir)
    plot_cyton_vs_daisy(d, out_dir)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} charts in {out_dir}/")


if __name__ == "__main__":
    main()
