"""Relabel legacy HDF5 files with correct canonical channel names.

HDF5 files recorded before the current montage have channel names from
BrainFlow's default mapping or a prior montage version. This utility
creates a copy with correct labels.

Usage:
    python -m openbci_eeg.realtime.io.relabel_legacy input.h5 output.h5
    python -m openbci_eeg.realtime.io.relabel_legacy input.h5 output.h5 --target 2026-04-19-midline
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

MONTAGE_VERSIONS: dict[str, dict[int, str]] = {
    "2026-04-19-verified": {
        1: "Fp1", 2: "Fp2", 3: "F3", 4: "F4", 5: "F7", 6: "F8",
        7: "T7", 8: "T8",
        9: "C3", 10: "C4", 11: "Pz", 12: "Cz",
        13: "P3", 14: "P4", 15: "O1", 16: "O2",
    },
    "2026-04-19-midline": {
        1: "Fp1", 2: "Fp2", 3: "F3", 4: "F4", 5: "F7", 6: "F8",
        7: "Fz", 8: "Oz",
        9: "C3", 10: "C4", 11: "Pz", 12: "Cz",
        13: "P3", 14: "P4", 15: "O1", 16: "O2",
    },
}

CURRENT_VERSION = "2026-04-19-midline"


def relabel(
    input_path: str | Path,
    output_path: str | Path,
    target_version: str = CURRENT_VERSION,
) -> None:
    """Copy an HDF5 file and update channel_names to the target montage.

    Does NOT modify the original file. The copy gets:
    - Updated channel_names attr
    - montage_version attr
    - relabeled_from_legacy=True attr
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    if target_version not in MONTAGE_VERSIONS:
        raise ValueError(f"Unknown target version: {target_version}. "
                         f"Available: {list(MONTAGE_VERSIONS.keys())}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)

    positions = MONTAGE_VERSIONS[target_version]

    with h5py.File(output_path, "a") as f:
        n_ch = int(f.attrs.get("n_channels", f["raw"].shape[0]))
        old_version = f.attrs.get("montage_version", "unknown")
        if isinstance(old_version, bytes):
            old_version = old_version.decode()

        canonical = [positions[i] for i in range(1, n_ch + 1)]
        f.attrs["channel_names"] = np.array(canonical, dtype="S32")
        f.attrs["montage_version"] = target_version
        f.attrs["relabeled_from_legacy"] = True
        f.attrs["relabeled_from_version"] = str(old_version)

    print(f"Relabeled {input_path} -> {output_path}")
    print(f"  From version: {old_version}")
    print(f"  To version:   {target_version}")
    print(f"  Channel names: {canonical}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m openbci_eeg.realtime.io.relabel_legacy "
              "<input.h5> <output.h5> [--target VERSION]")
        print(f"Available versions: {list(MONTAGE_VERSIONS.keys())}")
        sys.exit(1)

    target = CURRENT_VERSION
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        target = sys.argv[idx + 1]

    relabel(sys.argv[1], sys.argv[2], target_version=target)


if __name__ == "__main__":
    main()
