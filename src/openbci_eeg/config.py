"""
Configuration management for OpenBCI EEG pipeline.

Loads settings from YAML config files with sensible defaults.
Config hierarchy: defaults < config file < environment variables < CLI args.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from openbci_eeg import BOARD_SAMPLE_RATE, CHANNEL_NAMES, RING_ORDER


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoardConfig:
    """OpenBCI Cyton+Daisy board settings."""
    serial_port: Optional[str] = None  # None = auto-detect
    board_id: int = 2                  # CYTON_DAISY_BOARD
    sample_rate: int = BOARD_SAMPLE_RATE
    channel_names: list[str] = field(default_factory=lambda: list(CHANNEL_NAMES))
    ring_order: list[str] = field(default_factory=lambda: list(RING_ORDER))
    buffer_size: int = 45000           # ~6 min ring buffer
    log_level: int = 0                 # 0=off, 1=trace, 2=debug


@dataclass
class PreprocessConfig:
    """Signal preprocessing parameters."""
    notch_freq: float = 60.0           # Hz (US line noise)
    notch_quality: float = 30.0        # Q factor
    bandpass_low: float = 0.5          # Hz
    bandpass_high: float = 50.0        # Hz
    filter_method: str = "fir"         # "fir" or "iir"
    artifact_threshold_uv: float = 150.0  # µV, reject epochs above this
    ica_n_components: int = 15         # For artifact removal


@dataclass
class PNConfig:
    """Positive-Negative neuron model parameters."""
    lambda_a: float = 0.1             # Excitatory decay rate
    lambda_c: float = 0.05            # Inhibitory growth rate
    rms_window_sec: float = 0.1       # 100ms RMS envelope window
    initial_a: float = 0.5            # Initial excitatory state
    initial_c: float = 0.5            # Initial inhibitory state
    solver: str = "euler"             # "euler" or "rk4"


@dataclass
class AWSConfig:
    """AWS infrastructure settings."""
    bucket: str = "qdnu-eeg-data"
    region: str = "us-east-1"
    raw_prefix: str = "raw"
    processed_prefix: str = "processed"
    results_prefix: str = "results"
    lambda_function: str = "qdnu-process-eeg"
    dynamodb_table: str = "qdnu-eeg-sessions"


@dataclass
class SessionConfig:
    """Recording session parameters."""
    subject_id: str = "S000"
    session_type: str = "resting"      # resting, oddball, sternberg, meditation
    duration_sec: float = 300.0        # Default 5 min
    data_dir: Path = field(default_factory=lambda: Path("data"))
    auto_upload: bool = False          # Upload to S3 after recording


@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""
    board: BoardConfig = field(default_factory=BoardConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    pn: PNConfig = field(default_factory=PNConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str | Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    If no path given, returns defaults. YAML keys map directly to
    dataclass fields (board.serial_port, preprocess.bandpass_low, etc.).
    """
    config = PipelineConfig()

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            _apply_dict(config.board, raw.get("board", {}))
            _apply_dict(config.preprocess, raw.get("preprocess", {}))
            _apply_dict(config.pn, raw.get("pn", {}))
            _apply_dict(config.aws, raw.get("aws", {}))
            _apply_dict(config.session, raw.get("session", {}))

    return config


def save_config(config: PipelineConfig, path: str | Path) -> None:
    """Save current configuration to YAML file."""
    import dataclasses

    def _to_dict(obj: object) -> dict:
        d = dataclasses.asdict(obj)
        # Convert Path objects to strings
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    data = {
        "board": _to_dict(config.board),
        "preprocess": _to_dict(config.preprocess),
        "pn": _to_dict(config.pn),
        "aws": _to_dict(config.aws),
        "session": _to_dict(config.session),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _apply_dict(dataclass_obj: object, overrides: dict) -> None:
    """Apply dictionary values to dataclass fields."""
    for key, value in overrides.items():
        if hasattr(dataclass_obj, key):
            field_type = type(getattr(dataclass_obj, key))
            if field_type is Path:
                value = Path(value)
            setattr(dataclass_obj, key, value)
