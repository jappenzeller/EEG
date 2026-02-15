"""Tests for configuration loading and saving."""

from pathlib import Path

import pytest

from openbci_eeg.config import (
    PipelineConfig,
    BoardConfig,
    PreprocessConfig,
    PNConfig,
    load_config,
    save_config,
)


class TestDefaults:
    def test_pipeline_defaults(self):
        config = PipelineConfig()
        assert config.board.board_id == 2
        assert config.board.sample_rate == 125
        assert config.preprocess.notch_freq == 60.0
        assert config.pn.lambda_a == 0.1

    def test_board_channel_names(self):
        config = BoardConfig()
        assert len(config.channel_names) == 16
        assert config.channel_names[0] == "Fp1"


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        config = PipelineConfig()
        config.board.serial_port = "/dev/ttyUSB0"
        config.pn.lambda_a = 0.2

        path = tmp_path / "test_config.yaml"
        save_config(config, path)
        loaded = load_config(path)

        assert loaded.board.serial_port == "/dev/ttyUSB0"
        assert loaded.pn.lambda_a == 0.2
        assert loaded.preprocess.notch_freq == 60.0

    def test_missing_file_returns_defaults(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.board.board_id == 2

    def test_partial_override(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text("board:\n  serial_port: COM5\n")
        config = load_config(path)
        assert config.board.serial_port == "COM5"
        assert config.board.sample_rate == 125  # Default preserved
