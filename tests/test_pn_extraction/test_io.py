"""Tests for PN parameter serialization and QDNU data contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from openbci_eeg.pn_extraction.io import (
    save_pn_parameters,
    load_pn_parameters,
    pn_to_qdnu_format,
    pn_at_time,
)


class TestSaveLoadPN:
    def test_roundtrip(self, sample_pn_params, tmp_path):
        path = tmp_path / "test_pn.npz"
        save_pn_parameters(sample_pn_params, path, metadata={"test": True})
        loaded, meta = load_pn_parameters(path)

        assert set(loaded.keys()) == set(sample_pn_params.keys())
        for ch in loaded:
            for key in ("a", "b", "c"):
                np.testing.assert_array_almost_equal(
                    loaded[ch][key], sample_pn_params[ch][key]
                )
        assert meta["test"] is True

    def test_no_metadata(self, sample_pn_params, tmp_path):
        path = tmp_path / "no_meta.npz"
        save_pn_parameters(sample_pn_params, path)
        loaded, meta = load_pn_parameters(path)
        assert meta is None


class TestQDNUFormat:
    def test_structure(self, sample_pn_params, channel_names):
        result = pn_to_qdnu_format(
            sample_pn_params,
            subject_id="S001",
            session_id="test_session",
        )
        assert "metadata" in result
        assert "pn_parameters" in result
        assert result["metadata"]["subject_id"] == "S001"
        assert result["metadata"]["channels"] == channel_names

    def test_json_serializable(self, sample_pn_params):
        result = pn_to_qdnu_format(sample_pn_params)
        # Should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_values_are_lists(self, sample_pn_params):
        result = pn_to_qdnu_format(sample_pn_params)
        for ch in result["pn_parameters"].values():
            assert isinstance(ch["a"], list)


class TestPNAtTime:
    def test_all_channels(self, sample_pn_params, channel_names):
        result = pn_at_time(sample_pn_params, time_idx=0)
        assert set(result.keys()) == set(channel_names)

    def test_tuple_format(self, sample_pn_params):
        result = pn_at_time(sample_pn_params, time_idx=0)
        for ch, abc in result.items():
            assert len(abc) == 3
            assert all(isinstance(v, float) for v in abc)

    def test_channel_subset(self, sample_pn_params):
        subset = ["Fp1", "Fp2"]
        result = pn_at_time(sample_pn_params, time_idx=0, channels=subset)
        assert set(result.keys()) == set(subset)
