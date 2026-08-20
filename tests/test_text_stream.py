"""Tests for the text-stream analysis (SPEC: plan splendid-conjuring-goblet).

Pure-numpy parts run on CPU; the torch-gated ``_extract_text_stream`` test skips without
torch (runs on Colab), mirroring the existing capture tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments import text_stream_qualitative as ts


# --- classify_token_positions (pure) ------------------------------------------


def test_classify_token_positions_prompt_eos_pad():
    # ids: two prompt tokens then EOS(=1); n_text=6 -> rest is padding
    kinds = ts.classify_token_positions([10, 20, 1], n_text=6, eos_token_id=1, pad_token_id=0)
    assert kinds == ["prompt", "prompt", "eos", "pad", "pad", "pad"]


def test_classify_token_positions_inline_pad_id():
    kinds = ts.classify_token_positions([10, 0, 1], n_text=3, eos_token_id=1, pad_token_id=0)
    assert kinds == ["prompt", "pad", "eos"]


def test_classify_token_positions_no_special_ids():
    kinds = ts.classify_token_positions([5, 6], n_text=4)
    assert kinds == ["prompt", "prompt", "pad", "pad"]


# --- analyze_text_layer (pure numpy) ------------------------------------------


def _planted_text(n_text=20, d=32, sink_pos=3, sink_ch=0, seed=0):
    """Text stream whose high norm at one position comes from one massive channel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_text, d))
    x[sink_pos, sink_ch] = 400.0
    return x


def test_analyze_text_layer_finds_sink_and_channel():
    x = _planted_text(sink_pos=3, sink_ch=0)
    a = ts.analyze_text_layer(x, base_k=1, outlier_frac=0.1)
    assert a["massive_channels"] == [0]
    assert 3 in a["high_norm_positions"], "the sink position must be high-norm"
    assert a["n_text"] == 20


def test_analyze_text_layer_deconfounding_kills_the_sink_norm():
    x = _planted_text(sink_pos=3, sink_ch=0)
    a = ts.analyze_text_layer(x, base_k=1, outlier_frac=0.1)
    # full norm at the sink is huge; minus the massive channel it drops to background
    assert a["norms"][3] > 100
    assert a["n_ex"][3] < 0.2 * a["norms"][3]


def test_analyze_text_layer_channel_overlap_with_image():
    x = _planted_text(sink_ch=0)
    same = ts.analyze_text_layer(x, 1, 0.1, image_channels=np.array([0]))
    diff = ts.analyze_text_layer(x, 1, 0.1, image_channels=np.array([7]))
    assert same["channel_overlap_with_image"] == pytest.approx(1.0)
    assert diff["channel_overlap_with_image"] == pytest.approx(0.0)
    assert ts.analyze_text_layer(x, 1, 0.1)["channel_overlap_with_image"] is None


# --- output_name (pure) -------------------------------------------------------


def test_output_name_folders_by_source():
    import os

    assert ts.output_name(18, 1, "dit") == os.path.join("text_dit", "text_L18_ch1.png")
    assert ts.output_name(5, 2, "t5") == os.path.join("text_t5", "text_L5_ch2.png")
    # dit vs t5 at the same layer must not collide
    assert ts.output_name(5, 1, "dit") != ts.output_name(5, 1, "t5")


# --- _extract_text_stream (needs torch; skips on CPU) -------------------------


def test_extract_text_stream_tuple_and_concat():
    torch = pytest.importorskip("torch")
    from src.common import model_utils

    n_text, n_image, d = 5, 16, 3
    text = torch.arange(n_text * d, dtype=torch.float32).reshape(1, n_text, d)
    image = torch.zeros(1, n_image, d)

    # FLUX-style tuple (text, image): pick the seq==n_text tensor
    out = model_utils._extract_text_stream((text, image), n_text, n_image)
    assert out.shape == (n_text, d)
    np.testing.assert_allclose(out, text[0].numpy())

    # concatenated [text, image]: text is the FIRST n_text tokens
    concat = torch.cat([text, image], dim=1)  # (1, n_text+n_image, d)
    out2 = model_utils._extract_text_stream(concat, n_text, n_image)
    assert out2.shape == (n_text, d)
    np.testing.assert_allclose(out2, text[0].numpy())

    # image-only block (PixArt DiT): no text slice -> None
    assert model_utils._extract_text_stream(image, n_text, n_image) is None


# --- lazy imports -------------------------------------------------------------


def test_modules_import_without_heavy_deps(monkeypatch):
    import builtins
    import importlib

    blocked = {"torch", "diffusers", "transformers", "matplotlib"}
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} must not be imported at module scope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    for mod in ("src.common.model_utils", "src.experiments.text_stream_qualitative"):
        importlib.reload(importlib.import_module(mod))
