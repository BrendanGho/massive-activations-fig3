"""IO: IoU, nearest upsample, resumable shard round-trip, prompt loading, hashing."""

from __future__ import annotations

import importlib
import json

import numpy as np

from src.common import io


def test_iou_basic_and_empty():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [0, 0]], dtype=bool)
    assert io.iou(a, b) == 0.5  # inter 1, union 2
    empty = np.zeros((2, 2), dtype=bool)
    assert io.iou(empty, empty) == 1.0  # both empty -> perfect
    assert io.iou(a, empty) == 0.0


def test_upsample_nearest_shape_and_values():
    arr = np.array([[0, 1], [2, 3]])
    up = io.upsample_nearest_2d(arr, 4, 4)
    assert up.shape == (4, 4)
    assert up[0, 0] == 0 and up[0, 3] == 1
    assert up[3, 0] == 2 and up[3, 3] == 3


def _make_record(pid: int) -> io.PromptRecord:
    rng = np.random.default_rng(pid)
    return io.PromptRecord(
        prompt_id=pid,
        prompt=f"a photo of subject {pid}",
        h_lat=2,
        w_lat=2,
        d=5,
        img_h=8,
        img_w=8,
        layers=[0, 1],
        n_random_trials=2,
        arrays={
            "rgb": (rng.random((8, 8, 3)) * 255).astype(np.uint8),
            "scores": rng.random((2, 5)).astype(np.float16),
            "top_idx": np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int32),
            "bottom_idx": np.array([[4, 3, 2], [1, 0, 4]], dtype=np.int32),
            "random_idx": rng.integers(0, 5, size=(2, 2, 3)).astype(np.int32),
            "mask_top": rng.integers(0, 2, size=(2, 2, 2)).astype(np.uint8),
            "mask_bottom": rng.integers(0, 2, size=(2, 2, 2)).astype(np.uint8),
            "mask_random": rng.integers(0, 2, size=(2, 2, 2, 2)).astype(np.uint8),
        },
    )


def test_shard_roundtrip_and_resume(tmp_path):
    cache = str(tmp_path / "cache")
    rec = _make_record(0)
    with io.ShardWriter(cache, batch_size=8) as w:
        w.add(rec)

    assert io.completed_prompt_ids(cache) == {0}
    loaded = list(io.iter_cache(cache))
    assert len(loaded) == 1
    got = loaded[0]
    assert got.prompt == rec.prompt
    assert got.layers == [0, 1]
    assert got.n_random_trials == 2
    for key, arr in rec.arrays.items():
        assert np.array_equal(got.arrays[key], arr), key

    # Resume: a fresh writer must skip the already-completed prompt (no new shard).
    import glob
    import os

    n_shards_before = len(glob.glob(os.path.join(cache, "shard_*.npz")))
    with io.ShardWriter(cache, batch_size=8) as w2:
        w2.add(_make_record(0))
    n_shards_after = len(glob.glob(os.path.join(cache, "shard_*.npz")))
    assert n_shards_after == n_shards_before


def test_batch_flush_creates_multiple_shards(tmp_path):
    cache = str(tmp_path / "cache")
    with io.ShardWriter(cache, batch_size=2) as w:
        for pid in range(5):
            w.add(_make_record(pid))
    assert io.completed_prompt_ids(cache) == {0, 1, 2, 3, 4}
    assert len(list(io.iter_cache(cache))) == 5


def test_load_prompts_txt_json_jsonl(tmp_path):
    txt = tmp_path / "p.txt"
    txt.write_text("a cat\n\nb dog\n")
    assert io.load_prompts(str(txt)) == ["a cat", "b dog"]

    js = tmp_path / "p.json"
    js.write_text(json.dumps(["x", "y"]))
    assert io.load_prompts(str(js)) == ["x", "y"]

    jsl = tmp_path / "p.jsonl"
    jsl.write_text('{"prompt": "u"}\n{"text": "v"}\n')
    assert io.load_prompts(str(jsl)) == ["u", "v"]


def test_content_hash_order_sensitive():
    h1 = io.prompts_content_hash(["a", "b"])
    h2 = io.prompts_content_hash(["a", "b"])
    h3 = io.prompts_content_hash(["b", "a"])
    assert h1 == h2
    assert h1 != h3


def test_stage_modules_import_without_torch():
    # The model stack is not installed in this env; importing the stage modules
    # must still succeed (torch/diffusers/matplotlib are lazily imported).
    for name in (
        "src.stage1_generate_and_cache",
        "src.stage2_channel_ranking",
        "src.stage3_mask_construction",
        "src.stage4_evaluate_figure3d",
    ):
        mod = importlib.import_module(name)
        assert hasattr(mod, "main")
