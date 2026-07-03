"""Stage 1 — generate with FLUX.2-klein, capture activations, cache reduced artifacts.

Default (and recommended) mode is ``--fused``: for each prompt we run stages 2+3
in-process right after capture and persist only the small reduced artifacts
(scores, channel indices, binary masks, decoded RGB, qualitative PNGs). The full
``[N_I, D]`` per-layer tensor is discarded — at FLUX hidden width across ~all layers
and 1,600 prompts it would be hundreds of GB.

``--no-fused`` instead persists the full per-layer activations to
``<cache>/full/prompt_{id}.npz`` so stages 2/3 can be debugged standalone.

Resumable: prompts already present in the cache are skipped.

    python -m src.stage1_generate_and_cache --config configs/default.yaml --fused
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.common import io
from src.common.config import Config, config_json, load_config, parse_set_overrides
from src.stage2_channel_ranking import rank_channels
from src.stage3_mask_construction import build_prompt_record, save_qualitative_dump


def write_run_metadata(cfg: Config, prompts: list[str], model_info: dict | None) -> str:
    """Log resolved config + ambiguity choices + provenance to run_metadata.json."""
    versions = io.package_versions()
    meta = {
        "resolved_config": {k: v for k, v in cfg.to_dict().items()},
        "versions": versions,
        "prompt_source": {
            "path": cfg.prompt_source,
            "n_prompts": len(prompts),
            "content_sha256": io.prompts_content_hash(prompts),
        },
        # Open ambiguities the paper does not specify — logged, not silently picked.
        "assumptions": {
            "random_k_trials": cfg.random_k_trials,
            "random_k_note": "not paper-specified; our default",
            "kmeans": {
                "library": "scikit-learn",
                "version": versions.get("sklearn"),
                "init": "library default (k-means++)",
                "n_init": "library default",
                "random_state": "derived per (seed, prompt_id, layer, strategy)",
            },
            "capture": {
                "timestep": "last denoising step only (hook overwrite -> last forward wins)",
                "stream": "image-stream tokens only; text/image split derived at runtime from N_I",
                "score": "abs(mean_over_tokens) — mean then abs",
            },
        },
        "model_info": model_info or {},
    }
    path = os.path.join(cfg.output_dir, "run_metadata.json")
    io.save_json(path, meta)
    return path


def _save_full_activations(cfg: Config, pid: int, streams: dict, rgb, prompt, info) -> None:
    full_dir = os.path.join(cfg.activation_cache_dir, "full")
    os.makedirs(full_dir, exist_ok=True)
    payload = {f"acts_{layer}": s.astype(np.float16) for layer, s in streams.items()}
    payload["rgb"] = rgb
    payload["prompt"] = np.array(prompt)
    payload["h_lat"] = np.array(info["h_lat"])
    payload["w_lat"] = np.array(info["w_lat"])
    np.savez_compressed(os.path.join(full_dir, f"prompt_{pid}.npz"), **payload)


def _progress(iterable, total, enabled=True):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total)
    except Exception:
        return iterable


def run(cfg: Config, fused: bool, limit: int | None, skip_cached: bool) -> None:
    prompts = io.load_prompts(cfg.prompt_source)
    if not prompts:
        raise ValueError(f"No prompts loaded from {cfg.prompt_source}")
    if limit is not None:
        prompts = prompts[:limit]

    # Write metadata up front (geometry filled in after the first generation).
    write_run_metadata(cfg, prompts, model_info=None)

    from src.common import model_utils

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    transformer = pipe.transformer
    blocks = model_utils.select_layers(model_utils.discover_blocks(transformer), cfg.layers)
    state = model_utils.CaptureState()
    handles = model_utils.register_capture_hooks(transformer, blocks, state)

    completed = io.completed_prompt_ids(cfg.activation_cache_dir) if skip_cached else set()
    writer = io.ShardWriter(cfg.activation_cache_dir, cfg.cache_batch_size) if fused else None

    model_info_written = False
    try:
        n_done = 0
        for pid, prompt in _progress(list(enumerate(prompts)), total=len(prompts)):
            if skip_cached and _already_cached(cfg, pid, fused, completed):
                continue

            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            streams = dict(state.image_streams)  # copy last-step image streams
            if not streams:
                raise RuntimeError(f"No image streams captured for prompt {pid}; check hooks.")

            if not model_info_written:
                info2 = dict(info)
                info2["n_layers"] = len(blocks)
                info2["layer_ids"] = [b.layer_id for b in blocks]
                write_run_metadata(cfg, prompts, model_info=info2)
                model_info_written = True

            if fused:
                _process_fused(cfg, pid, prompt, streams, rgb, info, writer)
            else:
                _save_full_activations(cfg, pid, streams, rgb, prompt, info)
            n_done += 1
    finally:
        for h in handles:
            h.remove()
        if writer is not None:
            writer.close()
    print(f"[stage1] processed {n_done} prompt(s); cache at {cfg.activation_cache_dir}")


def _already_cached(cfg: Config, pid: int, fused: bool, completed: set) -> bool:
    if fused:
        return pid in completed
    return os.path.isfile(os.path.join(cfg.activation_cache_dir, "full", f"prompt_{pid}.npz"))


def _process_fused(cfg, pid, prompt, streams, rgb, info, writer) -> None:
    rankings = {
        layer: rank_channels(stream, cfg.top_k, cfg.random_k_trials, cfg.seed, pid, layer)
        for layer, stream in streams.items()
    }
    want_qual = pid < cfg.num_example_prompts
    record, qualitative = build_prompt_record(
        pid,
        prompt,
        streams,
        rgb,
        rankings,
        cfg,
        info["h_lat"],
        info["w_lat"],
        collect_qualitative=want_qual,
    )
    writer.add(record)
    if want_qual:
        for layer, strat_map in qualitative.items():
            for strategy, (heatmap, mask) in strat_map.items():
                save_qualitative_dump(cfg.output_dir, pid, layer, strategy, heatmap, mask, rgb)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1: generate + cache (fused by default).")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config key: --set key=value (repeatable).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--fused",
        dest="fused",
        action="store_true",
        default=True,
        help="Run stages 2+3 in-process; persist only reduced artifacts (default).",
    )
    mode.add_argument(
        "--no-fused",
        dest="fused",
        action="store_false",
        help="Persist full per-layer activations for standalone debugging.",
    )
    p.add_argument(
        "--skip-if-cached",
        dest="skip_cached",
        action="store_true",
        default=True,
        help="Skip prompts already cached (default; enables resume).",
    )
    p.add_argument(
        "--no-skip", dest="skip_cached", action="store_false", help="Recompute even if cached."
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N prompts.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, parse_set_overrides(args.overrides))
    print("[stage1] resolved config:\n" + config_json(cfg))
    run(cfg, fused=args.fused, limit=args.limit, skip_cached=args.skip_cached)


if __name__ == "__main__":
    main()
