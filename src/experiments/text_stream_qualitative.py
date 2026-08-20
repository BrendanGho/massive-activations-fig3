"""Qualitative look at the TEXT stream: high-norm text tokens + massive channels.

The text-stream analog of ``highnorm_qualitative`` (which is the image stream). Same channel
lens (``src/common/highnorm.py``): rank channels by mean|abs| over text tokens, per-text-token
L2 norm, and the post-hoc "norm minus the massive channels" — but on the TEXT tokens, which are
a 1-D sequence (positions), not a 2-D grid, so the outputs are per-position plots, not heatmaps.

Two text sources, because the architectures differ (see the plan / model_utils):
* **FLUX (MMDiT):** text is a live per-DiT-layer residual stream. Each block returns a
  ``(text[512], image[4096])`` tuple; we capture ``out[0]`` via ``capture_text=True``.
* **PixArt (cross-attn DiT):** the DiT has NO text stream (text is a frozen T5 encoding used via
  cross-attention). The real text stream is inside the **T5 encoder**, captured per T5 layer via
  ``register_text_encoder_hooks``. (The reference repo's PixArt "text" hook is buggy — it slices
  the image-only DiT output and grabs the bottom-right image corner, not text.)

Motivation: in LLMs the massive activations sit on text "sink" tokens (first token / EOS /
padding). We ask whether the same holds here, and whether the massive channels are shared with
the image stream.

    python -m src.experiments.text_stream_qualitative --config configs/highnorm_tokens.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from src.common import highnorm
from src.experiments.highnorm_tokens import load_highnorm_config

# --- pure helpers (no torch / matplotlib) -------------------------------------


def classify_token_positions(
    input_ids: list[int],
    n_text: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> list[str]:
    """Label each of ``n_text`` positions 'prompt' / 'eos' / 'pad'.

    ``input_ids`` are the real (unpadded) prompt token ids in order (T5 appends EOS). Positions
    at/after ``len(input_ids)`` are padding; within it, a pad-id or eos-id is labelled as such,
    everything else 'prompt'. Pure so it unit-tests without a tokenizer.
    """
    ids = list(input_ids)
    out: list[str] = []
    for i in range(int(n_text)):
        if i >= len(ids) or (pad_token_id is not None and ids[i] == pad_token_id):
            out.append("pad")
        elif eos_token_id is not None and ids[i] == eos_token_id:
            out.append("eos")
        else:
            out.append("prompt")
    return out


def analyze_text_layer(
    text_stream: np.ndarray,
    base_k: int,
    outlier_frac: float,
    image_channels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Channel-lens analysis of one ``(n_text, D)`` text stream. Pure numpy => CPU-testable.

    Reuses the same numeric core as the image tool: massive channels (top-k by mean|abs|),
    per-token L2 norm, the deconfounded norm (minus those channels), the high-norm token
    positions, and — if ``image_channels`` is given — the channel overlap (Jaccard) with the
    image stream (are the massive channels shared across streams?).
    """
    x = np.asarray(text_stream)
    chans = highnorm.top_channels(x, base_k)
    norms = highnorm.token_norms(x)
    n_ex = highnorm.token_norms(x, exclude=chans)
    massive = highnorm.massive_score(x, chans)
    high_norm_pos = highnorm.top_fraction_indices(norms, outlier_frac)

    overlap = None
    if image_channels is not None:
        overlap = float(
            highnorm.overlap_stats(chans, np.asarray(image_channels), x.shape[1])["iou"]
        )
    return {
        "n_text": int(x.shape[0]),
        "massive_channels": chans.tolist(),
        "norms": norms,
        "n_ex": n_ex,
        "massive": massive,
        "high_norm_positions": high_norm_pos.tolist(),
        "channel_overlap_with_image": overlap,
    }


# --- figure (matplotlib lazy) -------------------------------------------------


def _save_text_figure(
    path: str,
    rows: list[dict[str, Any]],
    layer: int,
    base_k: int,
) -> None:
    """One row per prompt: per-token-position L2 norm (full vs minus massive channels),
    high-norm positions marked, prompt/padding boundary drawn, top tokens annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    for r, row in enumerate(rows):
        ax = axes[r][0]
        a = row["analysis"]
        norms, n_ex = a["norms"], a["n_ex"]
        pos = np.arange(norms.size)
        ax.plot(pos, norms, color="#c0392b", lw=1.0, label="full norm")
        ax.plot(
            pos, n_ex, color="#2c3e50", lw=1.0, alpha=0.8, label=f"minus top-{base_k} massive ch"
        )
        hi = a["high_norm_positions"]
        ax.scatter(hi, norms[hi], s=18, color="#e67e22", zorder=3, label="high-norm token")
        # prompt / padding boundary
        n_real = row.get("n_real_tokens")
        if n_real:
            ax.axvline(n_real - 0.5, ls=":", c="gray", lw=1)
        # annotate the top few high-norm positions with their decoded token
        labels = row.get("token_labels") or []
        order = sorted(hi, key=lambda p: -norms[p])[:5]
        for p in order:
            tok = labels[p][1] if p < len(labels) else ""
            ax.annotate(
                f"{p}:{tok}",
                (p, norms[p]),
                fontsize=7,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        ov = a["channel_overlap_with_image"]
        ovs = f", ch-overlap w/ image={ov:.2f}" if ov is not None else ""
        ax.set_ylabel(row["prompt"][:24], fontsize=8)
        ax.set_title(f"massive ch {a['massive_channels']}{ovs}", fontsize=8)
        if r == 0:
            ax.legend(fontsize=7, loc="upper right")
        if r == n - 1:
            ax.set_xlabel("text token position (prompt … EOS … padding)")
    fig.suptitle(f"Text stream — high-norm tokens & massive channels — layer {layer}", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def output_name(layer: int, base_k: int, source: str) -> str:
    """Relative path ``text_<source>/text_L<layer>_ch<base_k>.png`` (foldered per source)."""
    return os.path.join(f"text_{source}", f"text_L{int(layer)}_ch{int(base_k)}.png")


# --- runner (torch imported lazily inside) ------------------------------------


def _text_source(cfg, override: str | None) -> str:
    if override and override != "auto":
        return override
    return "t5" if "pixart" in str(cfg.model_ckpt).lower() else "dit"


def _decode_labels(pipe, prompt: str, n_text: int, source: str):
    """(token_labels, n_real_tokens): per-position (kind, token_str) + real (unpadded) length."""
    tok = getattr(pipe, "tokenizer_2", None) if source == "dit" else None
    tok = tok or getattr(pipe, "tokenizer", None)
    if tok is None:
        return None, None
    ids = tok(prompt, return_tensors="pt", truncation=True).input_ids[0].tolist()
    kinds = classify_token_positions(
        ids, n_text, getattr(tok, "eos_token_id", None), getattr(tok, "pad_token_id", None)
    )
    strs = tok.convert_ids_to_tokens(ids)
    labels = [(kinds[i], strs[i] if i < len(strs) else "<pad>") for i in range(n_text)]
    return labels, len(ids)


def run(cfg, limit: int | None, layers_spec: str | None, text_source: str | None) -> list[str]:
    from src.common import model_utils

    prompts = cfg.prompts if limit is None else cfg.prompts[:limit]
    source = _text_source(cfg, text_source)

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    all_blocks = model_utils.discover_blocks(pipe.transformer)
    all_ids = [b.layer_id for b in all_blocks]

    state = model_utils.CaptureState()
    if source == "dit":
        # FLUX: text lives in the DiT blocks (out[0]); capture image too (for channel overlap).
        want = _resolve_layers(layers_spec, all_ids, cfg.target_layer)
        blocks = model_utils.select_layers(all_blocks, want)
        handles = model_utils.register_capture_hooks(
            pipe.transformer, blocks, state, capture_text=True
        )
        text_key = "DiT block"
    else:
        # PixArt: text lives in the T5 encoder; the DiT is image-only. Hook both (T5 for text,
        # DiT for the image channels used in the overlap).
        handles = model_utils.register_text_encoder_hooks(pipe, state)
        handles += model_utils.register_capture_hooks(pipe.transformer, all_blocks, state)
        want = None  # resolved after the first forward (T5 layer count is only known then)
        text_key = "T5 layer"

    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    try:
        for pid, prompt in enumerate(prompts):
            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            if not state.text_streams:
                raise RuntimeError(f"no text stream captured (source={source}) for prompt {pid}")
            layers = want if want is not None else sorted(state.text_streams)
            # one representative image channel-set (target layer) for the overlap metric
            img = state.image_streams.get(cfg.target_layer)
            img_ch = highnorm.top_channels(img, cfg.base_k) if img is not None else None
            labels, n_real = _decode_labels(pipe, prompt, info["n_text"], source)
            for ly in layers:
                if ly not in state.text_streams:
                    continue
                a = analyze_text_layer(state.text_streams[ly], cfg.base_k, cfg.outlier_frac, img_ch)
                rows_by_layer.setdefault(ly, []).append(
                    {
                        "prompt": prompt,
                        "analysis": a,
                        "token_labels": labels,
                        "n_real_tokens": n_real,
                    }
                )
                top = ", ".join(str(p) for p in a["high_norm_positions"][:8])
                print(
                    f"[text] prompt {pid + 1} {text_key} {ly}: massive ch {a['massive_channels']}"
                    f" | high-norm text positions [{top}]"
                    + (
                        f" | ch-overlap w/image {a['channel_overlap_with_image']:.2f}"
                        if a["channel_overlap_with_image"] is not None
                        else ""
                    )
                )
    finally:
        for h in handles:
            h.remove()

    out_paths: list[str] = []
    for ly, rows in rows_by_layer.items():
        out_path = os.path.join(cfg.output_dir, output_name(ly, cfg.base_k, source))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _save_text_figure(out_path, rows, ly, cfg.base_k)
        print(f"[text] wrote {out_path}")
        out_paths.append(out_path)
    return out_paths


def _resolve_layers(spec: str | None, all_ids: list[int], target_layer: int) -> list[int]:
    from src.experiments.highnorm_qualitative import resolve_layers

    return resolve_layers(spec, all_ids, target_layer)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Qualitative TEXT-stream high-norm / massive analysis.")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", default="", help="'all', '0,5,10', or empty for target_layer.")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument(
        "--text-source",
        choices=["auto", "dit", "t5"],
        default="auto",
        help="dit = FLUX per-block text (out[0]); t5 = T5 encoder layers (PixArt); auto by model.",
    )
    args = p.parse_args(argv)
    cfg = load_highnorm_config(args.config)
    run(cfg, limit=args.limit, layers_spec=args.layers, text_source=args.text_source)


if __name__ == "__main__":
    main()
