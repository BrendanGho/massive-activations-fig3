"""Model-touching helpers for Stage 1 (FLUX.2-klein) and Stage 4 (BiRefNet).

Everything heavy (torch / diffusers / transformers) is imported **lazily inside
functions**, so importing this module is cheap and the numeric core + tests never
pull in a GPU stack.

Design notes (see spec invariants):
* The text/image split is DERIVED at runtime, not hard-coded: a forward pre-hook
  on the transformer reads ``hidden_states.shape[1]`` (the image-latent token
  count ``N_I``) from the packed sequence. Block hooks then take the output tensor
  whose seq-len == ``N_I`` (image-only / MMDiT double-stream block) or the last
  ``N_I`` tokens of a longer [text, image] sequence (single-stream block).
* "Only the last denoising timestep" is achieved by hooks overwriting a per-layer
  buffer on every forward; the value retained after generation is the last step's
  (robust to CFG / step count).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def torch_dtype(dtype_str: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]


# --- pipeline / model loading -------------------------------------------------


def load_pipeline(cfg, offload: bool = False) -> Any:
    """Load the FLUX diffusion pipeline onto the configured device/dtype.

    ``offload=True`` uses ``enable_model_cpu_offload`` instead of moving the whole
    pipeline onto ``cfg.device`` — needed to fit large checkpoints (e.g. the 12B
    FLUX.1-dev) on smaller GPUs, at the cost of speed.
    """
    from diffusers import DiffusionPipeline

    dtype = torch_dtype(cfg.dtype)
    try:
        pipe = DiffusionPipeline.from_pretrained(cfg.model_ckpt, torch_dtype=dtype)
    except TypeError:
        # Some custom pipelines need trust_remote_code.
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_ckpt, torch_dtype=dtype, trust_remote_code=True
        )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(cfg.device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def load_birefnet(cfg) -> Any:
    """Load BiRefNet for pseudo-GT foreground masks (Stage 4)."""
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        cfg.birefnet_weights, trust_remote_code=True
    )
    model.to(cfg.device)
    model.eval()
    return model


# --- transformer block discovery ---------------------------------------------


@dataclass
class BlockRef:
    layer_id: int
    module: Any
    kind: str  # "double" | "single" | "block"


def discover_blocks(transformer: Any) -> list[BlockRef]:
    """Auto-detect every transformer block, numbered sequentially.

    Prefers the FLUX layout (``transformer_blocks`` then ``single_transformer_blocks``);
    falls back to scanning for any ModuleList of *Block modules.
    """
    import torch.nn as nn

    refs: list[BlockRef] = []
    idx = 0
    named = [
        ("transformer_blocks", "double"),
        ("single_transformer_blocks", "single"),
    ]
    found_named = False
    for attr, kind in named:
        blocks = getattr(transformer, attr, None)
        if blocks is not None and len(blocks) > 0:
            found_named = True
            for m in blocks:
                refs.append(BlockRef(layer_id=idx, module=m, kind=kind))
                idx += 1

    if not found_named:
        # Fallback: first ModuleList whose children look like transformer blocks.
        for _name, mod in transformer.named_children():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                child = mod[0]
                if "block" in type(child).__name__.lower():
                    for m in mod:
                        refs.append(BlockRef(layer_id=idx, module=m, kind="block"))
                        idx += 1
    if not refs:
        raise RuntimeError(
            "Could not auto-detect transformer blocks; inspect the model and pass "
            "an explicit `layers` list once the block container is known."
        )
    return refs


def select_layers(blocks: list[BlockRef], layers_cfg: Any) -> list[BlockRef]:
    if layers_cfg == "all":
        return blocks
    wanted = set(int(x) for x in layers_cfg)
    selected = [b for b in blocks if b.layer_id in wanted]
    missing = wanted - {b.layer_id for b in selected}
    if missing:
        raise ValueError(f"Requested layers not found in model: {sorted(missing)}")
    return selected


# --- capture hooks ------------------------------------------------------------


@dataclass
class CaptureState:
    n_image: int | None = None
    n_text: int | None = None
    n_batch: int | None = None  # batch size of the transformer forward; >1 => CFG is active
    forward_count: int = 0
    image_streams: dict[int, np.ndarray] = field(default_factory=dict)
    # Optional text-stream capture (opt-in via register_capture_hooks(capture_text=True) for
    # FLUX's per-DiT-layer text tokens, or register_text_encoder_hooks for a T5 text encoder).
    # Keyed by layer id, [N_text, D], last forward wins — same convention as image_streams.
    text_streams: dict[int, np.ndarray] = field(default_factory=dict)
    # Optional multi-timestep capture: denoising-step indices to snapshot. The last-step
    # `image_streams` buffer is always kept regardless; `step_streams` is keyed by
    # (step, layer_id). One transformer forward == one denoising step for FLUX
    # (no CFG batch duplication), so the step index is `forward_count - 1`.
    capture_steps: set[int] | None = None
    step_streams: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.n_image = None
        self.n_text = None
        self.n_batch = None
        self.forward_count = 0
        self.image_streams = {}
        self.text_streams = {}
        self.step_streams = {}  # capture_steps (the request) survives reset


def _extract_image_stream(output: Any, n_image: int) -> np.ndarray | None:
    """Pull the [N_I, D] image slice from a block's output.

    Chooses the tensor whose seq-len == N_I (image-only block); else the last
    N_I tokens of a longer [text, image] sequence (single-stream block). Takes the
    LAST batch element: diffusers batches classifier-free guidance as [uncond, cond],
    so -1 is the conditional branch. FLUX runs batch-1 (guidance is an embedding, not a
    doubled batch), so -1 is identical to 0 there.

    Caveat for real-CFG models (PixArt): we capture the *conditional forward*, which is a
    property of the network on the text-conditioned input — the right object for studying
    massive activations. It is NOT the guidance-extrapolated latent (uncond + s*(cond-uncond))
    that actually produced the rendered image. Verify CFG is active via ``CaptureState.n_batch``
    (== 2). This assumes the diffusers [uncond, cond] ordering; a pipeline that batches CFG
    differently would need this revisited.
    """
    import torch

    if isinstance(output, torch.Tensor):
        candidates = [output]
    elif isinstance(output, (tuple, list)):
        candidates = [t for t in output if isinstance(t, torch.Tensor)]
    else:
        return None

    exact = None
    longer = None
    for t in candidates:
        if t.dim() != 3:
            continue
        seq = t.shape[1]
        if seq == n_image:
            exact = t
            break
        if seq > n_image and longer is None:
            longer = t
    chosen = exact if exact is not None else longer
    if chosen is None:
        return None
    if chosen.shape[1] > n_image:
        chosen = chosen[:, -n_image:, :]
    return chosen[-1].detach().float().cpu().numpy()


def _extract_text_stream(output: Any, n_text: int, n_image: int) -> np.ndarray | None:
    """Pull the [N_text, D] text slice from a FLUX block's output.

    Mirror of ``_extract_image_stream`` for the complementary (text) tokens. FLUX blocks
    return a ``(text[N_text], image[N_image])`` tuple, so we pick the tensor whose seq-len
    == ``n_text`` (== ``out[0]`` in the reference). If instead a single concatenated
    ``[text, image]`` sequence is returned, text is the FIRST ``n_text`` tokens (image is
    the last ``n_image``, which ``_extract_image_stream`` takes). Returns None when there is
    no text slice (e.g. an image-only PixArt DiT block). Last batch element (CFG conditional).
    """
    import torch

    if n_text is None or n_text <= 0:
        return None
    if isinstance(output, torch.Tensor):
        candidates = [output]
    elif isinstance(output, (tuple, list)):
        candidates = [t for t in output if isinstance(t, torch.Tensor)]
    else:
        return None

    exact = None
    concat = None
    for t in candidates:
        if t.dim() != 3:
            continue
        seq = t.shape[1]
        if seq == n_text:
            exact = t
            break
        if seq == n_text + n_image and concat is None:
            concat = t
    if exact is not None:
        return exact[-1].detach().float().cpu().numpy()
    if concat is not None:
        return concat[:, :n_text, :][-1].detach().float().cpu().numpy()
    return None


def register_capture_hooks(
    transformer: Any,
    blocks: list[BlockRef],
    state: CaptureState,
    capture_text: bool = False,
):
    """Register the pre-hook (derives N_I / N_text) + per-block hooks (capture image stream).

    ``capture_text=True`` additionally stores the per-block text slice in
    ``state.text_streams`` (FLUX: the ``out[0]`` text tensor of each block). No-op for
    image-only blocks (e.g. PixArt DiT), whose text stream lives in the T5 encoder instead
    (see ``register_text_encoder_hooks``). Returns a list of hook handles; ``.remove()`` each.
    """
    handles = []

    def pre_hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None and len(args) > 0:
            hidden = args[0]
        enc = kwargs.get("encoder_hidden_states")
        if enc is None and len(args) > 1:
            enc = args[1]
        if hidden is not None and hasattr(hidden, "shape"):
            state.n_batch = int(hidden.shape[0])
            if hidden.dim() == 4:
                # Conv-latent input (B, C, H, W), e.g. PixArt/DiT: the transformer
                # patchifies internally, so N_I = (H/patch)*(W/patch). FLUX instead
                # passes an already-packed (B, N_I, D) sequence, handled by the else.
                # Fail loud if patch_size is unavailable: guessing it silently yields a
                # wrong token count and a confusing "no capture" error downstream.
                ps = getattr(getattr(module, "config", None), "patch_size", None)
                if not ps:
                    raise RuntimeError(
                        "4D transformer input but transformer.config.patch_size is missing; "
                        "cannot infer the image-token count. Add patch_size handling for this "
                        "model before capturing."
                    )
                ps = int(ps)
                state.n_image = (hidden.shape[-2] // ps) * (hidden.shape[-1] // ps)
            else:
                state.n_image = int(hidden.shape[1])
        if enc is not None and hasattr(enc, "shape"):
            state.n_text = int(enc.shape[1])
        state.forward_count += 1

    handles.append(transformer.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def make_hook(layer_id: int):
        def hook(_module, _inp, output):
            if state.n_image is None:
                return
            stream = _extract_image_stream(output, state.n_image)
            if stream is not None:
                state.image_streams[layer_id] = stream  # overwrite -> last step wins
                step = state.forward_count - 1  # pre-hook already counted this forward
                if state.capture_steps is not None and step in state.capture_steps:
                    state.step_streams[(step, layer_id)] = stream
            if capture_text and state.n_text:
                tstream = _extract_text_stream(output, state.n_text, state.n_image)
                if tstream is not None:
                    state.text_streams[layer_id] = tstream  # overwrite -> last step wins

        return hook

    for b in blocks:
        handles.append(b.module.register_forward_hook(make_hook(b.layer_id)))
    return handles


def find_text_encoder_layers(pipe: Any) -> list[Any]:
    """Return the per-layer modules of the pipe's T5 text encoder, in order.

    PixArt's diffusion transformer has no text residual stream (text enters via
    cross-attention as a frozen T5 encoding), so the real per-layer text stream lives inside
    the T5 encoder. Prefers ``text_encoder`` (PixArt's T5); falls back to ``text_encoder_2``
    (FLUX's T5). Each returned module is a T5 encoder block whose output[0] is [B, N_text, D].
    """
    for attr in ("text_encoder", "text_encoder_2"):
        enc = getattr(pipe, attr, None)
        block = getattr(getattr(enc, "encoder", None), "block", None)
        if block is not None and len(block) > 0:
            return list(block)
    raise RuntimeError(
        "Could not locate a T5 encoder (pipe.text_encoder[.encoder.block]); "
        "cannot capture the text-encoder stream for this pipeline."
    )


def register_text_encoder_hooks(pipe: Any, state: CaptureState):
    """Hook each T5 text-encoder layer to capture the per-layer TEXT stream.

    This is the correct PixArt text stream (the DiT exposes none). The T5 encoder runs once
    per prompt encoding, so this is a per-encoder-layer snapshot (not per denoising step).
    Stores ``[N_text, D]`` in ``state.text_streams`` keyed by encoder-layer index, last
    forward wins (under CFG the encoder may run for the negative prompt too; the conditional
    call overwrites). Returns handles; ``.remove()`` each when done.
    """
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(t, "dim") and t.dim() == 3:
                state.text_streams[layer_id] = t[-1].detach().float().cpu().numpy()

        return hook

    for i, layer in enumerate(find_text_encoder_layers(pipe)):
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


def latent_grid(n_image: int) -> tuple[int, int]:
    """Infer (H_lat, W_lat) from the image token count for a square image."""
    root = int(round(math.sqrt(n_image)))
    if root * root != n_image:
        raise ValueError(
            f"N_I={n_image} is not a perfect square; cannot infer a square latent grid. "
            "Pass an explicit grid or use a square resolution."
        )
    return root, root


# --- generation ---------------------------------------------------------------


def generate_with_capture(pipe: Any, prompt: str, cfg, state: CaptureState):
    """Run one generation, capturing last-step image streams. Returns (rgb, info).

    rgb: uint8 (H, W, 3). info: dict with n_image, n_text, h_lat, w_lat, forward_count.
    """
    import torch

    state.reset()
    generator = torch.Generator(device=cfg.device).manual_seed(int(cfg.seed))
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(cfg.num_denoising_steps),
        "height": int(cfg.resolution),
        "width": int(cfg.resolution),
        "generator": generator,
        "output_type": "np",
    }
    if cfg.guidance_scale is not None:
        kwargs["guidance_scale"] = float(cfg.guidance_scale)

    with torch.no_grad():
        result = pipe(**kwargs)

    image = result.images[0]  # (H, W, 3) float in [0, 1]
    rgb = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if state.n_image is None:
        raise RuntimeError(
            "Capture pre-hook never fired; transformer forward signature unexpected."
        )
    h_lat, w_lat = latent_grid(state.n_image)
    info = {
        "n_image": state.n_image,
        "n_text": state.n_text,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "forward_count": state.forward_count,
    }
    return rgb, info


def birefnet_mask(
    model: Any, rgb: np.ndarray, cfg, out_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """Run BiRefNet on an RGB uint8 image -> binary foreground mask (bool).

    out_hw: optional (H, W) to resize the mask to; defaults to the input image size.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    h, w = rgb.shape[:2]
    out_h, out_w = out_hw if out_hw is not None else (h, w)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Match the input to BiRefNet's actual weight dtype (load_birefnet keeps it fp32).
    # Casting to cfg.dtype instead would feed fp16/bf16 activations into fp32 conv
    # weights -> "Input type and bias type should be the same" RuntimeError on GPU runs.
    model_dtype = next(model.parameters()).dtype
    x = tfm(rgb).unsqueeze(0).to(cfg.device, model_dtype)
    with torch.no_grad():
        preds = model(x)
        logits = preds[-1] if isinstance(preds, (list, tuple)) else preds
        prob = logits.sigmoid().float().cpu()
    prob = F.interpolate(prob, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return prob[0, 0].numpy() > 0.5
