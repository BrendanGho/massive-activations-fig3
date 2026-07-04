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
    forward_count: int = 0
    image_streams: dict[int, np.ndarray] = field(default_factory=dict)
    # Optional multi-timestep capture: denoising-step indices to snapshot. The last-step
    # `image_streams` buffer is always kept regardless; `step_streams` is keyed by
    # (step, layer_id). One transformer forward == one denoising step for FLUX
    # (no CFG batch duplication), so the step index is `forward_count - 1`.
    capture_steps: set[int] | None = None
    step_streams: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.n_image = None
        self.n_text = None
        self.forward_count = 0
        self.image_streams = {}
        self.step_streams = {}  # capture_steps (the request) survives reset


def _extract_image_stream(output: Any, n_image: int) -> np.ndarray | None:
    """Pull the [N_I, D] image slice from a block's output, batch index 0.

    Chooses the tensor whose seq-len == N_I (image-only block); else the last
    N_I tokens of a longer [text, image] sequence (single-stream block).
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
    return chosen[0].detach().float().cpu().numpy()


def register_capture_hooks(transformer: Any, blocks: list[BlockRef], state: CaptureState):
    """Register the pre-hook (derives N_I) + per-block hooks (capture image stream).

    Returns a list of hook handles; call ``.remove()`` on each when done.
    """
    handles = []

    def pre_hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None and len(args) > 0:
            hidden = args[0]
        enc = kwargs.get("encoder_hidden_states")
        if enc is None and len(args) > 1:
            enc = args[1]
        if hidden is not None and hasattr(hidden, "shape"):
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

        return hook

    for b in blocks:
        handles.append(b.module.register_forward_hook(make_hook(b.layer_id)))
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
