"""FLUX Q9 execution. Raw activations are transient CPU/GPU memory only."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import generation_function as q7
from .text_image_coupling import (
    EDGE_METHODS,
    PRESETS,
    REVISION,
    Q9Config,
    Reservoir,
    fingerprint,
    matched_positions,
    norm_candidates,
    select_text,
    state_metrics,
    token_classes,
)


def json_value(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path, value):
    """Atomic replace of an identity-specific compact artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=json_value, allow_nan=False, indent=2))
    temporary.replace(path)


def cpu_tree(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def device_tree(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(device_tree(v, device) for v in value)
    return value


def numpy(x):
    return x[0].detach().float().cpu().numpy()


def streams(output, nt):
    """Block outputs, not attention outputs: (text,image), or a concatenated tensor."""
    import torch

    if isinstance(output, (tuple, list)) and len(output) == 2:
        text, image = output
        if text.ndim == image.ndim == 3 and text.shape[1] == nt:
            return text, image
    if torch.is_tensor(output) and output.ndim == 3 and output.shape[1] > nt:
        return output[:, :nt], output[:, nt:]
    raise RuntimeError("Unsupported FLUX block output; cannot identify text/image streams")


def replace_streams(output, text, image):
    import torch

    if isinstance(output, (tuple, list)):
        return type(output)((text, image))
    return torch.cat((text, image), dim=1)


def edit_state(x, mask, method, vector=None, channel=None, donor=None, seed=0):
    """Return edited tensor and actual edit audit; never mutate cached conditioning."""
    import torch

    mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    y = x.clone()
    z = y[:, mask].float()
    if not mask.any():
        return y, {"tokens": 0, "delta_l2": 0.0, "status": "no_candidates"}
    if method in {"remove_direction", "norm_matched", "random_direction"}:
        if vector is None:
            return y, {"tokens": int(mask.sum()), "delta_l2": 0.0, "status": "no_direction"}
        v = torch.as_tensor(vector, device=x.device, dtype=torch.float32)
        removed = (z @ v)[..., None] * v
        perpendicular = z - removed
        if method == "remove_direction":
            new = perpendicular
        elif method == "norm_matched":
            new = z * (perpendicular.norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-20))[..., None]
        else:
            # Random orthogonal rotation: matches both final norm and edit L2 of projection.
            rng = torch.Generator(device=x.device).manual_seed(seed)
            unit = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-20)
            orth = torch.randn(z.shape, generator=rng, device=x.device)
            orth -= (orth * unit).sum(-1, keepdim=True) * unit
            orth /= orth.norm(dim=-1, keepdim=True).clamp_min(1e-20)
            ratio = (perpendicular.norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-20)).clamp(0, 1)
            new = perpendicular.norm(dim=-1, keepdim=True) * (
                ratio[..., None] * unit + (1 - ratio**2).sqrt()[..., None] * orth
            )
    elif method == "suppress_channel":
        if channel is None:
            return y, {"tokens": int(mask.sum()), "delta_l2": 0.0, "status": "no_direction"}
        new = z.clone()
        new[..., channel] = 0
    elif method in {"zero", "ordinary_zero"}:
        new = torch.zeros_like(z)
    elif method == "donor_swap":
        if donor is None:
            return y, {"tokens": int(mask.sum()), "delta_l2": 0.0, "status": "no_matched_donor"}
        new = torch.as_tensor(donor, device=x.device, dtype=torch.float32)[None, mask]
    else:
        raise ValueError(method)
    y[:, mask] = new.to(x.dtype)
    delta = (y[:, mask].float() - x[:, mask].float()).norm().item()
    return y, {
        "tokens": int(mask.sum()),
        "delta_l2": delta,
        "status": "edited" if delta > 0 else "zero_effect",
    }


def rescue_state(x, mask, clean_values, vector, kind):
    import torch

    y = x.clone()
    ix = torch.as_tensor(mask, device=x.device, dtype=torch.bool)
    clean = torch.as_tensor(clean_values, device=x.device, dtype=torch.float32)[None]
    if kind in {"projection", "ordinary_projection"}:
        v = torch.as_tensor(vector, device=x.device, dtype=torch.float32)
        z = y[:, ix].float()
        y[:, ix] = (z + ((clean - z) @ v)[..., None] * v).to(x.dtype)
    elif kind == "state":
        y[:, ix] = clean.to(x.dtype)
    elif kind == "sham":
        y[:, ix] = x[:, ix]
    return y


def attention_reductions(
    query, key, nt, image_mask, classes, chunk_size, fixed_image_mask=None, fixed_text_mask=None
):
    """Exact all-key softmax; reductions retain per-head incoming mass, never NxN traces."""
    import torch

    if query.shape[0] != 1:
        raise RuntimeError("Q9 currently requires one scenario per forward")
    n, heads = query.shape[1:3]
    key_float = key.float()
    incoming = torch.zeros((2, heads, n), device=query.device)
    entropy = torch.zeros((2, heads), device=query.device)
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        scores = torch.einsum("bqhd,bkhd->bhqk", query[:, lo:hi].float(), key_float)
        p = (scores * query.shape[-1] ** -0.5).softmax(-1)[0]
        for group, take in enumerate(
            (
                torch.arange(lo, hi, device=query.device) < nt,
                torch.arange(lo, hi, device=query.device) >= nt,
            )
        ):
            if take.any():
                part = p[:, take]
                incoming[group] += part.sum(1)
                entropy[group] += -(part * part.clamp_min(1e-30).log()).sum((1, 2))
    incoming[0] /= nt
    incoming[1] /= n - nt
    entropy[0] /= nt
    entropy[1] /= n - nt
    inc = incoming.cpu().numpy()
    ent = entropy.cpu().numpy()
    rows = []
    for g, name in enumerate(("text_query", "image_query")):
        for h in range(heads):
            row = {
                "population": f"{name}/head_{h}",
                "text_mass": float(inc[g, h, :nt].sum()),
                "_text_incoming": inc[g, h, :nt],
                "image_mass": float(inc[g, h, nt:].sum()),
                "entropy": float(ent[g, h]),
                "register_mass": float(inc[g, h, nt:][image_mask].sum()),
                "image_sink_position": int(np.argmax(inc[g, h, nt:])),
                "image_sink_mass": float(inc[g, h, nt:].max()),
                "text_sink_position": int(np.argmax(inc[g, h, :nt])),
            }
            for cls in np.unique(classes):
                take = classes == cls
                row[f"{cls}_mass"] = float(inc[g, h, :nt][take].sum())
                row[f"{cls}_per_token"] = float(inc[g, h, :nt][take].mean())
            if fixed_image_mask is not None:
                row["fixed_register_mass"] = float(inc[g, h, nt:][fixed_image_mask].sum())
            if fixed_text_mask is not None:
                row["fixed_text_candidate_mass"] = float(inc[g, h, :nt][fixed_text_mask].sum())
            rows.append(row)
    return rows, inc[1, :, :nt]


def sink_mask(incoming, cfg):
    total = incoming.sum(-1, keepdims=True)
    conditional = incoming / np.maximum(total, 1e-20)
    return np.any(
        (conditional > cfg.sink_enrichment / incoming.shape[-1]) & (incoming >= cfg.sink_min_mass),
        axis=0,
    )


def edge_indices(method, nt, ni, frame, classes):
    text = np.flatnonzero(frame["text_mask"])
    image = np.flatnonzero(frame["image_mask"]) + nt
    if method.startswith("image_reads_text"):
        return np.arange(nt, nt + ni), text
    if method.startswith("text_reads_register"):
        return text, image
    if method.startswith("text_reads_content"):
        return text, np.flatnonzero(classes == "content")
    raise ValueError(method)


def edge_attention(query, key, value, queries, keys, kind, original, chunk_size):
    """Edited raw attention or removed value contribution for selected query rows."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    qi = torch.as_tensor(queries, device=query.device, dtype=torch.long)
    ki = torch.as_tensor(keys, device=query.device, dtype=torch.long)
    parts = []
    for lo in range(0, len(qi), chunk_size):
        q = query[:, qi[lo : lo + chunk_size]]
        if kind == "score":
            mask = torch.zeros(
                (1, query.shape[2], q.shape[1], key.shape[1]), device=q.device, dtype=q.dtype
            )
            mask[..., ki] = -float("inf")
            out = dispatch_attention_fn(
                q,
                key,
                value,
                attn_mask=mask,
                backend=getattr(original, "_attention_backend", None),
                parallel_config=getattr(original, "_parallel_config", None),
            )
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), key.float()) * q.shape[-1] ** -0.5
            weights = scores.softmax(-1)[..., ki]
            out = torch.einsum("bhqk,bkhd->bqhd", weights, value[:, ki].float()).to(q.dtype)
        parts.append(out.flatten(2, 3))
    return torch.cat(parts, dim=1)


def patch_attention_output(attn, original_output, raw, queries, nt, dual, subtract=False):
    """Preserve untouched query rows exactly; value subtraction excludes projection bias."""
    import torch
    import torch.nn.functional as F

    if not dual:
        y = original_output.clone()
        idx = torch.as_tensor(queries, device=y.device)
        y[:, idx] = y[:, idx] - raw if subtract else raw
        return y
    image, text = original_output
    image, text = image.clone(), text.clone()
    for is_text, target in ((True, text), (False, image)):
        selected = queries < nt if is_text else queries >= nt
        if not selected.any():
            continue
        idx = torch.as_tensor(queries[selected] - (0 if is_text else nt), device=target.device)
        part = raw[:, torch.as_tensor(selected, device=raw.device)]
        projection = attn.to_add_out if is_text else attn.to_out[0]
        projected = F.linear(part, projection.weight, None) if subtract else projection(part)
        if not is_text:
            projected = attn.to_out[1](projected)
        target[:, idx] = target[:, idx] - projected if subtract else projected
    return image, text


class Trace:
    def __init__(self):
        self.rows, self.frames, self.inputs, self.snapshots, self.predictions = [], {}, {}, {}, {}
        self.audit = []
        self.noise_sha256 = None
        self.timesteps = {}


class Q9Hooks:
    def __init__(
        self,
        pipe,
        blocks,
        cfg,
        classes,
        bank=None,
        reference=None,
        condition="baseline",
        site=-1,
        target_step=-1,
        rescue="none",
        donor=None,
        forced_step=None,
        collect=None,
        capture_only=False,
    ):
        self.pipe, self.blocks, self.cfg, self.classes = pipe, blocks, cfg, classes
        self.bank, self.reference = bank or {}, reference
        self.condition, self.site, self.target_step, self.rescue = (
            condition,
            site,
            target_step,
            rescue,
        )
        self.donor, self.forced_step, self.collect = donor, forced_step, collect
        self.capture_only = capture_only
        self.work = {
            "observations_computed": 0,
            "observations_reused": 0,
            "attention_computed": 0,
            "attention_reused": 0,
            "input_image_transfers": 0,
            "input_masks_reused": 0,
        }
        self.reference_rows = {}
        if reference is not None:
            for row in reference.rows:
                self.reference_rows.setdefault(
                    (row["step"], row["layer"], row["stage"]), []
                ).append(row)
        self.trace, self.step = Trace(), -1
        self.handles, self.processors = [], []
        self.counts = {b.layer_id: 0 for b in blocks}
        self.nt = len(classes)

    def active(self, layer):
        return self.step == self.target_step and layer == self.site

    def clean_prefix(self, layer, before_attention=False):
        """No intervention has fired yet; downstream and later-step diagnostics stay live."""
        return (
            self.cfg.optimize_probes
            and self.reference is not None
            and (
                self.step < self.target_step
                or (
                    self.step == self.target_step
                    and (layer < self.site or (before_attention and layer == self.site))
                )
            )
        )

    def reuse_rows(self, layer, stage):
        self.trace.rows.extend(self.reference_rows.get((self.step, layer, stage), ()))

    def direction(self, layer, stream):
        return self.bank.get(f"{stream}_{layer}", {}).get("vector")

    def row(self, layer, stream, population, values, stage="dit"):
        self.trace.rows.append(
            {
                "stage": stage,
                "step": self.step,
                "layer": layer,
                "stream": stream,
                "population": population,
                **values,
            }
        )

    def record(self, layer, text, image=None):
        cfg, key = self.cfg, (self.step, layer)
        if self.clean_prefix(layer):
            self.trace.frames[key] = self.reference.frames[key]
            self.reuse_rows(layer, "dit")
            self.work["observations_reused"] += 1
            return
        self.work["observations_computed"] += 1
        arr = numpy(text)
        text_norms = np.linalg.norm(arr, axis=-1) if cfg.optimize_probes else None
        sinks = self.trace.inputs.get(key, {}).get("sinks")
        mask, uncapped = select_text(arr, self.classes, cfg, sinks)
        frame = {"text_mask": mask, "text_control": matched_positions(arr, mask, self.classes)}
        if layer in cfg.sites or layer == -1:
            frame["text_full"] = arr.copy()
        ref = self.reference.frames.get(key) if self.reference else None
        v = self.direction(layer, "text")
        ch = self.bank.get(f"text_{layer}", {}).get("channel")
        for cls in np.unique(self.classes):
            take = self.classes == cls
            vals = state_metrics(arr, take, v, ch, text_norms)
            squared = np.square(arr[take])
            channel_energy = squared.sum(0)
            top = np.argsort(channel_energy)[-5:][::-1]
            vals.update(
                top_channels=top.tolist(),
                dominant_channel=int(top[0]),
                dominant_energy=float(channel_energy[top[0]] / max(channel_energy.sum(), 1e-20)),
                norm_excluding_dominant=float(
                    np.sqrt(np.maximum(squared.sum(-1) - arr[take, top[0]] ** 2, 0)).mean()
                ),
            )
            self.row(layer, "text", cls, vals)
        self.row(
            layer,
            "text",
            "candidates",
            state_metrics(arr, mask, v, ch, text_norms)
            | {"uncapped_count": uncapped, "positions": np.flatnonzero(mask).tolist()},
        )
        self.row(
            layer,
            "text",
            "fixed",
            state_metrics(arr, ref["text_mask"] if ref else mask, v, ch, text_norms),
        )
        if self.collect is not None:
            self.collect.setdefault(f"text_{layer}", Reservoir(cfg.reservoir_size)).add(arr[mask])
        if image is not None:
            ia = numpy(image)
            image_norms = np.linalg.norm(ia, axis=-1) if cfg.optimize_probes else None
            im, count = norm_candidates(ia, cfg.image_norm_threshold, cfg.image_max_registers)
            control = matched_positions(ia, im, np.full(len(ia), "image"))
            frame.update(image_mask=im, image_control=control)
            if layer == cfg.rescue_layer:
                frame.update(
                    image_values=ia[im].copy(),
                    control_values=ia[control].copy() if control is not None else None,
                )
            iv = self.direction(layer, "image")
            self.row(
                layer,
                "image",
                "registers",
                state_metrics(ia, im, iv, cfg.image_channel, image_norms)
                | {"uncapped_count": count, "positions": np.flatnonzero(im).tolist()},
            )
            fixed = ref["image_mask"] if ref else im
            self.row(
                layer,
                "image",
                "fixed",
                state_metrics(ia, fixed, iv, cfg.image_channel, image_norms),
            )
            union = np.logical_or(im, fixed).sum()
            self.row(
                layer,
                "image",
                "positions",
                {"jaccard": float((im & fixed).sum() / union) if union else None},
            )
            if self.collect is not None:
                self.collect.setdefault(f"image_{layer}", Reservoir(cfg.reservoir_size)).add(ia[im])
        self.trace.frames[key] = frame

    def state_edit(self, layer, text, image):
        if not self.active(layer) or self.condition in EDGE_METHODS or self.condition == "baseline":
            return text, image
        key = (self.step, layer)
        ref = self.reference.frames[key]
        reverse = self.condition.startswith("image_")
        stream = "image" if reverse else "text"
        method = self.condition.removeprefix("image_") if reverse else self.condition
        mask = ref[f"{stream}_control"] if method == "ordinary_zero" else ref[f"{stream}_mask"]
        if mask is None:
            self.trace.audit.append(
                {
                    "step": self.step,
                    "layer": layer,
                    "status": "no_matched_control",
                    "tokens": 0,
                    "delta_l2": 0.0,
                }
            )
            return text, image
        if reverse and image is None:
            raise RuntimeError("no image stream at projected text input")
        donor = self.donor.frames.get(key, {}).get("text_full") if self.donor else None
        vector = self.direction(layer, stream)
        channel = self.bank.get(f"{stream}_{layer}", {}).get("channel")
        value, audit = edit_state(image if reverse else text, mask, method, vector, channel, donor)
        self.trace.audit.append({"step": self.step, "layer": layer, **audit})
        return (text, value) if reverse else (value, image)

    def attach(self):
        def pre(_m, args, kw):
            if args:
                raise RuntimeError("Q9 replay requires keyword transformer inputs")
            self.step = self.forced_step if self.forced_step is not None else self.step + 1
            self.reuse_interblock_masks = self.cfg.optimize_probes and not any(
                kw.get(name) is not None
                for name in ("controlnet_block_samples", "controlnet_single_block_samples")
            )
            self.trace.timesteps[self.step] = kw["timestep"].detach().float().cpu().tolist()
            if self.step == 0:
                raw = kw["hidden_states"].detach().float().cpu().numpy()
                self.trace.noise_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()
            if kw["hidden_states"].shape[0] != 1:
                raise RuntimeError("Q9 supports batch size 1; no implicit CFG batch editing")
            if (
                self.step in self.cfg.steps
                and self.reference is None
                and self.collect is None
                and not self.capture_only
            ):
                self.trace.snapshots[self.step] = cpu_tree(kw)

        def final(_m, _args, out):
            if self.step in self.cfg.steps and not self.capture_only:
                pred = out[0] if isinstance(out, tuple) else out.sample
                self.trace.predictions[self.step] = pred.detach().cpu()

        def projected(_m, _args, out):
            if self.step not in self.cfg.steps:
                return None
            if self.capture_only:
                if -1 in self.cfg.sites:
                    self.trace.frames[(self.step, -1)] = {"text_full": numpy(out).copy()}
                return None
            text, _ = self.state_edit(-1, out, None)
            self.record(-1, text)
            return text

        self.handles.extend(
            [
                self.pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True),
                self.pipe.transformer.register_forward_hook(final),
                self.pipe.transformer.context_embedder.register_forward_hook(projected),
            ]
        )
        attention_layers = (
            (set(self.cfg.attention_layers) | {l for l in self.cfg.sites if l >= 0})
            if not self.capture_only
            else set()
        )
        for ref in self.blocks:
            layer = ref.layer_id

            def before(_m, args, kw, layer=layer):
                if self.step not in self.cfg.steps or layer not in attention_layers:
                    return
                key = (self.step, layer)
                if self.clean_prefix(layer, before_attention=True):
                    self.trace.inputs[key] = self.reference.inputs[key]
                    return
                image = kw.get("hidden_states", args[0] if args else None)
                text = kw.get("encoder_hidden_states")
                if text is None:
                    text, image = image[:, : self.nt], image[:, self.nt :]
                ta = numpy(text)
                tm, _ = select_text(ta, self.classes, self.cfg)
                previous = self.trace.frames.get((self.step, layer - 1), {})
                if self.reuse_interblock_masks and "image_mask" in previous:
                    im = previous["image_mask"]
                    self.work["input_masks_reused"] += 1
                else:
                    im, _ = norm_candidates(
                        numpy(image), self.cfg.image_norm_threshold, self.cfg.image_max_registers
                    )
                    self.work["input_image_transfers"] += 1
                self.trace.inputs[(self.step, layer)] = {
                    "text_mask": tm,
                    "image_mask": im,
                    "text_array": ta,
                }

            def after(_m, _args, out, layer=layer):
                self.counts[layer] += 1
                if self.step not in self.cfg.steps:
                    return None
                text, image = streams(out, self.nt)
                if self.capture_only:
                    if layer in self.cfg.sites:
                        self.trace.frames[(self.step, layer)] = {"text_full": numpy(text).copy()}
                    return None
                text, image = self.state_edit(layer, text, image)
                if (
                    self.rescue != "none"
                    and self.step == self.target_step
                    and layer == self.cfg.rescue_layer
                ):
                    frame = self.reference.frames[(self.step, layer)]
                    ordinary = self.rescue == "ordinary_projection"
                    mask = frame["image_control" if ordinary else "image_mask"]
                    vector = self.direction(layer, "image")
                    if (
                        mask is None
                        or not mask.any()
                        or ("projection" in self.rescue and vector is None)
                    ):
                        self.trace.audit.append(
                            {
                                "kind": "rescue",
                                "status": "unavailable",
                                "step": self.step,
                                "layer": layer,
                            }
                        )
                    else:
                        previous = image
                        image = rescue_state(
                            image,
                            mask,
                            frame["control_values" if ordinary else "image_values"],
                            vector,
                            self.rescue,
                        )
                        self.trace.audit.append(
                            {
                                "kind": "rescue",
                                "status": "restored",
                                "step": self.step,
                                "layer": layer,
                                "tokens": int(mask.sum()),
                                "delta_l2": float((image.float() - previous.float()).norm()),
                            }
                        )
                self.record(layer, text, image)
                return replace_streams(out, text, image)

            self.handles.append(ref.module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(ref.module.register_forward_hook(after))
            if layer in attention_layers:
                attn = ref.module.attn
                original = attn.processor
                self.processors.append((attn, original))
                attn.set_processor(Q9Attention(original, self, layer))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        for attn, original in self.processors:
            attn.set_processor(original)

    def validate(self, expected):
        if any(c != expected for c in self.counts.values()):
            raise RuntimeError(f"Q9 block invocation mismatch: {self.counts}")
        if self.condition != "baseline":
            edits = [a for a in self.trace.audit if a.get("kind") != "rescue"]
            if (
                len(edits) != 1
                or edits[0]["step"] != self.target_step
                or edits[0]["layer"] != self.site
            ):
                raise RuntimeError(f"Expected one targeted edit attempt, got {edits}")


class Q9Attention:
    def __init__(self, original, hooks, layer):
        self.original, self.hooks, self.layer = original, hooks, layer

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        h = self.hooks
        output = self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )
        if h.step not in h.cfg.steps:
            return output
        if attention_mask is not None:
            raise RuntimeError(
                "Q9 FLUX native path expected no DiT attention mask; refusing to ignore one"
            )
        key = (h.step, self.layer)
        cached = h.clean_prefix(self.layer, before_attention=True)
        if cached:
            h.reuse_rows(self.layer, "attention_input")
            h.work["attention_reused"] += 1
            if not h.active(self.layer) or h.condition not in EDGE_METHODS:
                return output
        q, k, v = q7._flux_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if not cached:
            h.work["attention_computed"] += 1
            frame = h.trace.inputs[key]
            fixed = h.reference.inputs[key] if h.reference else frame
            rows, incoming = attention_reductions(
                q,
                k,
                h.nt,
                frame["image_mask"],
                h.classes,
                h.cfg.query_chunk,
                fixed["image_mask"],
                fixed["text_mask"],
            )
            sinks = sink_mask(incoming, h.cfg)
            frame["sinks"] = sinks
            frame["text_mask"], _ = select_text(frame.pop("text_array"), h.classes, h.cfg, sinks)
            for row in rows:
                row["fixed_text_candidate_mass"] = float(
                    row.pop("_text_incoming")[fixed["text_mask"]].sum()
                )
                h.row(self.layer, "attention", row.pop("population"), row, "attention_input")
            h.row(
                self.layer,
                "attention",
                "text_candidates",
                {
                    "sink_count": int(sinks.sum()),
                    "positions": np.flatnonzero(sinks).tolist(),
                    "overlap_count": int((sinks & frame["text_mask"]).sum()),
                },
                "attention_input",
            )
        if not h.active(self.layer) or h.condition not in EDGE_METHODS:
            return output
        reference = h.reference.inputs[key]
        queries, keys = edge_indices(h.condition, h.nt, q.shape[1] - h.nt, reference, h.classes)
        if not len(queries) or not len(keys):
            h.trace.audit.append(
                {
                    "step": h.step,
                    "layer": self.layer,
                    "status": "no_candidates",
                    "tokens": 0,
                    "delta_l2": 0.0,
                }
            )
            return output
        if len(keys) == k.shape[1]:
            raise RuntimeError("attention intervention would mask every key")
        raw = edge_attention(
            q, k, v, queries, keys, h.condition.rsplit("_", 1)[-1], self.original, h.cfg.query_chunk
        )
        edited = patch_attention_output(
            attn,
            output,
            raw,
            queries,
            h.nt,
            encoder_hidden_states is not None,
            h.condition.endswith("value"),
        )
        import torch

        delta = (
            sum(float((a.float() - b.float()).square().sum()) for a, b in zip(edited, output))
            if isinstance(output, tuple)
            else float((edited.float() - output.float()).square().sum())
        )
        h.trace.audit.append(
            {
                "step": h.step,
                "layer": self.layer,
                "status": "edited" if delta else "zero_effect",
                "tokens": len(keys),
                "queries": len(queries),
                "delta_l2": float(np.sqrt(delta)),
            }
        )
        if not torch.isfinite(raw).all():
            raise RuntimeError("nonfinite attention edit")
        return edited


@contextlib.contextmanager
def installed(hooks):
    try:
        hooks.attach()
        yield hooks
    finally:
        hooks.detach()


def pipeline_run(pipe, cfg, conditioning, seed, latent_only=False):
    import torch

    kwargs = dict(
        conditioning,
        height=cfg.resolution,
        width=cfg.resolution,
        num_inference_steps=PRESETS[cfg.model_preset]["num_steps"],
        guidance_scale=PRESETS[cfg.model_preset]["guidance_scale"],
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent" if latent_only else "pil",
    )
    with torch.inference_mode():
        return pipe(**kwargs).images[0]


def replay(pipe, snapshot):
    import torch

    kwargs = device_tree(snapshot, pipe._execution_device)
    with torch.inference_mode():
        output = pipe.transformer(**kwargs)
    return (output[0] if isinstance(output, tuple) else output.sample).detach().cpu()


def conditioning(pipe, cfg, prompt, q7cfg, cache, t5_rows):
    if prompt in cache:
        return cache[prompt]
    tok = pipe.tokenizer_2(
        prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    ids = tok.input_ids[0].tolist()
    classes = token_classes(
        ids,
        pipe.tokenizer_2.eos_token_id,
        pipe.tokenizer_2.pad_token_id,
        pipe.tokenizer_2.all_special_ids,
    )
    handles = []
    actual_masks = []

    def encoder_pre(_m, args, kw):
        actual_ids = kw.get("input_ids", args[0] if args else None)
        if actual_ids is None or actual_ids[0].tolist() != ids:
            raise RuntimeError("Tokenizer labels do not match the actual T5 input")
        actual_masks.append(cpu_tree(kw.get("attention_mask")))

    handles.append(pipe.text_encoder_2.register_forward_pre_hook(encoder_pre, with_kwargs=True))
    for layer, module in enumerate(pipe.text_encoder_2.encoder.block):

        def enable_attention(_m, args, kw):
            # Collect one native T5 attention matrix at a time; remove it from the block
            # return before the encoder can accumulate a full-depth attention archive.
            kw = dict(kw, output_attentions=True)
            return args, kw

        def observe(_m, _args, out, layer=layer):
            arr = numpy(out[0])
            weights = out[-1]
            if weights.ndim != 4 or weights.shape[-2:] != (len(ids), len(ids)):
                raise RuntimeError("T5 did not expose native per-block attention weights")
            incoming = weights[0].detach().float().mean(1).cpu().numpy()
            selected, _ = select_text(arr, classes, cfg, sink_mask(incoming, cfg))
            sample = Reservoir(cfg.reservoir_size)
            sample.add(arr[selected])
            fitted = sample.fit()
            if fitted:
                t5_rows.append(
                    {
                        "prompt": prompt,
                        "layer": layer,
                        "token_class": "candidates",
                        "direction": fitted,
                        "positions": np.flatnonzero(selected).tolist(),
                    }
                )
            for cls in np.unique(classes):
                take = classes == cls
                energy = np.square(arr[take]).sum(0)
                channel = int(energy.argmax())
                t5_rows.append(
                    {
                        "prompt": prompt,
                        "layer": layer,
                        "token_class": cls,
                        **state_metrics(arr, take, channel=channel),
                        "channel": channel,
                        "incoming_mass": float(incoming[:, take].sum(-1).mean()),
                        "incoming_per_token": float(incoming[:, take].mean()),
                        "norm_excluding_dominant": float(
                            np.sqrt(
                                np.maximum(
                                    np.square(arr[take]).sum(-1) - arr[take, channel] ** 2, 0
                                )
                            ).mean()
                        ),
                    }
                )
            return out[:-1]

        handles.append(module.register_forward_pre_hook(enable_attention, with_kwargs=True))
        handles.append(module.register_forward_hook(observe))
    try:
        tensors = q7._conditioning_for_prompt(pipe, q7cfg, prompt, {})
    finally:
        for handle in handles:
            handle.remove()
    if len(actual_masks) != 1 or tensors["prompt_embeds"].shape[1] != len(ids):
        raise RuntimeError("Unexpected T5 execution count or token length")
    meta = {
        "ids": ids,
        "classes": classes.tolist(),
        "content_length": int((classes == "content").sum()),
        "tokenizer_mask": tok.attention_mask[0].tolist(),
        "encoder_mask": None if actual_masks[0] is None else actual_masks[0][0].tolist(),
        "dit_mask": None,
    }
    cache[prompt] = (tensors, classes, meta)
    return cache[prompt]


def pair_rows(clean, edited, metadata):
    reference = {
        (r["stage"], r["step"], r["layer"], r["stream"], r["population"]): r for r in clean.rows
    }
    paired = []
    for row in edited.rows:
        keys = ("stage", "step", "layer", "stream", "population")
        base = reference.get(tuple(row[k] for k in keys))
        if base is None:
            raise RuntimeError("missing clean readout")
        for metric, value in row.items():
            if metric in keys or metric in {
                "dominant_channel",
                "image_sink_position",
                "text_sink_position",
            }:
                continue
            if isinstance(value, (int, float)) and isinstance(base.get(metric), (int, float)):
                paired.append(
                    metadata
                    | {k: row[k] for k in keys}
                    | {
                        "metric": metric,
                        "clean": base[metric],
                        "edited": value,
                        "delta": value - base[metric],
                    }
                )
        for metric in ("image_sink_position", "text_sink_position"):
            if metric in row and metric in base:
                agreement = float(row[metric] == base[metric])
                paired.append(
                    metadata
                    | {k: row[k] for k in keys}
                    | {
                        "metric": metric + "_agreement",
                        "clean": 1.0,
                        "edited": agreement,
                        "delta": agreement - 1.0,
                    }
                )
    return paired


def birth_summary(trace):
    """Operational channel-specific birth, separately from generic norm outliers."""
    output = []
    for step in sorted({r["step"] for r in trace.rows}):
        rows = [
            r
            for r in trace.rows
            if r["step"] == step and r["stream"] == "image" and r["population"] == "registers"
        ]
        norm_layers = [r["layer"] for r in rows if r["count"] > 0]
        channel_layers = [
            r["layer"] for r in rows if r["count"] > 0 and r.get("channel_energy", 0) >= 0.5
        ]
        output.append(
            {
                "step": step,
                "first_norm_outlier_layer": min(norm_layers) if norm_layers else None,
                "first_channel_dominated_register_layer": min(channel_layers)
                if channel_layers
                else None,
                "channel_dominated_layers": channel_layers,
                "channel_energy_threshold": 0.5,
            }
        )
    return output


def model_metadata(pipe):
    names = ("torch", "diffusers", "transformers", "numpy")
    versions = {name: importlib.metadata.version(name) for name in names}
    return {
        "versions": versions,
        "scheduler": dict(pipe.scheduler.config),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("text_image_coupling.py", "q9_runtime.py")
        },
        "transformer_config": dict(pipe.transformer.config),
        "checkpoint_revision": getattr(pipe.transformer.config, "_commit_hash", None),
        "attention_processors": sorted(
            {type(b.processor).__name__ for b in [r.module.attn for r in q7_model_blocks(pipe)]}
        ),
        "cuda_device": str(pipe._execution_device),
    }


def q7_model_blocks(pipe):
    from src.common.model_utils import discover_blocks

    return discover_blocks(pipe.transformer)


def calibrate(pipe, blocks, cfg, q7cfg, cache, root, provenance):
    path = root / f"calibration_{cfg.calibration_identity()}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["provenance"] != provenance:
            raise RuntimeError("Calibration environment changed; use a fresh calibration directory")
        bank = {
            k: v | {"vector": np.array(v["vector"], np.float32)} for k, v in saved["bank"].items()
        }
        return bank, saved
    collectors, t5_rows, scenario_rows, stability, tokens = {}, [], [], [], {}
    per_prompt, per_seed = {}, {}
    for pi, prompt in enumerate(cfg.calibration_prompts):
        cond, classes, meta = conditioning(pipe, cfg, prompt, q7cfg, cache, t5_rows)
        tokens[prompt] = meta
        local = {}
        for seed in cfg.calibration_seeds:
            print(
                f"Q9 calibration prompt {pi + 1}/{len(cfg.calibration_prompts)}, seed={seed}",
                flush=True,
            )
            scenario_samples = {}
            hooks = Q9Hooks(pipe, blocks, cfg, classes, collect=scenario_samples)
            with installed(hooks):
                pipeline_run(pipe, cfg, cond, seed, latent_only=True)
            hooks.validate(PRESETS[cfg.model_preset]["num_steps"])
            scenario_rows.extend(r | {"prompt_id": pi, "seed": seed} for r in hooks.trace.rows)
            for key, sample in scenario_samples.items():
                if sample.rows:
                    local.setdefault(key, Reservoir(cfg.reservoir_size)).add(sample.rows)
                    per_seed[(pi, seed, key)] = sample.fit()
        for key, sample in local.items():
            if not sample.rows:
                continue
            # Each prompt contributes the same bounded maximum sample budget.
            collectors.setdefault(key, Reservoir(cfg.reservoir_size)).add(sample.rows)
            fitted = sample.fit()
            per_prompt[(pi, key)] = fitted
    bank = {k: fitted for k, sample in collectors.items() if (fitted := sample.fit()) is not None}
    for (pi, key), value in per_prompt.items():
        stability.append(
            {
                "prompt_id": pi,
                "direction": key,
                "energy": value["energy"],
                "channel": value["channel"],
                "sampled": value["sampled"],
                "abs_cosine_pooled": float(abs(value["vector"] @ bank[key]["vector"])),
            }
        )
    for (pi, key), a in per_prompt.items():
        for (pj, other), b in per_prompt.items():
            if key == other and pi < pj:
                stability.append(
                    {
                        "prompt_id": pi,
                        "other_prompt_id": pj,
                        "direction": key,
                        "abs_cosine_pair": float(abs(a["vector"] @ b["vector"])),
                    }
                )
    for (pi, seed, key), a in per_seed.items():
        for (pj, other_seed, other_key), b in per_seed.items():
            if pi == pj and key == other_key and seed < other_seed:
                stability.append(
                    {
                        "prompt_id": pi,
                        "seed": seed,
                        "other_seed": other_seed,
                        "direction": key,
                        "abs_cosine_across_seeds": float(abs(a["vector"] @ b["vector"])),
                    }
                )
    t5_stability = []
    t5_directions = [r for r in t5_rows if "direction" in r]
    for i, a in enumerate(t5_directions):
        for b in t5_directions[i + 1 :]:
            if a["layer"] == b["layer"] and a["prompt"] != b["prompt"]:
                t5_stability.append(
                    {
                        "prompt_a": a["prompt"],
                        "prompt_b": b["prompt"],
                        "layer": a["layer"],
                        "abs_cosine": float(
                            abs(a["direction"]["vector"] @ b["direction"]["vector"])
                        ),
                    }
                )
    saved = {
        "identity": cfg.calibration_identity(),
        "revision": REVISION,
        "provenance": provenance,
        "config": asdict(cfg),
        "bank": bank,
        "stability": stability,
        "t5_stability": t5_stability,
        "tokens": tokens,
        "interpretation": "Pooled single-axis hypotheses; inspect energy/stability before confirmation.",
    }
    write_json(
        root / f"discovery_{cfg.calibration_identity()}.json",
        {"rows": scenario_rows, "t5": t5_rows},
    )
    write_json(path, saved)
    return bank, json.loads(path.read_text())


def job_complete(folder, job_id, cfg):
    path = folder / f"{job_id}.json"
    if not path.is_file():
        return False
    return (
        cfg.mode != "confirm"
        or not cfg.save_images
        or (folder / f"{job_id}.png").is_file()
        or json.loads(path.read_text()).get("execution") == "skipped_unavailable"
    )


def unavailable_job(cfg, clean, bank, classes, donor, method, site, step, rescue):
    """Skip only contrasts whose targets/controls are unavailable in the clean trace."""
    if method in EDGE_METHODS:
        frame = clean.inputs[(step, site)]
        queries, keys = edge_indices(method, len(classes), len(frame["image_mask"]), frame, classes)
        if not len(queries) or not len(keys):
            return "no_candidates"
    else:
        reverse = method.startswith("image_")
        stream = "image" if reverse else "text"
        operation = method.removeprefix("image_") if reverse else method
        frame = clean.frames[(step, site)]
        mask = frame[f"{stream}_control" if operation == "ordinary_zero" else f"{stream}_mask"]
        if mask is None:
            return "no_matched_control"
        if not mask.any():
            return "no_candidates"
        fitted = bank.get(f"{stream}_{site}", {})
        if (
            operation in {"remove_direction", "norm_matched", "random_direction"}
            and fitted.get("vector") is None
        ):
            return "no_direction"
        if operation == "suppress_channel" and fitted.get("channel") is None:
            return "no_direction"
        if operation == "donor_swap" and (donor is None or (step, site) not in donor.frames):
            return "no_matched_donor"
    if rescue != "none":
        frame = clean.frames[(step, cfg.rescue_layer)]
        mask = frame["image_control" if rescue == "ordinary_projection" else "image_mask"]
        if mask is None or not mask.any():
            return "unavailable_rescue"
        if (
            "projection" in rescue
            and bank.get(f"image_{cfg.rescue_layer}", {}).get("vector") is None
        ):
            return "unavailable_rescue"
    return None


def run(cfg: Q9Config):
    import torch

    cfg.validate()
    q7cfg = q7.Q7Config(
        **PRESETS[cfg.model_preset],
        prompts=tuple(cfg.prompts),
        resolution=cfg.resolution,
        dtype=cfg.dtype,
        device=cfg.device,
        offload=cfg.offload,
    )
    root = Path(cfg.output_dir)
    if "/drive/" in root.as_posix().lower():
        raise ValueError("Q9 output_dir must be local Colab storage; use compact export for Drive")
    root.mkdir(parents=True, exist_ok=True)
    pipe, blocks = q7._load_q7_model(q7cfg)
    provenance = model_metadata(pipe)
    provenance = json.loads(json.dumps(provenance, default=json_value))
    cache = {}
    calibration_root = Path(cfg.calibration_dir) if cfg.calibration_dir else root / "calibration"
    bank, calibration = calibrate(pipe, blocks, cfg, q7cfg, cache, calibration_root, provenance)
    # Calibration conditioning is no longer needed; do not accumulate T5 embeddings for
    # calibration plus every evaluation prompt on the GPU at the same time.
    cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    identity = fingerprint(
        {
            "config": asdict(cfg),
            "calibration": fingerprint(calibration),
            "provenance": provenance,
            "revision": REVISION,
        }
    )
    run_root = root / "runs" / identity
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"latest_{cfg.mode}.json", {"run_root": str(run_root), "identity": identity})
    write_json(run_root / "config.json", asdict(cfg))
    write_json(run_root / "calibration.json", calibration)
    write_json(run_root / "provenance.json", provenance)
    if cfg.mode == "discovery":
        from .q9_report import report

        report(cfg)
        return
    t5_rows = []
    prompts = cfg.prompts + ([""] if cfg.include_empty and "" not in cfg.prompts else [])
    all_meta = {}
    for prompt in prompts:
        _cond, _classes, meta = conditioning(pipe, cfg, prompt, q7cfg, cache, t5_rows)
        all_meta[prompt] = meta
    write_json(run_root / "tokens.json", all_meta)
    write_json(run_root / "t5.json", t5_rows)
    for pi, prompt in enumerate(prompts):
        cond, classes, meta = cache[prompt]
        donor_prompt = next(
            (
                p
                for p in cfg.prompts
                if p != prompt
                and all_meta[p]["classes"] == meta["classes"]
                and all_meta[p]["encoder_mask"] == meta["encoder_mask"]
            ),
            None,
        )
        for seed in cfg.seeds:
            folder = run_root / f"p{pi:03d}_s{seed}"
            folder.mkdir(exist_ok=True)
            jobs = [
                (method, site, step, rescue)
                for site in cfg.sites
                for step in cfg.steps
                for method in cfg.methods
                for rescue in cfg.rescues
                if rescue == "none" or method == "remove_direction"
                if not (site == -1 and (method in EDGE_METHODS or method.startswith("image_")))
            ]
            # Native empty prompt is a clean reference only; it is not a length-matched causal pair.
            if prompt == "":
                jobs = []
            expected = [fingerprint(job) for job in jobs]
            baseline_complete = (folder / "baseline.json").exists() and (
                cfg.mode != "confirm" or not cfg.save_images or (folder / "baseline.png").exists()
            )
            if baseline_complete and all(job_complete(folder, j, cfg) for j in expected):
                print(f"Q9 resume p={pi} seed={seed}: complete", flush=True)
                continue
            print(
                f"Q9 {cfg.mode}: prompt {pi + 1}/{len(prompts)}, seed={seed}, {len(jobs)} jobs",
                flush=True,
            )
            start = time.monotonic()
            baseline_hooks = Q9Hooks(pipe, blocks, cfg, classes, bank)
            with installed(baseline_hooks):
                baseline_image = pipeline_run(
                    pipe, cfg, cond, seed, latent_only=cfg.mode != "confirm"
                )
            baseline_hooks.validate(PRESETS[cfg.model_preset]["num_steps"])
            clean = baseline_hooks.trace
            if cfg.mode == "confirm" and cfg.save_images:
                baseline_image.save(folder / "baseline.png")
            replay_checks = {}
            for step in cfg.steps:
                repeated = replay(pipe, clean.snapshots[step])
                delta = float((repeated.float() - clean.predictions[step].float()).abs().max())
                if not torch.allclose(repeated, clean.predictions[step], rtol=1e-3, atol=1e-3):
                    raise RuntimeError(f"Clean replay mismatch at step {step}: max abs {delta}")
                replay_checks[step] = delta
            write_json(
                folder / "baseline.json",
                {
                    "prompt": prompt,
                    "prompt_id": pi,
                    "seed": seed,
                    "rows": clean.rows,
                    "replay_max_abs": replay_checks,
                    "birth": birth_summary(clean),
                    "noise_sha256": clean.noise_sha256,
                    "transformer_timesteps": clean.timesteps,
                    "scheduler_timesteps": cpu_tree(pipe.scheduler.timesteps).tolist(),
                    "scheduler_sigmas": cpu_tree(pipe.scheduler.sigmas).tolist()
                    if hasattr(pipe.scheduler, "sigmas")
                    else None,
                    "elapsed_seconds": time.monotonic() - start,
                    "identity": identity,
                },
            )
            donor = None
            if donor_prompt is not None and "donor_swap" in cfg.methods:
                # Donor text states are obtained with recipient image latent and pooled CLIP.
                donor = Trace()
                donor_cond = cache[donor_prompt][0]
                for step in cfg.steps:
                    snapshot = cpu_tree(clean.snapshots[step])
                    snapshot["encoder_hidden_states"] = cpu_tree(donor_cond["prompt_embeds"])
                    dh = Q9Hooks(
                        pipe,
                        blocks,
                        cfg,
                        classes,
                        bank,
                        forced_step=step,
                        capture_only=cfg.optimize_probes,
                    )
                    with installed(dh):
                        replay(pipe, snapshot)
                    dh.validate(1)
                    donor.frames.update(dh.trace.frames)
            for index, (method, site, step, rescue) in enumerate(jobs):
                job_id = fingerprint((method, site, step, rescue))
                path = folder / f"{job_id}.json"
                if job_complete(folder, job_id, cfg):
                    continue
                started = time.monotonic()
                metadata = {
                    "identity": identity,
                    "prompt_id": pi,
                    "seed": seed,
                    "condition": method,
                    "site": site,
                    "target_step": step,
                    "rescue": rescue,
                }
                unavailable = (
                    unavailable_job(cfg, clean, bank, classes, donor, method, site, step, rescue)
                    if cfg.skip_unavailable
                    else None
                )
                if unavailable:
                    write_json(
                        path,
                        metadata
                        | {
                            "prompt": prompt,
                            "donor_prompt": donor_prompt,
                            "execution": "skipped_unavailable",
                            "audit": [
                                {
                                    "step": step,
                                    "layer": site,
                                    "status": unavailable,
                                    "tokens": 0,
                                    "delta_l2": 0.0,
                                    "model_forwards": 0,
                                }
                            ],
                            "paired": [],
                            "birth": [],
                            "positions": [],
                            "image_metrics": {},
                            "work": {"model_forwards": 0},
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
                    print(
                        f"  {index + 1}/{len(jobs)} {method} l={site} t={step}: skipped ({unavailable})",
                        flush=True,
                    )
                    continue
                hooks = Q9Hooks(
                    pipe,
                    blocks,
                    cfg,
                    classes,
                    bank,
                    clean,
                    method,
                    site,
                    step,
                    rescue,
                    donor,
                    forced_step=step if cfg.mode != "confirm" else None,
                )
                with installed(hooks):
                    if cfg.mode == "confirm":
                        edited_image = pipeline_run(pipe, cfg, cond, seed)
                        prediction = hooks.trace.predictions[step]
                    else:
                        prediction = replay(pipe, clean.snapshots[step])
                hooks.validate(
                    PRESETS[cfg.model_preset]["num_steps"] if cfg.mode == "confirm" else 1
                )
                if cfg.mode == "confirm" and hooks.trace.noise_sha256 != clean.noise_sha256:
                    raise RuntimeError("Paired generation initial noise mismatch")
                paired = pair_rows(clean, hooks.trace, metadata)
                diff = prediction.float() - clean.predictions[step].float()
                paired.append(
                    metadata
                    | {
                        "stage": "prediction",
                        "step": step,
                        "layer": 57,
                        "stream": "image",
                        "population": "all",
                        "metric": "prediction_relative_l2",
                        "clean": 0.0,
                        "edited": float(
                            diff.norm() / clean.predictions[step].float().norm().clamp_min(1e-20)
                        ),
                        "delta": float(
                            diff.norm() / clean.predictions[step].float().norm().clamp_min(1e-20)
                        ),
                    }
                )
                image_metrics = {}
                if cfg.mode == "confirm":
                    if cfg.save_images:
                        edited_image.save(folder / f"{job_id}.png")
                    image_metrics = q7.frequency_distances(
                        np.asarray(baseline_image), np.asarray(edited_image)
                    )
                write_json(
                    path,
                    metadata
                    | {
                        "prompt": prompt,
                        "donor_prompt": donor_prompt,
                        "audit": hooks.trace.audit,
                        "work": hooks.work,
                        "paired": paired,
                        "birth": birth_summary(hooks.trace),
                        "positions": [r for r in hooks.trace.rows if "positions" in r],
                        "image_metrics": image_metrics,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                print(
                    f"  {index + 1}/{len(jobs)} {method} l={site} t={step} rescue={rescue}: "
                    f"{hooks.trace.audit[0]['status']} ({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
            del clean, baseline_hooks, donor
    from .q9_report import report

    report(cfg)
    if cfg.mode == "smoke":
        status = json.loads((run_root / "report_status.json").read_text())
        if status["effective_jobs"] == 0:
            raise RuntimeError(
                "Smoke found no effective interventions. Inspect candidate selection "
                "and calibration before launching a larger run."
            )
