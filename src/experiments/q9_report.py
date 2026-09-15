"""Compact Q9 reports. Never export raw activations or full generated-image grids."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .text_image_coupling import cluster_summary


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def locate(cfg):
    pointer = Path(cfg.output_dir) / f"latest_{cfg.mode}.json"
    root = Path(json.loads(pointer.read_text())["run_root"])
    if root.resolve().parent.parent != Path(cfg.output_dir).resolve():
        raise ValueError("run pointer escapes configured output directory")
    return root


def image_evaluation(cfg, root, manifest):
    """Explicit optional metrics; cache results so replotting is model-free."""
    path = root / "image_metrics.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    pending = [r for r in manifest if r["job_id"] not in saved and r.get("image_path")]
    if not pending:
        return list(saved.values())
    import torch
    from PIL import Image

    from .q9_runtime import write_json

    lpips_model = clip_model = preprocess = tokenizer = None
    if cfg.evaluate_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval()
    if cfg.evaluate_clip:
        import open_clip

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clean_cache = {}
    with torch.inference_mode():
        for row in pending:
            cp = row["clean_path"]
            if cp not in clean_cache:
                with Image.open(cp) as im:
                    clean = im.convert("RGB").copy()
                clean_tensor = (
                    torch.from_numpy(np.array(clean)).permute(2, 0, 1)[None].float() / 127.5 - 1
                )
                if clip_model:
                    text = clip_model.encode_text(tokenizer([row["prompt"]]))
                    text = text / text.norm(dim=-1, keepdim=True)
                    embedding = clip_model.encode_image(preprocess(clean)[None])
                    clean_score = float(
                        (embedding / embedding.norm(dim=-1, keepdim=True) * text).sum()
                    )
                else:
                    text, clean_score = None, None
                clean_cache[cp] = (clean_tensor, text, clean_score)
            clean_tensor, text, clean_score = clean_cache[cp]
            with Image.open(row["image_path"]) as im:
                edited = im.convert("RGB").copy()
            result = {
                k: row[k]
                for k in (
                    "job_id",
                    "condition",
                    "site",
                    "target_step",
                    "rescue",
                    "prompt_id",
                    "seed",
                )
            }
            result.update(row["image_metrics"])
            if lpips_model:
                et = torch.from_numpy(np.array(edited)).permute(2, 0, 1)[None].float() / 127.5 - 1
                result["lpips"] = float(lpips_model(clean_tensor, et).item())
            if clip_model:
                emb = clip_model.encode_image(preprocess(edited)[None])
                score = float((emb / emb.norm(dim=-1, keepdim=True) * text).sum())
                result.update(
                    clip_clean=clean_score, clip_edited=score, clip_delta=score - clean_score
                )
            saved[row["job_id"]] = result
    write_json(path, saved)
    return list(saved.values())


def report(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .q9_runtime import write_json

    root = locate(cfg)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    calibration = json.loads((root / "calibration.json").read_text())
    csv_write(root / "direction_stability.csv", calibration["stability"])
    csv_write(root / "t5_direction_stability.csv", calibration.get("t5_stability", []))
    direction_rows = [
        {"direction": k, **{a: b for a, b in v.items() if a != "vector"}}
        for k, v in calibration["bank"].items()
    ]
    csv_write(root / "directions.csv", direction_rows)
    if direction_rows:
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        for stream in ("text", "image"):
            selected = sorted(
                [r for r in direction_rows if r["direction"].startswith(stream)],
                key=lambda r: int(r["direction"].split("_")[-1]),
            )
            ax.plot(
                [int(r["direction"].split("_")[-1]) for r in selected],
                [r["energy"] for r in selected],
                label=stream,
            )
        ax.set(
            xlabel="DiT layer (-1 = projected T5)",
            ylabel="Leading direction energy fraction",
            title="Calibration: does one shared direction explain the candidates?",
        )
        ax.legend()
        fig.savefig(figures / "q9_direction_energy.png", dpi=160)
        plt.close(fig)
    manifest, audits, births, t5_flat = [], [], [], []
    if (root / "t5.json").exists():
        for r in json.loads((root / "t5.json").read_text()):
            entry = {k: v for k, v in r.items() if k != "direction"}
            if "direction" in r:
                entry.update({k: v for k, v in r["direction"].items() if k != "vector"})
            t5_flat.append(entry)
        csv_write(root / "t5_summary.csv", t5_flat)
    for folder in sorted(root.glob("p*_s*")):
        if not (folder / "baseline.json").exists():
            continue
        baseline = json.loads((folder / "baseline.json").read_text())
        births.extend(
            {
                "prompt_id": baseline["prompt_id"],
                "seed": baseline["seed"],
                "condition": "baseline",
                **b,
            }
            for b in baseline.get("birth", [])
        )
        for path in sorted(folder.glob("*.json")):
            if path.name == "baseline.json":
                continue
            item = json.loads(path.read_text())
            if "paired" not in item:
                continue
            job_id = folder.name + "_" + path.stem
            target = [a for a in item["audit"] if a.get("kind") != "rescue"]
            effective = len(target) == 1 and target[0]["status"] == "edited"
            rescued = [a for a in item["audit"] if a.get("kind") == "rescue"]
            if item["rescue"] != "none" and (
                len(rescued) != 1 or rescued[0]["status"] != "restored"
            ):
                effective = False
            meta = {
                k: item[k]
                for k in (
                    "identity",
                    "condition",
                    "site",
                    "target_step",
                    "rescue",
                    "prompt_id",
                    "seed",
                    "prompt",
                )
            }
            image_path = folder / (path.stem + ".png")
            meta.update(
                job_id=job_id,
                status=target[0]["status"] if target else "missing_audit",
                eligible_for_summary=effective,
                json_path=str(path),
                clean_path=str(folder / "baseline.png"),
                image_path=str(image_path) if image_path.exists() else None,
                image_metrics=item["image_metrics"],
                elapsed_seconds=item["elapsed_seconds"],
                execution=item.get("execution", "executed"),
                work=item.get("work", {}),
            )
            manifest.append(meta)
            audits.extend({"job_id": job_id, **a} for a in item["audit"])
            births.extend(
                {
                    "job_id": job_id,
                    "condition": item["condition"],
                    "rescue": item["rescue"],
                    "site": item["site"],
                    "target_step": item["target_step"],
                    "prompt_id": item["prompt_id"],
                    "seed": item["seed"],
                    **b,
                }
                for b in item.get("birth", [])
            )
        # compact baseline token-class/time/depth atlas, including the native empty reference
        csv_write(root / f"clean_{folder.name}.csv", baseline["rows"])
    write_json(root / "manifest.json", manifest)
    csv_write(root / "audits.csv", audits)
    csv_write(root / "birth.csv", births)
    if manifest:

        def rows():
            with gzip.open(root / "paired_metrics.csv.gz", "wt", newline="", encoding="utf-8") as f:
                writer = None
                for item in manifest:
                    data = json.loads(Path(item["json_path"]).read_text())
                    for row in data["paired"]:
                        row = row | {"eligible_for_summary": item["eligible_for_summary"]}
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        if item["eligible_for_summary"]:
                            yield row

        summary = cluster_summary(rows())
        csv_write(root / "summary.csv", summary)
        # The selected layer is after the intervention; avoid claiming the edited-site change
        # or a rescue-site overwrite itself as evidence of causal propagation.
        for stream, metric, filename in (
            ("image", "channel_energy", "q9_text_to_image.png"),
            ("text", "alignment_squared", "q9_image_to_text.png"),
        ):
            selected = [
                r
                for r in summary
                if r["stream"] == stream
                and r["metric"] == metric
                and r["population"] == "fixed"
                and r["layer"] > max(r["site"], cfg.rescue_layer if r["rescue"] != "none" else -1)
                and r["step"] == r["target_step"]
            ]
            reverse = {"image_zero", "image_ordinary_zero", "image_remove_direction"}
            selected = [
                r
                for r in selected
                if (
                    r["condition"] not in reverse
                    if stream == "image"
                    else r["condition"] in reverse
                    or r["condition"].startswith("text_reads_register")
                )
            ]
            if not selected:
                continue
            # Keep the inline curve legible. The map and CSV cover every measured cell.
            example_site, example_step = min((r["site"], r["target_step"]) for r in selected)
            selected = [
                r for r in selected if (r["site"], r["target_step"]) == (example_site, example_step)
            ]
            fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
            groups = {}
            for r in selected:
                key = (r["condition"], r["site"], r["target_step"], r["rescue"])
                groups.setdefault(key, []).append(r)
            for key, points in groups.items():
                points.sort(key=lambda r: r["layer"])
                ax.plot(
                    [p["layer"] for p in points],
                    [p["mean_delta"] for p in points],
                    label=f"{key[0]} L{key[1]} t{key[2]} {key[3]}",
                )
            ax.axhline(0, color="gray", lw=0.7)
            ax.set(
                xlabel="Downstream DiT layer",
                ylabel=f"Edited - clean {metric}",
                title=f"{stream.capitalize()} response: example site {example_site}, step {example_step}",
            )
            ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
            fig.savefig(figures / filename, dpi=160)
            plt.close(fig)
        # Endpoint maps keep the dose (one site/step) explicit, rather than pooling stages.
        mapped = [
            r
            for r in summary
            if r["stream"] == "image"
            and r["population"] == "fixed"
            and r["metric"] == "channel_energy"
            and r["layer"] == cfg.readout_layer
            and r["step"] == r["target_step"]
            and r["rescue"] == "none"
        ]
        conditions = sorted({r["condition"] for r in mapped})
        if conditions:
            columns = min(4, len(conditions))
            nrows = (len(conditions) + columns - 1) // columns
            fig, axes = plt.subplots(
                nrows,
                columns,
                figsize=(3.5 * columns, 3.5 * nrows),
                squeeze=False,
                layout="constrained",
            )
            vmax = max(max(abs(r["mean_delta"]) for r in mapped), 1e-8)
            for ax in axes.ravel()[len(conditions) :]:
                ax.set_visible(False)
            for ax, method in zip(axes.ravel(), conditions):
                values = np.full((len(cfg.steps), len(cfg.sites)), np.nan)
                for r in mapped:
                    if r["condition"] == method:
                        values[cfg.steps.index(r["target_step"]), cfg.sites.index(r["site"])] = r[
                            "mean_delta"
                        ]
                im = ax.imshow(
                    np.ma.masked_invalid(values),
                    vmin=-vmax,
                    vmax=vmax,
                    cmap="coolwarm",
                    aspect="auto",
                )
                ax.set_xticks(range(len(cfg.sites)), cfg.sites)
                ax.set_yticks(range(len(cfg.steps)), cfg.steps)
                ax.set(title=method, xlabel="Intervention layer", ylabel="Denoising step")
            fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                label=f"Block {cfg.readout_layer} channel-energy change at clean registers",
            )
            fig.savefig(figures / "q9_causal_map.png", dpi=150)
            plt.close(fig)
        equivalence = []
        if cfg.equivalence_bound is not None:
            for r in summary:
                if (
                    r["stream"] == "image"
                    and r["metric"] == "channel_energy"
                    and r["population"] == "fixed"
                ):
                    equivalence.append(
                        r
                        | {
                            "bound": cfg.equivalence_bound,
                            "within_bound": r["ci_low"] is not None
                            and r["ci_low"] > -cfg.equivalence_bound
                            and r["ci_high"] < cfg.equivalence_bound,
                        }
                    )
            csv_write(root / "equivalence.csv", equivalence)
        images = image_evaluation(cfg, root, manifest)
        # Frequency/structured evaluation is still available if optional learned metrics
        # were disabled and generated PNGs were not retained.
        known = {r["job_id"] for r in images}
        images.extend(
            {
                k: m[k]
                for k in (
                    "job_id",
                    "condition",
                    "site",
                    "target_step",
                    "rescue",
                    "prompt_id",
                    "seed",
                )
            }
            | m["image_metrics"]
            for m in manifest
            if m["job_id"] not in known and m["image_metrics"]
        )
        if cfg.structured_scores:
            with Path(cfg.structured_scores).open(newline="") as f:
                scores = list(csv.DictReader(f))
            mapping = {r["job_id"]: r for r in scores}
            if len(mapping) != len(scores) or set(mapping) - {r["job_id"] for r in manifest}:
                raise ValueError("Structured score keys must be unique and match this run")
            identity = manifest[0]["identity"]
            for r in images:
                supplied = mapping.get(r["job_id"])
                if supplied:
                    if supplied.get("identity") != identity:
                        raise ValueError("Structured evaluator run identity mismatch")
                    for metric in ("counting", "attribute", "spatial", "overall", "image_reward"):
                        a, b = supplied.get(metric + "_clean"), supplied.get(metric + "_edited")
                        if a not in (None, "") and b not in (None, ""):
                            a, b = float(a), float(b)
                            if not np.isfinite([a, b]).all() or (
                                metric != "image_reward" and not (0 <= a <= 1 and 0 <= b <= 1)
                            ):
                                raise ValueError("Invalid structured score")
                            r.update(
                                {
                                    metric + "_clean": a,
                                    metric + "_edited": b,
                                    metric + "_delta": b - a,
                                }
                            )
        csv_write(root / "image_metrics.csv", images)
        eligible = {m["job_id"] for m in manifest if m["eligible_for_summary"]}
        image_pairs = []
        for item in images:
            if item["job_id"] not in eligible:
                continue
            for metric in (
                "lpips",
                "clip_delta",
                "counting_delta",
                "attribute_delta",
                "spatial_delta",
                "overall_delta",
                "image_reward_delta",
                "low_frequency_rms",
                "high_frequency_rms",
            ):
                if metric in item:
                    image_pairs.append(
                        {
                            k: item[k]
                            for k in ("condition", "site", "target_step", "rescue", "prompt_id")
                        }
                        | {
                            "stage": "final_image",
                            "step": -1,
                            "layer": -1,
                            "stream": "image",
                            "population": "all",
                            "metric": metric,
                            "delta": item[metric],
                        }
                    )
        csv_write(root / "image_summary.csv", cluster_summary(image_pairs))
        available = [m for m in manifest if m.get("image_path")]
        if available:
            from PIL import Image, ImageDraw

            first = available[0]
            chosen = [
                m
                for m in available
                if m["prompt_id"] == first["prompt_id"]
                and m["seed"] == first["seed"]
                and m["target_step"] == first["target_step"]
            ][:12]
            sheet = Image.new("RGB", (512, 290 * len(chosen)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, item in enumerate(chosen):
                draw.text((4, i * 290), f"{item['condition']} / {item['rescue']}", fill="black")
                draw.text((4, i * 290 + 15), "clean                          edited", fill="black")
                for col, key in enumerate(("clean_path", "image_path")):
                    with Image.open(item[key]) as im:
                        sheet.paste(im.convert("RGB").resize((256, 256)), (col * 256, i * 290 + 34))
            sheet.save(figures / "q9_example_pairs.png")
    # A class-based atlas from the first available baseline, or calibration for discovery.
    baselines = sorted(root.glob("p*_s*/baseline.json"))
    if baselines:
        atlas = json.loads(baselines[0].read_text())["rows"]
    else:
        calroot = (
            Path(cfg.calibration_dir)
            if cfg.calibration_dir
            else Path(cfg.output_dir) / "calibration"
        )
        atlas = json.loads((calroot / f"discovery_{cfg.calibration_identity()}.json").read_text())[
            "rows"
        ]
        atlas = [r for r in atlas if r["prompt_id"] == 0 and r["seed"] == cfg.calibration_seeds[0]]
    csv_write(root / "example_clean_atlas.csv", atlas)
    for metric, stream, filename in (
        ("selected_norm", "text", "q9_text_norm_atlas.png"),
        ("text_mass", "attention", "q9_routing.png"),
    ):
        rows = [
            r
            for r in atlas
            if r["stream"] == stream
            and metric in r
            and (
                r["population"] in {"content", "eos", "pad", "special"}
                if stream == "text"
                else True
            )
        ]
        if rows:
            fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
            groups = {}
            for r in rows:
                # Mean heads only for this illustrative routing plot; CSV preserves each head.
                group = r["population"].split("/")[0]
                if stream == "attention":
                    for target in ("text", "image"):
                        groups.setdefault(
                            (r["step"], group + " -> " + target + "_key"), {}
                        ).setdefault(r["layer"], []).append(r[target + "_mass"])
                else:
                    groups.setdefault((r["step"], group), {}).setdefault(r["layer"], []).append(
                        r[metric]
                    )
            for (step, group), bylayer in groups.items():
                layers = sorted(bylayer)
                ax.plot(layers, [np.mean(bylayer[l]) for l in layers], label=f"{group}, t={step}")
            if stream == "text":
                ax.set_yscale("log")
            ax.set(
                xlabel="DiT layer",
                ylabel="Mean attention mass" if stream == "attention" else "Mean token L2 norm",
                title="Example clean trajectory (not an aggregate claim)",
            )
            ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
            fig.savefig(figures / filename, dpi=160)
            plt.close(fig)
    write_json(
        root / "report_status.json",
        {
            "jobs": len(manifest),
            "effective_jobs": sum(m["eligible_for_summary"] for m in manifest),
            "status_counts": dict(Counter(m["status"] for m in manifest)),
            "control_note": "ordinary_zero requires a distinct noncandidate token of the same "
            "class for every selected token. A selected singleton EOS has no such match; "
            "no_matched_control is missing evidence for candidate specificity.",
            "calibrated_low_energy_directions": [
                r["direction"] for r in direction_rows if r["energy"] < 0.5
            ],
            "note": "No-candidate/missing-control jobs are excluded, not interpreted as causal nulls. "
            "CIs cluster by prompt. n=1 has no CI. Exploratory curves have no multiplicity claim.",
        },
    )
    print(f"Q9 reports: {root}", flush=True)


def export_compact(cfg, destination, max_bytes=250 * 1024**2):
    root = locate(cfg)
    names = (
        "config.json",
        "calibration.json",
        "provenance.json",
        "tokens.json",
        "t5.json",
        "manifest.json",
        "audits.csv",
        "directions.csv",
        "direction_stability.csv",
        "t5_direction_stability.csv",
        "paired_metrics.csv.gz",
        "summary.csv",
        "image_metrics.csv",
        "image_summary.csv",
        "equivalence.csv",
        "birth.csv",
        "t5_summary.csv",
        "example_clean_atlas.csv",
        "report_status.json",
    )
    sources = [root / name for name in names if (root / name).is_file()]
    sources += list((root / "figures").glob("*.png"))
    size = sum(p.stat().st_size for p in sources)
    if size > max_bytes:
        raise RuntimeError(f"Compact export is {size / 1024**2:.1f} MiB, exceeding 250 MiB budget")
    target = Path(destination) / root.name
    for source in sources:
        out = target / source.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out)
    print(f"Exported {size / 1024**2:.1f} MiB to {target}; no activations or full image grid.")
