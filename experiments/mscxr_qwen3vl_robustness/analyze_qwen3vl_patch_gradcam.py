#!/usr/bin/env python3
"""Patch-level Grad-CAM attribution for Qwen3-VL MS-CXR localization outputs.

This script does not occlude or otherwise modify the image beyond the original
robustness conditions. The default attribution is computed on an internal
vision encoder block output, which is closer to Grad-CAM over visual patches
than the older pixel-gradient smoke test.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_qwen3vl_attention_shift as att
import run_mscxr_qwen3vl_robustness as exp


DEFAULT_EXPERIMENT_DIR = Path("/home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_robustness_paper_perturbations")
DEFAULT_MODEL_PATH = Path("/data/yanjiangtao/yanjiangtao/checkpoints/Qwen3-VL-8B-Instruct")
DEFAULT_DATA_ROOT = Path("/home/yanjiangtao/Med-Grounding/data/MS-CXR")
DEFAULT_OUTPUT_DIR = Path("/home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_patch_gradcam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", default=[], help="Explicit case as sample_id:condition.")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target-token-scope", choices=["numeric", "all"], default="numeric")
    parser.add_argument(
        "--cam-source",
        choices=["visual_token", "vision_feature", "pixel_grad"],
        default="visual_token",
        help=(
            "visual_token hooks the merged visual tokens consumed by the language model; "
            "vision_feature hooks an internal visual encoder block; pixel_grad keeps the input-gradient baseline."
        ),
    )
    parser.add_argument(
        "--vision-layer",
        type=int,
        default=-1,
        help="Visual encoder block index for feature Grad-CAM. Negative values count from the end.",
    )
    parser.add_argument(
        "--attribution-mode",
        choices=["gradcam", "grad_x_activation_abs", "grad_x_activation_relu", "grad_norm"],
        default="grad_x_activation_abs",
    )
    parser.add_argument("--top-heatmap-frac", type=float, default=0.10)
    parser.add_argument("--viz-percentile", type=float, default=99.0)
    parser.add_argument("--viz-threshold-percentile", type=float, default=70.0)
    parser.add_argument("--viz-smooth-sigma", type=float, default=1.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def token_positions(processor: Any, target_ids: torch.Tensor, scope: str) -> torch.Tensor:
    positions = torch.arange(target_ids.shape[1], device=target_ids.device)
    if scope == "all":
        return positions
    tokens = processor.tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    selected = [idx for idx, token in enumerate(tokens) if any(ch.isdigit() for ch in token)]
    if not selected:
        return positions
    return torch.tensor(selected, dtype=torch.long, device=target_ids.device)


def image_for_condition(data_root: Path, experiment_dir: Path, record: dict[str, Any]) -> Image.Image:
    if record["condition"] == "clean_1":
        path = data_root / "images" / record["path"]
    else:
        path = experiment_dir / "perturbed_images" / record["condition"] / record["path"]
    return Image.open(path).convert("RGB")


def normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def normalized_layer_index(layer: int, layer_count: int) -> int:
    if layer < 0:
        layer = layer_count + layer
    if layer < 0 or layer >= layer_count:
        raise ValueError(f"vision layer {layer} is outside [0, {layer_count - 1}]")
    return layer


def build_inputs_with_target(
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    inputs = att.prepare_inputs(processor, image, prompt)
    target_ids = processor.tokenizer(
        target_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, :max_new_tokens]
    if target_ids.numel() == 0:
        raise ValueError("Target text produced no tokens.")

    prompt_len = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat([inputs["input_ids"], target_ids], dim=1)
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(target_ids, dtype=inputs["attention_mask"].dtype)],
        dim=1,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(target_ids, dtype=inputs["mm_token_type_ids"].dtype)],
            dim=1,
        )
    return inputs, target_ids, prompt_len


def target_logprob_score(
    model: Any,
    processor: Any,
    inputs: dict[str, torch.Tensor],
    target_ids_cpu: torch.Tensor,
    target_token_scope: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_ids = target_ids_cpu.to(model.device)
    selected = token_positions(processor, target_ids, target_token_scope)

    att.set_text_attention_impl(model, "sdpa")
    outputs = model(
        **inputs,
        use_cache=False,
        return_dict=True,
        logits_to_keep=target_ids.shape[1] + 1,
    )
    logits = outputs.logits[0, selected, :]
    selected_ids = target_ids[0, selected]
    token_log_probs = torch.log_softmax(logits.float(), dim=-1)[torch.arange(selected.numel(), device=model.device), selected_ids]
    score = token_log_probs.mean()
    return score, {
        "target_token_count": int(target_ids.shape[1]),
        "attributed_token_count": int(selected.numel()),
        "mean_target_logprob": float(score.detach().cpu().item()),
    }


def pixel_gradcam(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
    target_token_scope: str,
    attribution_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs, target_ids_cpu, _ = build_inputs_with_target(processor, image, prompt, target_text, max_new_tokens)
    grid_t, grid_h, grid_w = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError(f"Expected a single image grid, got T={grid_t}.")

    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    pixel_values = inputs["pixel_values"].to(model.dtype).detach().clone().requires_grad_(True)
    inputs["pixel_values"] = pixel_values

    model.zero_grad(set_to_none=True)
    score, meta = target_logprob_score(model, processor, inputs, target_ids_cpu, target_token_scope)
    score.backward()

    grad = pixel_values.grad.detach().float()
    activation = pixel_values.detach().float()
    if attribution_mode == "grad_norm":
        attribution = grad.norm(dim=-1)
    elif attribution_mode in {"grad_x_activation_relu", "gradcam"}:
        attribution = torch.relu((grad * activation).sum(dim=-1))
    else:
        attribution = (grad * activation).sum(dim=-1).abs()

    heatmap = attribution.cpu().numpy().reshape(grid_h, grid_w)
    heatmap = normalize_positive(heatmap)
    meta.update({
        "target_text": target_text,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "cam_source": "pixel_grad",
        "vision_layer": "",
        "input_width": image.width,
        "input_height": image.height,
    })
    return heatmap, meta


def feature_gradcam(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
    target_token_scope: str,
    attribution_mode: str,
    vision_layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs, target_ids_cpu, _ = build_inputs_with_target(processor, image, prompt, target_text, max_new_tokens)
    grid_t, grid_h, grid_w = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError(f"Expected a single image grid, got T={grid_t}.")

    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    pixel_values = inputs["pixel_values"].to(model.dtype).detach().clone().requires_grad_(True)
    inputs["pixel_values"] = pixel_values

    visual = model.model.visual
    layer_idx = normalized_layer_index(vision_layer, len(visual.blocks))
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["activation"] = output
        output.retain_grad()

    handle = visual.blocks[layer_idx].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        score, meta = target_logprob_score(model, processor, inputs, target_ids_cpu, target_token_scope)
        score.backward()
    finally:
        handle.remove()

    activation = captured["activation"].detach().float()
    grad = captured["activation"].grad.detach().float()
    if activation.shape[0] != grid_h * grid_w:
        raise ValueError(
            f"Hooked feature length {activation.shape[0]} does not match patch grid {grid_h}x{grid_w}."
        )

    if attribution_mode == "gradcam":
        channel_weights = grad.mean(dim=0)
        attribution = torch.relu((activation * channel_weights).sum(dim=-1))
    elif attribution_mode == "grad_norm":
        attribution = grad.norm(dim=-1)
    elif attribution_mode == "grad_x_activation_relu":
        attribution = torch.relu((grad * activation).sum(dim=-1))
    else:
        attribution = (grad * activation).sum(dim=-1).abs()

    heatmap = normalize_positive(attribution.cpu().numpy().reshape(grid_h, grid_w))
    meta.update({
        "target_text": target_text,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "cam_source": "vision_feature",
        "vision_layer": str(layer_idx),
        "input_width": image.width,
        "input_height": image.height,
    })
    return heatmap, meta


def visual_token_gradcam(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
    target_token_scope: str,
    attribution_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs, target_ids_cpu, _ = build_inputs_with_target(processor, image, prompt, target_text, max_new_tokens)
    grid_t, grid_h, grid_w = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError(f"Expected a single image grid, got T={grid_t}.")

    merge = int(model.config.vision_config.spatial_merge_size)
    merged_h = grid_h // merge
    merged_w = grid_w // merge
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    pixel_values = inputs["pixel_values"].to(model.dtype).detach().clone().requires_grad_(True)
    inputs["pixel_values"] = pixel_values

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["activation"] = output
        output.retain_grad()

    handle = model.model.visual.merger.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        score, meta = target_logprob_score(model, processor, inputs, target_ids_cpu, target_token_scope)
        score.backward()
    finally:
        handle.remove()

    activation = captured["activation"].detach().float()
    grad = captured["activation"].grad.detach().float()
    if activation.shape[0] != merged_h * merged_w:
        raise ValueError(
            f"Hooked visual token length {activation.shape[0]} does not match merged grid {merged_h}x{merged_w}."
        )

    if attribution_mode == "gradcam":
        channel_weights = grad.mean(dim=0)
        attribution = torch.relu((activation * channel_weights).sum(dim=-1))
    elif attribution_mode == "grad_norm":
        attribution = grad.norm(dim=-1)
    elif attribution_mode == "grad_x_activation_relu":
        attribution = torch.relu((grad * activation).sum(dim=-1))
    else:
        attribution = (grad * activation).sum(dim=-1).abs()

    heatmap = normalize_positive(attribution.cpu().numpy().reshape(merged_h, merged_w))
    meta.update({
        "target_text": target_text,
        "grid_h": merged_h,
        "grid_w": merged_w,
        "cam_source": "visual_token",
        "vision_layer": "merger",
        "input_width": image.width,
        "input_height": image.height,
    })
    return heatmap, meta


def patch_gradcam(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
    target_token_scope: str,
    attribution_mode: str,
    cam_source: str,
    vision_layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cam_source == "pixel_grad":
        return pixel_gradcam(
            model, processor, image, prompt, target_text, max_new_tokens, target_token_scope, attribution_mode
        )
    if cam_source == "visual_token":
        return visual_token_gradcam(
            model, processor, image, prompt, target_text, max_new_tokens, target_token_scope, attribution_mode
        )
    return feature_gradcam(
        model,
        processor,
        image,
        prompt,
        target_text,
        max_new_tokens,
        target_token_scope,
        attribution_mode,
        vision_layer,
    )


def attribution_center(heatmap: np.ndarray, width: int, height: int, condition: str) -> tuple[float, float]:
    mass = normalize_positive(heatmap)
    ys, xs = np.indices(mass.shape)
    cx = float(((xs + 0.5) / mass.shape[1] * width * mass).sum())
    cy = float(((ys + 0.5) / mass.shape[0] * height * mass).sum())
    if condition in {"black_border_10px", "white_border_10px"}:
        cx -= 10.0
        cy -= 10.0
    return cx, cy


def mass_in_box(
    heatmap: np.ndarray,
    box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> float:
    if box is None:
        return 0.0
    mass = normalize_positive(heatmap)
    ys, xs = np.indices(mass.shape)
    cx = (xs + 0.5) / mass.shape[1] * width
    cy = (ys + 0.5) / mass.shape[0] * height
    x1, y1, x2, y2 = box
    mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return float(mass[mask].sum())


def overlay_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    pred_box: tuple[float, float, float, float] | None,
    gt_box: tuple[float, float, float, float] | None,
    title: str,
    output_path: Path,
    percentile: float,
    threshold_percentile: float,
    smooth_sigma: float,
) -> None:
    max_side = 1200
    scale = min(1.0, max_side / max(image.width, image.height))
    display_size = (int(round(image.width * scale)), int(round(image.height * scale)))
    base = np.asarray(image.resize(display_size, Image.Resampling.BILINEAR).convert("RGB"))
    hm = heatmap.astype(np.float32)
    if smooth_sigma > 0:
        hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    hm = cv2.resize(hm, display_size, interpolation=cv2.INTER_CUBIC)
    hi = float(np.percentile(hm, percentile))
    hm = np.clip(hm / max(hi, 1e-12), 0.0, 1.0)
    threshold = float(np.percentile(hm, threshold_percentile))
    alpha = np.clip((hm - threshold) / max(1.0 - threshold, 1e-12), 0.0, 1.0) ** 0.7
    color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (0.70 * alpha)[..., None]
    blended = np.clip((1.0 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(blended)
    draw = ImageDraw.Draw(canvas)
    att.draw_box(draw, gt_box, scale, "lime", max(2, int(3 * scale)))
    att.draw_box(draw, pred_box, scale, "red", max(2, int(3 * scale)))
    draw.rectangle([0, 0, display_size[0], 30], fill=(0, 0, 0))
    draw.text((8, 7), title, fill="white", font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def analyze_record(
    model: Any,
    processor: Any,
    image: Image.Image,
    record: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    case_dir: Path,
) -> dict[str, Any]:
    target_text = att.canonical_bbox_text(record)
    heatmap, meta = patch_gradcam(
        model,
        processor,
        image,
        prompt,
        target_text,
        args.max_new_tokens,
        args.target_token_scope,
        args.attribution_mode,
        args.cam_source,
        args.vision_layer,
    )
    condition = record["condition"]
    np.save(case_dir / f"{condition}_patch_gradcam.npy", heatmap)
    pred_box = att.box_to_input(record["bbox_xyxy"], condition)
    gt_box = att.box_to_input(record["gt_bbox_xyxy"], condition)
    overlay_gradcam(
        image,
        heatmap,
        pred_box,
        gt_box,
        f"{record['sample_id']} {condition} patch Grad-CAM",
        case_dir / f"{condition}_patch_gradcam.jpg",
        args.viz_percentile,
        args.viz_threshold_percentile,
        args.viz_smooth_sigma,
    )
    cx, cy = attribution_center(heatmap, image.width, image.height, condition)
    return {
        "target_text": target_text,
        "cam_source": meta["cam_source"],
        "vision_layer": meta["vision_layer"],
        "mean_target_logprob": meta["mean_target_logprob"],
        "target_token_count": meta["target_token_count"],
        "attributed_token_count": meta["attributed_token_count"],
        "grid": f"{meta['grid_h']}x{meta['grid_w']}",
        "gradcam_cx": cx,
        "gradcam_cy": cy,
        "mass_in_pred_box": mass_in_box(heatmap, pred_box, image.width, image.height),
        "mass_in_gt_box": mass_in_box(heatmap, gt_box, image.width, image.height),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists and is not empty. Use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = att.read_predictions(args.experiment_dir / "predictions.jsonl")
    samples = {sample.sample_id: sample for sample in exp.read_single_box_samples(args.data_root, "test")}
    if args.case:
        cases = [tuple(item.split(":", 1)) for item in args.case]
    else:
        cases = att.choose_cases(predictions, args.max_cases)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)

    summary_rows: list[dict[str, Any]] = []
    for sample_id, condition in cases:
        sample = samples[sample_id]
        prompt = exp.build_prompt(sample.label_text, "qwen3vl")
        clean_record = predictions[sample_id]["clean_1"]
        perturbed_record = predictions[sample_id][condition]
        clean_image = image_for_condition(args.data_root, args.experiment_dir, clean_record)
        perturbed_image = image_for_condition(args.data_root, args.experiment_dir, perturbed_record)
        case_dir = args.output_dir / f"{sample_id}__{condition}"
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"Analyzing {sample_id} clean_1 and {condition}: {sample.label_text}")
        clean_stats = analyze_record(model, processor, clean_image, clean_record, prompt, args, case_dir)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        perturbed_stats = analyze_record(model, processor, perturbed_image, perturbed_record, prompt, args, case_dir)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        att.make_panel(
            case_dir / "clean_1_patch_gradcam.jpg",
            case_dir / f"{condition}_patch_gradcam.jpg",
            case_dir / "patch_gradcam_panel.jpg",
        )
        bbox_shift = exp.center_shift(clean_record["bbox_xyxy"], perturbed_record["bbox_xyxy"]) or 0.0
        gradcam_shift = math.hypot(
            float(clean_stats["gradcam_cx"]) - float(perturbed_stats["gradcam_cx"]),
            float(clean_stats["gradcam_cy"]) - float(perturbed_stats["gradcam_cy"]),
        )
        heatmap_metrics = att.heatmap_metrics(
            np.load(case_dir / "clean_1_patch_gradcam.npy"),
            np.load(case_dir / f"{condition}_patch_gradcam.npy"),
            args.top_heatmap_frac,
        )
        summary_rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "category_name": sample.category_name,
                "label_text": sample.label_text,
                "cam_source": clean_stats["cam_source"],
                "vision_layer": clean_stats["vision_layer"],
                "bbox_shift_px": bbox_shift,
                "gradcam_center_shift_px": gradcam_shift,
                "heatmap_pearson": heatmap_metrics["heatmap_pearson"],
                "heatmap_cosine": heatmap_metrics["heatmap_cosine"],
                "top_heatmap_iou": heatmap_metrics["top_attention_iou"],
                "clean_mass_in_pred_box": clean_stats["mass_in_pred_box"],
                "clean_mass_in_gt_box": clean_stats["mass_in_gt_box"],
                "perturbed_mass_in_pred_box": perturbed_stats["mass_in_pred_box"],
                "perturbed_mass_in_gt_box": perturbed_stats["mass_in_gt_box"],
                "clean_gradcam_cx": clean_stats["gradcam_cx"],
                "clean_gradcam_cy": clean_stats["gradcam_cy"],
                "perturbed_gradcam_cx": perturbed_stats["gradcam_cx"],
                "perturbed_gradcam_cy": perturbed_stats["gradcam_cy"],
                "clean_target_text": clean_stats["target_text"],
                "perturbed_target_text": perturbed_stats["target_text"],
                "clean_grid": clean_stats["grid"],
                "perturbed_grid": perturbed_stats["grid"],
                "panel_path": str(case_dir / "patch_gradcam_panel.jpg"),
            }
        )

    metrics_path = args.output_dir / "patch_gradcam_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# Qwen3-VL Patch Grad-CAM", ""]
    lines.append(
        f"Method: `{args.cam_source}` with attribution `{args.attribution_mode}`. "
        "Images are not occluded or additionally modified for attribution."
    )
    lines.append("")
    lines.append("| sample | perturbation | bbox shift px | Grad-CAM shift px | heatmap r | top10 IoU | clean mass@pred | pert mass@pred | clean mass@GT | pert mass@GT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['sample_id']} | {row['condition']} | {row['bbox_shift_px']:.2f} | "
            f"{row['gradcam_center_shift_px']:.2f} | {row['heatmap_pearson']:.4f} | "
            f"{row['top_heatmap_iou']:.4f} | {row['clean_mass_in_pred_box']:.4f} | "
            f"{row['perturbed_mass_in_pred_box']:.4f} | {row['clean_mass_in_gt_box']:.4f} | "
            f"{row['perturbed_mass_in_gt_box']:.4f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
