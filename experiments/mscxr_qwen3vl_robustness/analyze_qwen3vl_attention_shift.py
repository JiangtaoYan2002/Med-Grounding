#!/usr/bin/env python3
"""Diagnose whether Qwen3-VL image-token attention shifts under perturbations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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

import run_mscxr_qwen3vl_robustness as exp


DEFAULT_EXPERIMENT_DIR = Path("/home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_robustness_paper_perturbations")
DEFAULT_MODEL_PATH = Path("/data/yanjiangtao/yanjiangtao/checkpoints/Qwen3-VL-8B-Instruct")
DEFAULT_DATA_ROOT = Path("/home/yanjiangtao/Med-Grounding/data/MS-CXR")
DEFAULT_OUTPUT_DIR = Path("/home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_attention_shift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--top-attention-frac", type=float, default=0.10)
    parser.add_argument("--viz-percentile", type=float, default=99.5)
    parser.add_argument("--viz-threshold-percentile", type=float, default=80.0)
    parser.add_argument("--viz-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--attention-token-scope", choices=["all", "numeric"], default="numeric")
    parser.add_argument("--case", action="append", default=[], help="Explicit case as sample_id:condition.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            grouped[record["sample_id"]][record["condition"]] = record
    return grouped


def choose_cases(predictions: dict[str, dict[str, dict[str, Any]]], max_cases: int) -> list[tuple[str, str]]:
    preferred = [
        "gaussian_noise_sigma_20",
        "black_border_10px",
        "white_border_10px",
        "gaussian_blur_sigma_0_8",
        "contrast_plus_8",
        "brightness_plus_8",
    ]
    cases: list[tuple[str, str]] = []
    used_samples: set[str] = set()
    for condition in preferred:
        ranked: list[tuple[float, str]] = []
        for sample_id, records in predictions.items():
            clean = records.get("clean_1", {}).get("bbox_xyxy")
            perturbed = records.get(condition, {}).get("bbox_xyxy")
            if clean is None or perturbed is None:
                continue
            ranked.append((exp.center_shift(clean, perturbed) or 0.0, sample_id))
        ranked.sort(reverse=True)
        for _, sample_id in ranked:
            if sample_id not in used_samples:
                cases.append((sample_id, condition))
                used_samples.add(sample_id)
                break
        if len(cases) >= max_cases:
            break
    return cases


def set_text_attention_impl(model: Any, implementation: str) -> None:
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    for obj in [
        getattr(model.config, "text_config", None),
        getattr(language_model, "config", None),
    ]:
        if obj is not None:
            setattr(obj, "_attn_implementation", implementation)
            setattr(obj, "attn_implementation", implementation)


def prepare_inputs(processor: Any, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt")


def attention_grid_shape(inputs: dict[str, torch.Tensor], image_token_count: int) -> tuple[int, int]:
    _, grid_h, grid_w = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    for merge in (2, 1, 4):
        h = grid_h // merge
        w = grid_w // merge
        if h * w == image_token_count:
            return h, w
    raise ValueError(f"Cannot map {image_token_count} image tokens to grid {grid_h}x{grid_w}.")


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def canonical_bbox_text(record: dict[str, Any]) -> str:
    bbox = record.get("raw_bbox_xyxy")
    if not bbox:
        bbox = record.get("bbox_xyxy")
    if not bbox:
        raise ValueError(f"Record has no bbox: {record.get('sample_id')} {record.get('condition')}")
    values = [int(round(float(v))) for v in bbox]
    return json.dumps({"bbox": values}, separators=(",", ": "))


def target_query_indices(processor: Any, target_ids: torch.Tensor, scope: str) -> torch.Tensor:
    positions = torch.arange(target_ids.shape[1], device=target_ids.device)
    if scope == "all":
        return positions
    tokens = processor.tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    selected = [idx for idx, token in enumerate(tokens) if any(ch.isdigit() for ch in token)]
    if not selected:
        return positions
    return torch.tensor(selected, dtype=torch.long, device=target_ids.device)


def extract_attention_heatmap(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    target_text: str,
    max_new_tokens: int,
    last_layers: int,
    attention_token_scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs = prepare_inputs(processor, image, prompt)
    input_ids = inputs["input_ids"][0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = torch.where(input_ids == image_token_id)[0]
    grid_h, grid_w = attention_grid_shape(inputs, int(image_positions.numel()))

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    set_text_attention_impl(model, "sdpa")
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True, return_dict=True)

    target_ids = processor.tokenizer(
        target_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, :max_new_tokens].to(model.device)
    if target_ids.numel() == 0:
        raise ValueError("Target text produced no tokens.")
    query_indices = target_query_indices(processor, target_ids, attention_token_scope)
    target_attention_mask = torch.ones(
        (1, target_ids.shape[1]),
        dtype=inputs["attention_mask"].dtype,
        device=model.device,
    )
    attention_mask = torch.cat([inputs["attention_mask"], target_attention_mask], dim=1)
    text_positions = attention_mask.long().cumsum(-1) - 1
    text_positions = text_positions.masked_fill(attention_mask == 0, 0)[:, -target_ids.shape[1] :]
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = torch.zeros((target_ids.shape[0], 1), dtype=torch.long, device=model.device)
    position_ids = text_positions[None, ...] + rope_deltas.to(device=model.device, dtype=torch.long)

    set_text_attention_impl(model, "eager")
    with torch.inference_mode():
        decoded = model(
            input_ids=target_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefill.past_key_values,
            use_cache=False,
            return_dict=True,
            output_attentions=True,
        )

    accumulator = torch.zeros(image_positions.numel(), dtype=torch.float32)
    used_steps = 0
    selected_layers = (decoded.attentions or ())[-last_layers:]
    for layer_attention in selected_layers:
        if layer_attention.shape[-1] <= int(image_positions[-1]):
            continue
        accumulator += layer_attention[0, :, query_indices, :][:, :, image_positions].float().mean(dim=(0, 1)).cpu()
        used_steps += 1

    if used_steps > 0:
        accumulator /= used_steps
    heatmap = normalize_heatmap(accumulator.numpy().reshape(grid_h, grid_w))
    meta = {
        "target_text": target_text,
        "input_width": image.width,
        "input_height": image.height,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "image_token_count": int(image_positions.numel()),
        "used_attention_layers": used_steps,
        "target_token_count": int(target_ids.shape[1]),
        "attended_token_count": int(query_indices.numel()),
    }
    return heatmap, meta


def attention_center(heatmap: np.ndarray, width: int, height: int, condition: str) -> tuple[float, float]:
    mass = normalize_heatmap(heatmap)
    ys, xs = np.indices(mass.shape)
    cx = float(((xs + 0.5) / mass.shape[1] * width * mass).sum())
    cy = float(((ys + 0.5) / mass.shape[0] * height * mass).sum())
    if condition in {"black_border_10px", "white_border_10px"}:
        cx -= 10.0
        cy -= 10.0
    return cx, cy


def top_attention_center(
    heatmap: np.ndarray,
    width: int,
    height: int,
    condition: str,
    top_frac: float,
) -> tuple[float, float]:
    flat = heatmap.reshape(-1)
    k = max(1, int(round(flat.size * top_frac)))
    mask = np.zeros_like(flat, dtype=bool)
    mask[np.argpartition(flat, -k)[-k:]] = True
    focused = np.zeros_like(flat, dtype=np.float64)
    focused[mask] = flat[mask]
    focused = normalize_heatmap(focused.reshape(heatmap.shape))
    ys, xs = np.indices(focused.shape)
    cx = float(((xs + 0.5) / focused.shape[1] * width * focused).sum())
    cy = float(((ys + 0.5) / focused.shape[0] * height * focused).sum())
    if condition in {"black_border_10px", "white_border_10px"}:
        cx -= 10.0
        cy -= 10.0
    return cx, cy


def resize_heatmap_for_compare(heatmap: np.ndarray, size: int = 64) -> np.ndarray:
    resized = cv2.resize(heatmap.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    return normalize_heatmap(resized)


def heatmap_metrics(clean: np.ndarray, perturbed: np.ndarray, top_frac: float) -> dict[str, float]:
    a = resize_heatmap_for_compare(clean)
    b = resize_heatmap_for_compare(perturbed)
    av = a.reshape(-1)
    bv = b.reshape(-1)
    pearson = 0.0
    if float(av.std()) > 0 and float(bv.std()) > 0:
        pearson = float(np.corrcoef(av, bv)[0, 1])
    cosine = float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))
    k = max(1, int(round(av.size * top_frac)))
    a_top = np.zeros_like(av, dtype=bool)
    b_top = np.zeros_like(bv, dtype=bool)
    a_top[np.argpartition(av, -k)[-k:]] = True
    b_top[np.argpartition(bv, -k)[-k:]] = True
    top_iou = float(np.logical_and(a_top, b_top).sum() / max(1, np.logical_or(a_top, b_top).sum()))
    return {"heatmap_pearson": pearson, "heatmap_cosine": cosine, "top_attention_iou": top_iou}


def box_to_input(box: list[float] | tuple[float, float, float, float] | None, condition: str) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if condition in {"black_border_10px", "white_border_10px"}:
        return x1 + 10.0, y1 + 10.0, x2 + 10.0, y2 + 10.0
    return x1, y1, x2, y2


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float] | None, scale: float, color: str, width: int) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [v * scale for v in box]
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def overlay_attention(
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
    hm_grid = heatmap.astype(np.float32)
    if smooth_sigma > 0:
        hm_grid = cv2.GaussianBlur(hm_grid, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    hm = cv2.resize(hm_grid, display_size, interpolation=cv2.INTER_CUBIC)
    hm = np.log1p(np.maximum(hm, 0.0) * float(hm.size))
    lo = float(np.percentile(hm, 5.0))
    hi = float(np.percentile(hm, percentile))
    if hi <= lo:
        lo = float(hm.min())
        hi = float(hm.max())
    hm = np.clip((hm - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    threshold = float(np.percentile(hm, threshold_percentile))
    alpha = np.clip((hm - threshold) / max(1.0 - threshold, 1e-12), 0.0, 1.0) ** 0.7
    color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (0.68 * alpha)[..., None]
    blended = np.clip((1.0 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(blended)
    draw = ImageDraw.Draw(canvas)
    draw_box(draw, gt_box, scale, "lime", max(2, int(3 * scale)))
    draw_box(draw, pred_box, scale, "red", max(2, int(3 * scale)))
    draw.rectangle([0, 0, display_size[0], 30], fill=(0, 0, 0))
    draw.text((8, 7), title, fill="white", font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def make_panel(left: Path, right: Path, output_path: Path) -> None:
    left_img = Image.open(left).convert("RGB")
    right_img = Image.open(right).convert("RGB")
    h = max(left_img.height, right_img.height)
    panel = Image.new("RGB", (left_img.width + right_img.width, h), "white")
    panel.paste(left_img, (0, 0))
    panel.paste(right_img, (left_img.width, 0))
    panel.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists and is not empty. Use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(args.experiment_dir / "predictions.jsonl")
    samples = {sample.sample_id: sample for sample in exp.read_single_box_samples(args.data_root, "test")}
    if args.case:
        cases = []
        for item in args.case:
            sample_id, condition = item.split(":", 1)
            cases.append((sample_id, condition))
    else:
        cases = choose_cases(predictions, args.max_cases)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    for sample_id, condition in cases:
        sample = samples[sample_id]
        clean_record = predictions[sample_id]["clean_1"]
        perturbed_record = predictions[sample_id][condition]
        prompt = exp.build_prompt(sample.label_text, "qwen3vl")
        clean_image = Image.open(args.data_root / "images" / sample.rel_path).convert("RGB")
        perturbed_path = args.experiment_dir / "perturbed_images" / condition / sample.rel_path
        perturbed_image = Image.open(perturbed_path).convert("RGB")

        print(f"Analyzing {sample_id} {condition}: {sample.label_text}")
        clean_target = canonical_bbox_text(clean_record)
        perturbed_target = canonical_bbox_text(perturbed_record)
        clean_heatmap, clean_meta = extract_attention_heatmap(
            model,
            processor,
            clean_image,
            prompt,
            clean_target,
            args.max_new_tokens,
            args.last_layers,
            args.attention_token_scope,
        )
        pert_heatmap, pert_meta = extract_attention_heatmap(
            model,
            processor,
            perturbed_image,
            prompt,
            perturbed_target,
            args.max_new_tokens,
            args.last_layers,
            args.attention_token_scope,
        )

        clean_center = attention_center(clean_heatmap, clean_meta["input_width"], clean_meta["input_height"], "clean_1")
        pert_center = attention_center(pert_heatmap, pert_meta["input_width"], pert_meta["input_height"], condition)
        att_shift = math.hypot(clean_center[0] - pert_center[0], clean_center[1] - pert_center[1])
        clean_top_center = top_attention_center(
            clean_heatmap,
            clean_meta["input_width"],
            clean_meta["input_height"],
            "clean_1",
            args.top_attention_frac,
        )
        pert_top_center = top_attention_center(
            pert_heatmap,
            pert_meta["input_width"],
            pert_meta["input_height"],
            condition,
            args.top_attention_frac,
        )
        top_att_shift = math.hypot(clean_top_center[0] - pert_top_center[0], clean_top_center[1] - pert_top_center[1])
        metrics = heatmap_metrics(clean_heatmap, pert_heatmap, args.top_attention_frac)
        bbox_shift = exp.center_shift(clean_record["bbox_xyxy"], perturbed_record["bbox_xyxy"]) or 0.0
        bbox_iou = exp.bbox_iou(clean_record["bbox_xyxy"], perturbed_record["bbox_xyxy"])

        case_dir = args.output_dir / f"{sample_id}__{condition}"
        clean_overlay = case_dir / "clean_attention.jpg"
        pert_overlay = case_dir / f"{condition}_attention.jpg"
        panel_path = case_dir / "attention_panel.jpg"
        overlay_attention(
            clean_image,
            clean_heatmap,
            box_to_input(clean_record["bbox_xyxy"], "clean_1"),
            box_to_input(clean_record["gt_bbox_xyxy"], "clean_1"),
            f"{sample_id} clean_1",
            clean_overlay,
            args.viz_percentile,
            args.viz_threshold_percentile,
            args.viz_smooth_sigma,
        )
        overlay_attention(
            perturbed_image,
            pert_heatmap,
            box_to_input(perturbed_record["bbox_xyxy"], condition),
            box_to_input(perturbed_record["gt_bbox_xyxy"], condition),
            f"{sample_id} {condition}",
            pert_overlay,
            args.viz_percentile,
            args.viz_threshold_percentile,
            args.viz_smooth_sigma,
        )
        make_panel(clean_overlay, pert_overlay, panel_path)

        np.save(case_dir / "clean_attention.npy", clean_heatmap)
        np.save(case_dir / f"{condition}_attention.npy", pert_heatmap)
        rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "category_name": sample.category_name,
                "label_text": sample.label_text,
                "bbox_shift_px": bbox_shift,
                "bbox_iou_with_clean": bbox_iou,
                "attention_center_shift_px": att_shift,
                "top_attention_center_shift_px": top_att_shift,
                "clean_attention_cx": clean_center[0],
                "clean_attention_cy": clean_center[1],
                "perturbed_attention_cx": pert_center[0],
                "perturbed_attention_cy": pert_center[1],
                "clean_top_attention_cx": clean_top_center[0],
                "clean_top_attention_cy": clean_top_center[1],
                "perturbed_top_attention_cx": pert_top_center[0],
                "perturbed_top_attention_cy": pert_top_center[1],
                "heatmap_pearson": metrics["heatmap_pearson"],
                "heatmap_cosine": metrics["heatmap_cosine"],
                "top_attention_iou": metrics["top_attention_iou"],
                "clean_target_text": clean_meta["target_text"],
                "perturbed_target_text": pert_meta["target_text"],
                "clean_grid": f"{clean_meta['grid_h']}x{clean_meta['grid_w']}",
                "perturbed_grid": f"{pert_meta['grid_h']}x{pert_meta['grid_w']}",
                "clean_attended_tokens": clean_meta["attended_token_count"],
                "perturbed_attended_tokens": pert_meta["attended_token_count"],
                "panel_path": str(panel_path),
            }
        )

    metrics_path = args.output_dir / "attention_shift_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Qwen3-VL Attention Shift Diagnostics", ""]
    lines.append("| sample | perturbation | bbox shift px | attn shift px | top-attn shift px | heatmap r | top10 IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['condition']} | {row['bbox_shift_px']:.2f} | "
            f"{row['attention_center_shift_px']:.2f} | {row['top_attention_center_shift_px']:.2f} | "
            f"{row['heatmap_pearson']:.4f} | {row['top_attention_iou']:.4f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
