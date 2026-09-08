#!/usr/bin/env python3
"""Run MS-CXR perturbation robustness evaluation with a vision-language model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import statistics
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration


DEFAULT_DATA_ROOT = Path("/home/yanjiangtao/Med-Grounding/data/MS-CXR")
DEFAULT_MODEL_PATH = Path("/data/yanjiangtao/yanjiangtao/checkpoints/Qwen3-VL-8B-Instruct")
DEFAULT_OUTPUT_DIR = Path("/home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_robustness")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dicom_id: str
    category_name: str
    label_text: str
    rel_path: str
    split: str
    bbox_xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str
    value: int | float | None = None


PERTURBATIONS = [
    Perturbation("gaussian_noise_sigma_20", "gaussian_noise", 20.0),
    Perturbation("black_border_10px", "outer_border", 10),
    Perturbation("white_border_10px", "outer_border", 10),
    Perturbation("brightness_plus_8", "brightness", 1.08),
    Perturbation("contrast_plus_8", "contrast", 1.08),
    Perturbation("gaussian_blur_sigma_0_8", "gaussian_blur", 0.8),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-family", choices=["qwen3vl", "hulumed"], default="qwen3vl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--max-samples", type=int, default=5, help="-1 means all filtered samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--hulumed-max-visual-tokens",
        type=int,
        default=None,
        help="Override HuLu-Med's internal dynamic-resize token cap; the original image is still the input.",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["qwen_1000", "pixel"],
        default="qwen_1000",
        help="How to interpret model-emitted coordinates before mapping back to original image pixels.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only build the filtered sample table.")
    parser.add_argument("--save-perturbed-images", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=24)
    parser.add_argument("--resume", action="store_true", help="Append to an existing predictions.jsonl and skip completed sample-condition pairs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_single_box_samples(data_root: Path, split: str) -> list[Sample]:
    csv_path = data_root / "annotations" / "MS_CXR_Local_Alignment_v1.1.0.csv"
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["dicom_id"],
                row["path"],
                row["category_name"],
                row["label_text"],
                row["split"],
            )
            groups[key].append(row)

    samples: list[Sample] = []
    for idx, ((dicom_id, rel_path, category_name, label_text, row_split), rows) in enumerate(groups.items()):
        if len(rows) != 1:
            continue
        if split != "all" and row_split != split:
            continue
        row = rows[0]
        x = float(row["x"])
        y = float(row["y"])
        w = float(row["w"])
        h = float(row["h"])
        width = int(float(row["image_width"]))
        height = int(float(row["image_height"]))
        samples.append(
            Sample(
                sample_id=f"{row_split}_{idx:05d}",
                dicom_id=dicom_id,
                category_name=category_name,
                label_text=label_text,
                rel_path=rel_path,
                split=row_split,
                bbox_xyxy=(x, y, x + w, y + h),
                image_width=width,
                image_height=height,
            )
        )
    return samples


def write_samples_csv(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "dicom_id",
                "category_name",
                "label_text",
                "path",
                "split",
                "gt_x1",
                "gt_y1",
                "gt_x2",
                "gt_y2",
                "image_width",
                "image_height",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "dicom_id": sample.dicom_id,
                    "category_name": sample.category_name,
                    "label_text": sample.label_text,
                    "path": sample.rel_path,
                    "split": sample.split,
                    "gt_x1": sample.bbox_xyxy[0],
                    "gt_y1": sample.bbox_xyxy[1],
                    "gt_x2": sample.bbox_xyxy[2],
                    "gt_y2": sample.bbox_xyxy[3],
                    "image_width": sample.image_width,
                    "image_height": sample.image_height,
                }
            )


def build_prompt(label_text: str, model_family: str) -> str:
    coordinate_description = (
        "Use integer coordinates in the Qwen grounding coordinate system: "
        if model_family == "qwen3vl"
        else "Use integer coordinates normalized over the displayed input image: "
    )
    return (
        "You are localizing a radiological finding in a chest X-ray.\n"
        f'Finding query: "{label_text}"\n\n'
        "Return exactly one bounding box that encloses the queried finding.\n"
        f"{coordinate_description}"
        "x and y values range from 0 to 1000 over the input image.\n"
        "x increases from left to right and y increases from top to bottom.\n"
        'Output only valid JSON: {"bbox": [x1, y1, x2, y2]}'
    )


def apply_perturbation(image: Image.Image, perturbation: Perturbation, rng: np.random.Generator) -> Image.Image:
    image = image.convert("RGB")
    if perturbation.kind == "gaussian_noise":
        sigma = float(perturbation.value)
        arr = np.asarray(image).astype(np.float32)
        noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if perturbation.kind == "outer_border":
        border = int(perturbation.value)
        fill = 0 if perturbation.name.startswith("black") else 255
        canvas = Image.new("RGB", (image.width + 2 * border, image.height + 2 * border), (fill, fill, fill))
        canvas.paste(image, (border, border))
        return canvas

    if perturbation.kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(perturbation.value))

    if perturbation.kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(perturbation.value))

    if perturbation.kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(perturbation.value)))

    raise ValueError(f"Unsupported perturbation: {perturbation}")


def raw_to_original_bbox(
    bbox: tuple[float, float, float, float] | None,
    condition: str,
    image_width: int,
    image_height: int,
    input_width: int,
    input_height: int,
    coordinate_mode: str,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if coordinate_mode == "qwen_1000":
        x1 = x1 / 1000.0 * input_width
        x2 = x2 / 1000.0 * input_width
        y1 = y1 / 1000.0 * input_height
        y2 = y2 / 1000.0 * input_height
    elif coordinate_mode != "pixel":
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")
    if condition in {"black_border_10px", "white_border_10px"}:
        x1 -= 10
        x2 -= 10
        y1 -= 10
        y2 -= 10
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        min(max(x1, 0.0), float(image_width)),
        min(max(y1, 0.0), float(image_height)),
        min(max(x2, 0.0), float(image_width)),
        min(max(y2, 0.0), float(image_height)),
    )


def parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    try:
        obj = json.loads(text)
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            return tuple(float(v) for v in bbox)  # type: ignore[return-value]
    except Exception:
        pass

    json_match = re.search(r"\{.*?bbox.*?\}", text, flags=re.S | re.I)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            bbox = obj.get("bbox") if isinstance(obj, dict) else None
            if isinstance(bbox, list) and len(bbox) == 4:
                return tuple(float(v) for v in bbox)  # type: ignore[return-value]
        except Exception:
            pass

    bracket_matches = re.findall(r"\[([^\[\]]+)\]", text)
    for match in bracket_matches:
        nums = re.findall(r"-?\d+(?:\.\d+)?", match)
        if len(nums) == 4:
            return tuple(float(v) for v in nums)  # type: ignore[return-value]

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 4:
        return tuple(float(v) for v in nums[:4])  # type: ignore[return-value]
    return None


def bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_shift(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float | None:
    if a is None or b is None:
        return None
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype: str | torch.dtype
    if args.torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.torch_dtype]

    if args.model_family == "hulumed":
        processor = load_hulumed_processor(args.model_path, args.hulumed_max_visual_tokens)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        )
    model.eval()
    return model, processor


def load_hulumed_processor(model_path: Path, max_visual_tokens: int | None = None) -> Any:
    package_name = "local_hulumed_checkpoint"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_path)]
        sys.modules[package_name] = package

    processing_path = model_path / "processing_hulumed.py"
    module_name = f"{package_name}.processing_hulumed"
    spec = importlib.util.spec_from_file_location(module_name, processing_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HuLu-Med processor from {processing_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    image_processor = module.image_processing_hulumed.HulumedImageProcessor.from_pretrained(model_path)
    if max_visual_tokens is not None:
        image_processor.max_tokens = max_visual_tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return module.HulumedProcessor(image_processor=image_processor, tokenizer=tokenizer)


def run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    model_family: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if model_family == "hulumed":
        image_inputs = processor.process_images(
            [("image", image)],
            merge_size=1,
            return_tensors="pt",
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_system_prompt=True,
            add_generation_prompt=True,
        )
        text_inputs = processor.process_text(text, image_inputs, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if model_family == "hulumed" and inputs.get("pixel_values") is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if model_family == "hulumed":
        generation_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    if model_family == "hulumed":
        output_ids = generated
    else:
        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated[:, input_len:]
    decode_kwargs: dict[str, Any] = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    if model_family == "hulumed":
        decode_kwargs["use_think"] = False
    return processor.batch_decode(output_ids, **decode_kwargs)[0].strip()


def make_prediction_record(
    sample: Sample,
    condition: str,
    input_width: int,
    input_height: int,
    coordinate_mode: str,
    raw_text: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    raw_bbox = parse_bbox(raw_text)
    bbox = raw_to_original_bbox(
        raw_bbox,
        condition,
        sample.image_width,
        sample.image_height,
        input_width,
        input_height,
        coordinate_mode,
    )
    gt = sample.bbox_xyxy
    return {
        "sample_id": sample.sample_id,
        "dicom_id": sample.dicom_id,
        "path": sample.rel_path,
        "split": sample.split,
        "category_name": sample.category_name,
        "label_text": sample.label_text,
        "condition": condition,
        "raw_response": raw_text,
        "raw_bbox_xyxy": list(raw_bbox) if raw_bbox else None,
        "bbox_xyxy": list(bbox) if bbox else None,
        "coordinate_mode": coordinate_mode,
        "input_image_width": input_width,
        "input_image_height": input_height,
        "gt_bbox_xyxy": list(gt),
        "image_width": sample.image_width,
        "image_height": sample.image_height,
        "valid_bbox": bbox is not None and bbox_area(bbox) > 0,
        "iou_with_gt": bbox_iou(bbox, gt),
        "center_shift_to_gt_px": center_shift(bbox, gt),
        "elapsed_sec": elapsed_sec,
    }


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_predictions(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        by_condition[record["condition"]].append(record)
        by_sample_condition[(record["sample_id"], record["condition"])] = record

    summary_rows: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        ious = finite_values(rows, "iou_with_gt")
        shifts = finite_values(rows, "center_shift_to_gt_px")
        summary_rows.append(
            {
                "condition": condition,
                "n": len(rows),
                "valid_rate": sum(bool(r["valid_bbox"]) for r in rows) / len(rows) if rows else None,
                "mean_iou_with_gt": mean_or_none(ious),
                "median_iou_with_gt": median_or_none(ious),
                "iou_gt_at_0_1": sum(v >= 0.1 for v in ious) / len(rows) if rows else None,
                "iou_gt_at_0_3": sum(v >= 0.3 for v in ious) / len(rows) if rows else None,
                "iou_gt_at_0_5": sum(v >= 0.5 for v in ious) / len(rows) if rows else None,
                "mean_center_shift_to_gt_px": mean_or_none(shifts),
                "median_center_shift_to_gt_px": median_or_none(shifts),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for sample_id in sorted({r["sample_id"] for r in records}):
        clean_1 = by_sample_condition.get((sample_id, "clean_1"))
        clean_2 = by_sample_condition.get((sample_id, "clean_2"))
        if not clean_1 or not clean_2:
            continue
        clean_1_box = tuple(clean_1["bbox_xyxy"]) if clean_1["bbox_xyxy"] else None
        clean_2_box = tuple(clean_2["bbox_xyxy"]) if clean_2["bbox_xyxy"] else None
        repeat_iou = bbox_iou(clean_1_box, clean_2_box)
        repeat_shift = center_shift(clean_1_box, clean_2_box)
        paired_rows.append(
            {
                "condition": "clean_repeat",
                "sample_id": sample_id,
                "iou_with_clean_1": repeat_iou,
                "center_shift_from_clean_1_px": repeat_shift,
                "area_ratio_vs_clean_1": bbox_area(clean_2_box) / bbox_area(clean_1_box)
                if bbox_area(clean_1_box) > 0
                else None,
                "iou_drop_vs_clean_1_gt": None,
                "repeat_center_shift_px": repeat_shift,
            }
        )
        for perturbation in PERTURBATIONS:
            row = by_sample_condition.get((sample_id, perturbation.name))
            if not row:
                continue
            pbox = tuple(row["bbox_xyxy"]) if row["bbox_xyxy"] else None
            clean_gt_iou = clean_1.get("iou_with_gt")
            perturb_gt_iou = row.get("iou_with_gt")
            paired_rows.append(
                {
                    "condition": perturbation.name,
                    "sample_id": sample_id,
                    "iou_with_clean_1": bbox_iou(clean_1_box, pbox),
                    "center_shift_from_clean_1_px": center_shift(clean_1_box, pbox),
                    "area_ratio_vs_clean_1": bbox_area(pbox) / bbox_area(clean_1_box)
                    if bbox_area(clean_1_box) > 0
                    else None,
                    "iou_drop_vs_clean_1_gt": clean_gt_iou - perturb_gt_iou
                    if isinstance(clean_gt_iou, float) and isinstance(perturb_gt_iou, float)
                    else None,
                    "repeat_center_shift_px": repeat_shift,
                }
            )

    for condition in ["clean_repeat"] + [p.name for p in PERTURBATIONS]:
        rows = [r for r in paired_rows if r["condition"] == condition]
        if not rows:
            continue
        ious = finite_values(rows, "iou_with_clean_1")
        shifts = finite_values(rows, "center_shift_from_clean_1_px")
        drops = finite_values(rows, "iou_drop_vs_clean_1_gt")
        repeat_exceeded = [
            r
            for r in rows
            if isinstance(r.get("center_shift_from_clean_1_px"), (int, float))
            and isinstance(r.get("repeat_center_shift_px"), (int, float))
            and r["center_shift_from_clean_1_px"] > r["repeat_center_shift_px"]
        ]
        summary_rows.append(
            {
                "condition": f"{condition}__vs_clean",
                "n": len(rows),
                "valid_rate": None,
                "mean_iou_with_gt": None,
                "median_iou_with_gt": None,
                "iou_gt_at_0_1": None,
                "iou_gt_at_0_3": None,
                "iou_gt_at_0_5": None,
                "mean_center_shift_to_gt_px": None,
                "median_center_shift_to_gt_px": None,
                "mean_iou_with_clean_1": mean_or_none(ious),
                "median_iou_with_clean_1": median_or_none(ious),
                "mean_center_shift_from_clean_1_px": mean_or_none(shifts),
                "median_center_shift_from_clean_1_px": median_or_none(shifts),
                "shift_gt_50px_rate": sum(v > 50 for v in shifts) / len(rows) if rows else None,
                "shift_gt_100px_rate": sum(v > 100 for v in shifts) / len(rows) if rows else None,
                "shift_gt_200px_rate": sum(v > 200 for v in shifts) / len(rows) if rows else None,
                "mean_iou_drop_vs_clean_1_gt": mean_or_none(drops),
                "shift_exceeds_clean_repeat_rate": len(repeat_exceeded) / len(rows) if rows else None,
            }
        )

    lines = ["# MS-CXR Vision-Language Model Robustness Summary", ""]
    clean_rows = [r for r in summary_rows if r["condition"] in {"clean_1", "clean_2"}]
    if clean_rows:
        lines.append("## Original Image Localization")
        for row in clean_rows:
            lines.append(
                f"- {row['condition']}: valid={row['valid_rate']:.3f}, "
                f"mean IoU(GT)={fmt(row['mean_iou_with_gt'])}, "
                f"median IoU(GT)={fmt(row['median_iou_with_gt'])}, "
                f"IoU@0.5={fmt(row['iou_gt_at_0_5'])}"
            )
        lines.append("")

    lines.append("## Output Stability")
    for row in summary_rows:
        if not row["condition"].endswith("__vs_clean"):
            continue
        lines.append(
            f"- {row['condition']}: mean IoU(clean)={fmt(row.get('mean_iou_with_clean_1'))}, "
            f"median IoU(clean)={fmt(row.get('median_iou_with_clean_1'))}, "
            f"mean shift={fmt(row.get('mean_center_shift_from_clean_1_px'))} px, "
            f"median shift={fmt(row.get('median_center_shift_from_clean_1_px'))} px, "
            f"shift>100px={fmt(row.get('shift_gt_100px_rate'))}, "
            f"exceeds clean-repeat={fmt(row.get('shift_exceeds_clean_repeat_rate'))}"
        )
    lines.append("")
    return summary_rows, "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_dicts_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line {line_number}: {path}")
    return records


def draw_box(draw: ImageDraw.ImageDraw, box: list[float] | tuple[float, float, float, float], color: str, width: int) -> None:
    draw.rectangle([float(v) for v in box], outline=color, width=width)


def save_visualizations(records: list[dict[str, Any]], data_root: Path, output_dir: Path, max_visualizations: int) -> None:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_sample[record["sample_id"]][record["condition"]] = record

    candidates: list[tuple[float, str, str]] = []
    for sample_id, rows in by_sample.items():
        clean = rows.get("clean_1")
        if not clean or not clean.get("bbox_xyxy"):
            continue
        clean_box = tuple(clean["bbox_xyxy"])
        for perturbation in PERTURBATIONS:
            row = rows.get(perturbation.name)
            if not row or not row.get("bbox_xyxy"):
                continue
            shift = center_shift(clean_box, tuple(row["bbox_xyxy"]))
            if shift is not None:
                candidates.append((shift, sample_id, perturbation.name))
    candidates.sort(reverse=True)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, sample_id, condition) in enumerate(candidates[:max_visualizations], start=1):
        rows = by_sample[sample_id]
        base = rows["clean_1"]
        image = Image.open(data_root / "images" / base["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_box(draw, base["gt_bbox_xyxy"], "lime", 6)
        if base.get("bbox_xyxy"):
            draw_box(draw, base["bbox_xyxy"], "cyan", 5)
        pert = rows[condition]
        if pert.get("bbox_xyxy"):
            draw_box(draw, pert["bbox_xyxy"], "red", 5)
        text = f"{rank}. {condition} | green=GT cyan=clean red=perturbed"
        draw.rectangle([0, 0, min(image.width, 24 * len(text)), 34], fill="black")
        draw.text((8, 8), text, fill="white")
        image.save(vis_dir / f"{rank:03d}_{sample_id}_{condition}.jpg", quality=95)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_family": args.model_family,
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "coordinate_mode": args.coordinate_mode,
        "hulumed_max_visual_tokens": args.hulumed_max_visual_tokens,
        "perturbations": [perturbation.name for perturbation in PERTURBATIONS],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    samples = read_single_box_samples(args.data_root, args.split)
    if args.shuffle:
        random.shuffle(samples)
    if args.max_samples >= 0:
        samples = samples[: args.max_samples]
    write_samples_csv(samples, args.output_dir / "single_box_samples.csv")
    print(f"Filtered single-box samples: {len(samples)}")
    print(f"Sample table: {args.output_dir / 'single_box_samples.csv'}")

    if args.dry_run:
        return

    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{predictions_path} exists. Use --overwrite to replace it.")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    model, processor = load_model(args)
    records: list[dict[str, Any]] = read_existing_predictions(predictions_path) if args.resume else []
    completed = {(r.get("sample_id"), r.get("condition")) for r in records}
    if args.resume:
        print(f"Loaded existing prediction records: {len(records)}")
    prompt_cache: dict[str, str] = {}
    perturbed_dir = args.output_dir / "perturbed_images"
    if args.save_perturbed_images:
        perturbed_dir.mkdir(parents=True, exist_ok=True)

    file_mode = "a" if args.resume else "w"
    with predictions_path.open(file_mode) as pred_f:
        for index, sample in enumerate(samples, start=1):
            image_path = args.data_root / "images" / sample.rel_path
            image = Image.open(image_path).convert("RGB")
            prompt = prompt_cache.setdefault(sample.label_text, build_prompt(sample.label_text, args.model_family))
            conditions: list[tuple[str, Image.Image]] = [("clean_1", image), ("clean_2", image)]
            for perturbation in PERTURBATIONS:
                pimage = apply_perturbation(image, perturbation, np_rng)
                conditions.append((perturbation.name, pimage))
                if args.save_perturbed_images:
                    out = perturbed_dir / perturbation.name / sample.rel_path
                    out.parent.mkdir(parents=True, exist_ok=True)
                    pimage.save(out, quality=95)

            print(f"[{index}/{len(samples)}] {sample.sample_id} {sample.category_name}: {sample.label_text[:80]}")
            for condition, condition_image in conditions:
                if (sample.sample_id, condition) in completed:
                    print(f"  - {condition}: skipped existing")
                    continue
                start = time.time()
                try:
                    raw_text = run_inference(
                        model=model,
                        processor=processor,
                        image=condition_image,
                        prompt=prompt,
                        model_family=args.model_family,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    raw_text = f"ERROR: {type(exc).__name__}: {exc}"
                elapsed = time.time() - start
                record = make_prediction_record(
                    sample=sample,
                    condition=condition,
                    input_width=condition_image.width,
                    input_height=condition_image.height,
                    coordinate_mode=args.coordinate_mode,
                    raw_text=raw_text,
                    elapsed_sec=elapsed,
                )
                records.append(record)
                pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pred_f.flush()
                print(
                    f"  - {condition}: valid={record['valid_bbox']} "
                    f"IoU(GT)={fmt(record['iou_with_gt'])} bbox={record['bbox_xyxy']}"
                )

    write_dicts_csv(records, args.output_dir / "predictions_long.csv")
    summary_rows, summary_md = summarize_predictions(records)
    write_dicts_csv(summary_rows, args.output_dir / "summary_by_condition.csv")
    (args.output_dir / "summary.md").write_text(summary_md)
    save_visualizations(records, args.data_root, args.output_dir, args.max_visualizations)
    print(summary_md)
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
