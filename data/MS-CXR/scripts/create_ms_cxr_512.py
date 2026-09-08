#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "annotations"
IMAGES_DIR = ROOT / "images"
OUT_ANNOTATIONS_DIR = ROOT / "annotations_512"
OUT_IMAGES_DIR = ROOT / "images_512"
TARGET_SIZE = 512

JSON_NAME = "MS_CXR_Local_Alignment_v1.1.0.json"
CSV_NAME = "MS_CXR_Local_Alignment_v1.1.0.csv"
PATHS_NAME = "ms_cxr_image_paths.txt"


def letterbox_params(width: int, height: int) -> dict[str, float | int | bool]:
    scale = TARGET_SIZE / max(width, height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_left = (TARGET_SIZE - resized_width) // 2
    pad_top = (TARGET_SIZE - resized_height) // 2
    return {
        "scale": scale,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "aspect_ratio_changed": not math.isclose(width / height, 1.0, rel_tol=0, abs_tol=1e-12),
    }


def transform_bbox(bbox: list[float], params: dict[str, float | int | bool]) -> list[float]:
    x, y, w, h = bbox
    scale = float(params["scale"])
    pad_left = int(params["pad_left"])
    pad_top = int(params["pad_top"])

    x1 = x * scale + pad_left
    y1 = y * scale + pad_top
    x2 = (x + w) * scale + pad_left
    y2 = (y + h) * scale + pad_top

    x1 = max(0.0, min(float(TARGET_SIZE), x1))
    y1 = max(0.0, min(float(TARGET_SIZE), y1))
    x2 = max(0.0, min(float(TARGET_SIZE), x2))
    y2 = max(0.0, min(float(TARGET_SIZE), y2))

    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_to_segmentation(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def rounded(values: list[float], ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def resize_images(image_paths: list[str]) -> dict[str, dict[str, float | int | bool]]:
    transforms: dict[str, dict[str, float | int | bool]] = {}
    for idx, rel_path in enumerate(image_paths, 1):
        src = IMAGES_DIR / rel_path
        dst = OUT_IMAGES_DIR / rel_path
        if not src.exists():
            raise FileNotFoundError(src)

        with Image.open(src) as img:
            img = img.convert("RGB")
            width, height = img.size
            params = letterbox_params(width, height)
            resized_width = int(params["resized_width"])
            resized_height = int(params["resized_height"])

            resized = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
            canvas.paste(resized, (int(params["pad_left"]), int(params["pad_top"])))

            dst.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst, format="JPEG", quality=95, optimize=True)

            transforms[rel_path] = {
                **params,
                "original_width": width,
                "original_height": height,
                "target_width": TARGET_SIZE,
                "target_height": TARGET_SIZE,
            }

        if idx % 250 == 0:
            print(f"processed_images={idx}")

    return transforms


def update_json(transforms: dict[str, dict[str, float | int | bool]]) -> None:
    with (ANNOTATIONS_DIR / JSON_NAME).open() as f:
        data = json.load(f)

    out = deepcopy(data)
    image_id_to_path: dict[int, str] = {}
    for image in out["images"]:
        rel_path = image["path"]
        params = transforms[rel_path]
        image_id_to_path[image["id"]] = rel_path
        image["original_width"] = params["original_width"]
        image["original_height"] = params["original_height"]
        image["width"] = TARGET_SIZE
        image["height"] = TARGET_SIZE
        image["resized_width"] = params["resized_width"]
        image["resized_height"] = params["resized_height"]
        image["scale"] = round(float(params["scale"]), 8)
        image["pad_left"] = params["pad_left"]
        image["pad_top"] = params["pad_top"]
        image["aspect_ratio_changed"] = params["aspect_ratio_changed"]

    for ann in out["annotations"]:
        params = transforms[image_id_to_path[ann["image_id"]]]
        bbox = rounded(transform_bbox(ann["bbox"], params))
        ann["original_bbox"] = ann["bbox"]
        ann["bbox"] = bbox
        ann["area"] = round(bbox[2] * bbox[3], 4)
        ann["segmentation"] = [rounded(bbox_to_segmentation(bbox)[0])]
        ann["width"] = TARGET_SIZE
        ann["height"] = TARGET_SIZE
        ann["scale"] = round(float(params["scale"]), 8)
        ann["pad_left"] = params["pad_left"]
        ann["pad_top"] = params["pad_top"]
        ann["aspect_ratio_changed"] = params["aspect_ratio_changed"]

    OUT_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_ANNOTATIONS_DIR / JSON_NAME).open("w") as f:
        json.dump(out, f, indent=2)


def update_csv(transforms: dict[str, dict[str, float | int | bool]]) -> None:
    with (ANNOTATIONS_DIR / CSV_NAME).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_fieldnames = reader.fieldnames or []

    extra_fields = [
        "original_x",
        "original_y",
        "original_w",
        "original_h",
        "original_image_width",
        "original_image_height",
        "resized_width",
        "resized_height",
        "scale",
        "pad_left",
        "pad_top",
        "aspect_ratio_changed",
    ]
    fieldnames = input_fieldnames + [f for f in extra_fields if f not in input_fieldnames]

    for row in rows:
        params = transforms[row["path"]]
        original_bbox = [
            float(row["x"]),
            float(row["y"]),
            float(row["w"]),
            float(row["h"]),
        ]
        bbox = transform_bbox(original_bbox, params)

        row["original_x"], row["original_y"], row["original_w"], row["original_h"] = [
            str(int(v)) if float(v).is_integer() else f"{v:.4f}" for v in original_bbox
        ]
        row["original_image_width"] = str(params["original_width"])
        row["original_image_height"] = str(params["original_height"])
        row["x"], row["y"], row["w"], row["h"] = [f"{v:.4f}" for v in bbox]
        row["image_width"] = str(TARGET_SIZE)
        row["image_height"] = str(TARGET_SIZE)
        row["resized_width"] = str(params["resized_width"])
        row["resized_height"] = str(params["resized_height"])
        row["scale"] = f"{float(params['scale']):.8f}"
        row["pad_left"] = str(params["pad_left"])
        row["pad_top"] = str(params["pad_top"])
        row["aspect_ratio_changed"] = str(params["aspect_ratio_changed"]).lower()

    OUT_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_ANNOTATIONS_DIR / CSV_NAME).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_paths_file() -> None:
    OUT_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    source = ANNOTATIONS_DIR / PATHS_NAME
    if source.exists():
        content = source.read_text()
    else:
        with (ANNOTATIONS_DIR / JSON_NAME).open() as f:
            data = json.load(f)
        paths = sorted({image["path"] for image in data["images"]})
        content = "\n".join(paths) + "\n"
    (OUT_ANNOTATIONS_DIR / PATHS_NAME).write_text(content)


def main() -> None:
    paths_file = ANNOTATIONS_DIR / PATHS_NAME
    if paths_file.exists():
        with paths_file.open() as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        with (ANNOTATIONS_DIR / JSON_NAME).open() as f:
            data = json.load(f)
        image_paths = sorted({image["path"] for image in data["images"]})

    transforms = resize_images(image_paths)
    update_json(transforms)
    update_csv(transforms)
    copy_paths_file()

    print(f"output_images={OUT_IMAGES_DIR}")
    print(f"output_annotations={OUT_ANNOTATIONS_DIR}")
    print(f"image_count={len(image_paths)}")


if __name__ == "__main__":
    main()
