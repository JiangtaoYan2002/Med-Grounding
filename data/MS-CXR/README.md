# MS-CXR Layout

This directory contains the extracted and normalized MS-CXR package.

## Directory Structure

```text
MS-CXR/
  annotations/
    MS_CXR_Local_Alignment_v1.1.0.csv
    MS_CXR_Local_Alignment_v1.1.0.json
    ms_cxr_image_paths.txt
  images/
    files/pXX/pXXXXXXXX/sXXXXXXXX/*.jpg
  annotations_512/
    MS_CXR_Local_Alignment_v1.1.0.csv
    MS_CXR_Local_Alignment_v1.1.0.json
    ms_cxr_image_paths.txt
  images_512/
    files/pXX/pXXXXXXXX/sXXXXXXXX/*.jpg
  scripts/
    convert_coco_json_to_csv.py
    create_ms_cxr_512.py
    receive_ms_cxr_images.py
  metadata/
    LICENSE.txt
    SHA256SUMS.txt
  logs/
    image_receive_log.jsonl
    missing_after_download.*
    missing_image_paths.json
```

## Path Convention

The annotation files store image paths relative to the MIMIC-CXR-JPG root, for example:

```text
files/p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg
```

To load the corresponding image from this normalized layout, join it with:

```text
MS-CXR/images/
```

## Bounding Box Format

Bounding boxes use COCO-style `xywh` format:

```text
[x, y, width, height]
```

where `x` and `y` are the top-left pixel coordinates.

## 512 Letterbox Version

`images_512/` contains 512 x 512 letterboxed images. Each original image is scaled so
that its longest side becomes 512 pixels, then centered on a black 512 x 512 canvas.

`annotations_512/` contains matching transformed annotations:

- `x`, `y`, `w`, `h`: transformed bounding box coordinates on the 512 x 512 canvas.
- `image_width`, `image_height`: both set to 512.
- `original_x`, `original_y`, `original_w`, `original_h`: original bounding box.
- `original_image_width`, `original_image_height`: original image size.
- `resized_width`, `resized_height`: content size before padding.
- `scale`: resize scale applied to the original image.
- `pad_left`, `pad_top`: black padding offset before the resized content.
- `aspect_ratio_changed`: whether the final 512 x 512 canvas aspect ratio differs from
  the original image aspect ratio. The image content itself is not stretched.

The 512 version can be regenerated with:

```bash
python3 scripts/create_ms_cxr_512.py
```

## Verified Counts

- JPG images: 1047
- JSON image records: 1120
- Unique image paths: 1047
- JSON annotations: 1448
- CSV rows: 1448
- Splits: train 1020, val 212, test 216
