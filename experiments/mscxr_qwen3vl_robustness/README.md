# MS-CXR Qwen3-VL Robustness Experiment

This experiment tests whether small image perturbations change Qwen3-VL localization outputs on MS-CXR.

## Data

The script uses the original MS-CXR annotations and images:

```text
data/MS-CXR/annotations/MS_CXR_Local_Alignment_v1.1.0.csv
data/MS-CXR/images/
```

It does not use the 512 letterboxed data. Samples are filtered to phrase-image cases where one query maps to exactly one bounding box.

## Conditions

Each sample is inferred eight times:

```text
clean_1
clean_2
gaussian_noise_sigma_20
black_border_10px
white_border_10px
brightness_plus_8
contrast_plus_8
gaussian_blur_sigma_0_8
```

The two clean runs measure repeat-input variability. The border perturbations add a 10px frame outside the original image instead of covering anatomy. The perturbations are chosen for paper-friendly display: they are visible to humans but should preserve the clinical judgment of the finding location.

By default, model coordinates are interpreted as Qwen grounding coordinates from 0 to 1000 over the input image and then mapped back to original-image pixels. This matches the coordinate style commonly emitted by Qwen VL models. Use `--coordinate-mode pixel` only if the model is known to emit true pixel coordinates.

## Run

Use the YJT conda environment:

```bash
/home/yanjiangtao/miniconda3/bin/conda run -n YJT python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/run_mscxr_qwen3vl_robustness.py \
  --dry-run
```

Smoke test:

```bash
/home/yanjiangtao/miniconda3/bin/conda run -n YJT python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/run_mscxr_qwen3vl_robustness.py \
  --split test \
  --max-samples 3 \
  --overwrite
```

Full test split:

```bash
/home/yanjiangtao/miniconda3/bin/conda run -n YJT python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/run_mscxr_qwen3vl_robustness.py \
  --split test \
  --max-samples -1 \
  --output-dir /home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_robustness_paper_perturbations \
  --save-perturbed-images \
  --overwrite
```

HuLu-Med-7B smoke test:

```bash
/home/yanjiangtao/miniconda3/envs/YJT/bin/python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/run_mscxr_qwen3vl_robustness.py \
  --model-family hulumed \
  --model-path /data/yanjiangtao/yanjiangtao/checkpoints/Hulu-Med-7B \
  --split test \
  --max-samples 3 \
  --output-dir /home/yanjiangtao/Med-Grounding/outputs/mscxr_hulumed_robustness_smoke \
  --overwrite
```

HuLu-Med uses the same filtered samples, perturbations, normalized 0-1000 coordinates, and metrics as the Qwen3-VL run. Its prompt describes the coordinate system without model-specific wording because HuLu-Med does not document a native grounding output format.

HuLu-Med-7B full test split:

```bash
/home/yanjiangtao/miniconda3/envs/YJT/bin/python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/run_mscxr_qwen3vl_robustness.py \
  --model-family hulumed \
  --model-path /data/yanjiangtao/yanjiangtao/checkpoints/Hulu-Med-7B \
  --hulumed-max-visual-tokens 4096 \
  --split test \
  --max-samples -1 \
  --max-new-tokens 64 \
  --output-dir /home/yanjiangtao/Med-Grounding/outputs/mscxr_hulumed_robustness_paper_perturbations \
  --save-perturbed-images \
  --overwrite
```

The 4096-token cap controls HuLu-Med's aspect-ratio-preserving internal dynamic resize. The experiment still reads the original MS-CXR files and does not use the separately prepared 512 images. Exact settings are written to `run_config.json` in the output directory.

## Outputs

Default output directory:

```text
outputs/mscxr_qwen3vl_robustness/
```

Important files:

```text
single_box_samples.csv
predictions.jsonl
predictions_long.csv
summary_by_condition.csv
summary.md
visualizations/
```

`summary.md` reports original-image localization ability via IoU against GT, clean-repeat stability, and perturbation-induced shifts relative to `clean_1`.

## Attention Diagnostics

To inspect whether Qwen3-VL attends to different image-token regions after perturbation, run:

```bash
/home/yanjiangtao/miniconda3/envs/YJT/bin/python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/analyze_qwen3vl_attention_shift.py \
  --max-cases 4 \
  --max-new-tokens 32 \
  --output-dir /home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_attention_shift \
  --overwrite
```

The script uses teacher forcing on the recorded `predictions.jsonl` bbox text, so the attention maps correspond to the actual clean and perturbed boxes from the robustness run. It first runs the visual/text prefill with SDPA to avoid full-image attention OOM, then switches the language decoder to eager attention only for the short bbox-token continuation.

Outputs include:

```text
attention_shift_metrics.csv
summary.md
<sample_id>__<condition>/attention_panel.jpg
<sample_id>__<condition>/*_attention.npy
```

The heatmap is a diagnostic, not a causal proof. Low heatmap correlation or low top-attention IoU indicates that the model used different visual-token evidence under perturbation. A large bbox shift can occur with either a large attention-center shift or a substantial change in attention distribution shape.

## Visual-Token Grad-CAM

For a less interventionist explanation of where Qwen3-VL uses visual evidence for a recorded localization output, run:

```bash
/home/yanjiangtao/miniconda3/envs/YJT/bin/python \
  /home/yanjiangtao/Med-Grounding/experiments/mscxr_qwen3vl_robustness/analyze_qwen3vl_patch_gradcam.py \
  --max-cases 4 \
  --cam-source visual_token \
  --attribution-mode grad_x_activation_abs \
  --output-dir /home/yanjiangtao/Med-Grounding/outputs/mscxr_qwen3vl_visual_token_gradcam \
  --overwrite
```

This analysis does not occlude or perturb the image beyond the already-defined clean/perturbed condition. It teacher-forces the bbox text already generated in `predictions.jsonl`, then backpropagates the bbox numeric-token log probability to Qwen3-VL's merged visual tokens, which are the visual embeddings consumed by the language model. This is closer to patch-level Grad-CAM for the localization decision than the decoder attention diagnostic above.

Outputs include:

```text
patch_gradcam_metrics.csv
summary.md
<sample_id>__<condition>/patch_gradcam_panel.jpg
<sample_id>__<condition>/*_patch_gradcam.npy
```

The main quantitative columns are:

```text
gradcam_center_shift_px
heatmap_pearson
top_heatmap_iou
clean_mass_in_pred_box / perturbed_mass_in_pred_box
clean_mass_in_gt_box / perturbed_mass_in_gt_box
```

Use this result carefully: if the visual-token Grad-CAM is diffuse or outside the radiology target, the supported conclusion is that the model's localization output is visually fragile, not necessarily that a clean medical attention focus moved from the correct lesion to a new lesion.
