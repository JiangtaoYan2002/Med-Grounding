# Qwen3-VL Patch Grad-CAM

Method: `visual_token` with attribution `grad_x_activation_abs`. Images are not occluded or additionally modified for attribution.

| sample | perturbation | bbox shift px | Grad-CAM shift px | heatmap r | top10 IoU | clean mass@pred | pert mass@pred | clean mass@GT | pert mass@GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_01113 | gaussian_noise_sigma_20 | 1311.49 | 230.72 | 0.1090 | 0.0705 | 0.2432 | 0.2190 | 0.0519 | 0.0356 |
| test_00218 | black_border_10px | 1610.32 | 298.88 | 0.0694 | 0.1156 | 0.0468 | 0.0645 | 0.0205 | 0.1024 |
| test_01023 | white_border_10px | 1162.62 | 288.88 | 0.0802 | 0.1310 | 0.1918 | 0.1441 | 0.0447 | 0.0180 |
| test_00510 | gaussian_blur_sigma_0_8 | 1336.17 | 208.99 | 0.2561 | 0.1816 | 0.3497 | 0.1938 | 0.1216 | 0.1963 |
