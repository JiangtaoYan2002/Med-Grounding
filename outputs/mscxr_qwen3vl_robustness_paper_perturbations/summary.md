# MS-CXR Qwen3-VL Robustness Summary

## Original Image Localization
- clean_1: valid=1.000, mean IoU(GT)=0.2555, median IoU(GT)=0.2252, IoU@0.5=0.1812
- clean_2: valid=1.000, mean IoU(GT)=0.2555, median IoU(GT)=0.2252, IoU@0.5=0.1812

## Output Stability
- clean_repeat__vs_clean: mean IoU(clean)=1.0000, median IoU(clean)=1.0000, mean shift=0.0000 px, median shift=0.0000 px, shift>100px=0.0000, exceeds clean-repeat=0.0000
- gaussian_noise_sigma_20__vs_clean: mean IoU(clean)=0.4939, median IoU(clean)=0.5715, mean shift=329.8772 px, median shift=144.3717 px, shift>100px=0.6087, exceeds clean-repeat=0.9928
- black_border_10px__vs_clean: mean IoU(clean)=0.7181, median IoU(clean)=0.8442, mean shift=153.6922 px, median shift=50.7435 px, shift>100px=0.2754, exceeds clean-repeat=1.0000
- white_border_10px__vs_clean: mean IoU(clean)=0.7022, median IoU(clean)=0.8215, mean shift=157.0083 px, median shift=53.9970 px, shift>100px=0.3043, exceeds clean-repeat=1.0000
- brightness_plus_8__vs_clean: mean IoU(clean)=0.8536, median IoU(clean)=0.9500, mean shift=74.7413 px, median shift=13.4450 px, shift>100px=0.1522, exceeds clean-repeat=0.7464
- contrast_plus_8__vs_clean: mean IoU(clean)=0.8212, median IoU(clean)=0.9527, mean shift=106.6092 px, median shift=18.4727 px, shift>100px=0.2029, exceeds clean-repeat=0.7391
- gaussian_blur_sigma_0_8__vs_clean: mean IoU(clean)=0.7728, median IoU(clean)=0.8993, mean shift=130.5113 px, median shift=28.6978 px, shift>100px=0.2754, exceeds clean-repeat=0.8406
