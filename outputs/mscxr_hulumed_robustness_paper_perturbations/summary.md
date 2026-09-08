# MS-CXR Vision-Language Model Robustness Summary

## Original Image Localization
- clean_1: valid=1.000, mean IoU(GT)=0.0446, median IoU(GT)=0.0000, IoU@0.5=0.0072
- clean_2: valid=1.000, mean IoU(GT)=0.0446, median IoU(GT)=0.0000, IoU@0.5=0.0072

## Output Stability
- clean_repeat__vs_clean: mean IoU(clean)=1.0000, median IoU(clean)=1.0000, mean shift=0.0000 px, median shift=0.0000 px, shift>100px=0.0000, exceeds clean-repeat=0.0000
- gaussian_noise_sigma_20__vs_clean: mean IoU(clean)=0.6257, median IoU(clean)=0.6800, mean shift=185.5220 px, median shift=101.0750 px, shift>100px=0.5072, exceeds clean-repeat=0.5290
- black_border_10px__vs_clean: mean IoU(clean)=0.6422, median IoU(clean)=0.8517, mean shift=161.7699 px, median shift=8.8081 px, shift>100px=0.4420, exceeds clean-repeat=1.0000
- white_border_10px__vs_clean: mean IoU(clean)=0.6042, median IoU(clean)=0.6489, mean shift=184.8465 px, median shift=131.3371 px, shift>100px=0.5652, exceeds clean-repeat=1.0000
- brightness_plus_8__vs_clean: mean IoU(clean)=0.7440, median IoU(clean)=1.0000, mean shift=132.3402 px, median shift=0.0000 px, shift>100px=0.3478, exceeds clean-repeat=0.3623
- contrast_plus_8__vs_clean: mean IoU(clean)=0.7265, median IoU(clean)=1.0000, mean shift=113.8765 px, median shift=0.0000 px, shift>100px=0.3913, exceeds clean-repeat=0.4203
- gaussian_blur_sigma_0_8__vs_clean: mean IoU(clean)=0.7351, median IoU(clean)=1.0000, mean shift=133.5831 px, median shift=0.0000 px, shift>100px=0.3841, exceeds clean-repeat=0.3913
