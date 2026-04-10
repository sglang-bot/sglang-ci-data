# Diffusion Cross-Framework Performance Dashboard

*Generated: Apr 10 | Commit: `5638d40`*

## Cross-Framework Performance Comparison

| Model | Risk | sglang (s) |
|-------|------|---------|
| FLUX.1-dev | ✅ | **6.68** |
| FLUX.2-dev | ✅ | **22.78** |
| Qwen-Image-2512 | ✅ | **13.16** |
| Qwen-Image-Edit-2511 | ✅ | **23.76** |
| Z-Image-Turbo | ✅ | **0.88** |
| Wan2.2-T2V-A14B-Diffusers | ✅ | **209.41** |
| Wan2.2-TI2V-5B-Diffusers | ✅ | **62.16** |
| LTX-2 | ✅ | **14.21** |
| Wan2.2-I2V-A14B-Diffusers | ✅ | **204.58** |

### Latency Trend: flux1_dev_t2i_1024

![Latency Trend flux1_dev_t2i_1024](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_flux1_dev_t2i_1024.png)


### Latency Trend: flux2_dev_t2i_1024

![Latency Trend flux2_dev_t2i_1024](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_flux2_dev_t2i_1024.png)


### Latency Trend: qwen_image_2512_t2i_1024

![Latency Trend qwen_image_2512_t2i_1024](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_qwen_image_2512_t2i_1024.png)


### Latency Trend: qwen_image_edit_2511

![Latency Trend qwen_image_edit_2511](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_qwen_image_edit_2511.png)


### Latency Trend: zimage_turbo_t2i_1024

![Latency Trend zimage_turbo_t2i_1024](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_zimage_turbo_t2i_1024.png)


### Latency Trend: wan22_t2v_a14b_720p

![Latency Trend wan22_t2v_a14b_720p](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_wan22_t2v_a14b_720p.png)


### Latency Trend: wan22_ti2v_5b_720p

![Latency Trend wan22_ti2v_5b_720p](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_wan22_ti2v_5b_720p.png)


### Latency Trend: ltx2_twostage_t2v

![Latency Trend ltx2_twostage_t2v](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_ltx2_twostage_t2v.png)


### Latency Trend: wan22_i2v_a14b_720p

![Latency Trend wan22_i2v_a14b_720p](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_wan22_i2v_a14b_720p.png)


## SGLang Performance Trend (Last 15 Runs)

| Date | Commit | flux1_dev_t2i_1024 (s) | flux2_dev_t2i_1024 (s) | qwen_image_2512_t2i_1024 (s) | qwen_image_edit_2511 (s) | zimage_turbo_t2i_1024 (s) | wan22_t2v_a14b_720p (s) | wan22_ti2v_5b_720p (s) | ltx2_twostage_t2v (s) | wan22_i2v_a14b_720p (s) | Trend |
|------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
| Apr 10 | `5638d40` | 6.68 | 22.78 | 13.16 | 23.76 | 0.88 | 209.41 | 62.16 | 14.21 | 204.58 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow: |
| Apr 09 | `2c4e113` | 6.71 | 22.93 | 13.20 | 23.80 | 0.89 | 210.35 | 62.16 | 14.71 | 206.54 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow: |
| Apr 08 | `dd73e9a` | 6.83 | 22.96 | 13.19 | 23.84 | 0.89 | 210.25 | 62.20 | 15.24 | 206.65 | :arrow_up:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow: |
| Apr 07 | `5cc246e` | 6.69 | 22.92 | 13.19 | 23.83 | 0.89 | 210.64 | 62.20 | 15.11 | 207.62 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow: |
| Apr 06 | `93109cc` | 6.70 | 22.86 | 13.61 | 23.92 | 0.89 | 210.73 | 62.15 | 14.13 | 207.67 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow: |
| Apr 05 | `70658bf` | 6.68 | 22.80 | 14.12 | 23.80 | 0.88 | 209.31 | 62.15 | 14.06 | 204.55 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :arrow_down:  :left_right_arrow:  :arrow_up:  :arrow_down:   :arrow_up: |
| Apr 04 | `1ad6839` | 6.71 | 22.85 | 13.56 | 29.57 | 0.89 | 16.10 | 64.20 | N/A | 21.07 | :arrow_down:  :left_right_arrow:  :arrow_up:  :arrow_up:  :arrow_down:  :arrow_down:  :arrow_up:   :arrow_down: |
| Apr 04 | `c84f085` | 6.86 | 22.87 | 13.16 | 23.95 | 0.93 | 210.88 | 62.21 | 14.89 | 206.55 | :arrow_up:  :left_right_arrow:  :arrow_down:  :arrow_down:  :left_right_arrow:  :arrow_up:  :left_right_arrow:   :arrow_up: |
| Apr 04 | `95cdbce` | 6.68 | 22.80 | 13.51 | 29.41 | 0.93 | 16.14 | 63.19 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 03 | `90e8680` | 6.72 | 22.89 | 13.57 | 29.56 | 0.93 | 16.09 | 62.17 | N/A | 21.06 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:   :left_right_arrow: |
| Apr 03 | `29d8e95` | 6.69 | 22.88 | 13.55 | 29.62 | 0.93 | 16.57 | 62.18 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:   :arrow_down: |
| Apr 02 | `d7256eb` | 6.70 | 22.87 | 13.18 | 29.59 | 0.93 | 16.11 | 62.20 | N/A | 22.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 01 | `a315d74` | 6.70 | 22.89 | 13.30 | 29.70 | 0.92 | 16.20 | 62.19 | N/A | 22.08 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_up: |
| Apr 01 | `a8759dd` | 6.70 | 22.88 | 13.19 | 30.10 | 0.93 | 16.16 | 62.20 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_up: |
| Mar 31 | `3650bfb` | 6.69 | 22.84 | 13.19 | 29.64 | 0.93 | 16.10 | 63.18 | N/A | 9.03 | -- |

---
*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*
