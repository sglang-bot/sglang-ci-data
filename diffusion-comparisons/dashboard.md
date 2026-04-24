# Diffusion Cross-Framework Performance Dashboard

*Generated: Apr 24 | Commit: `c016635`*

> [!WARNING]
> **Performance Regression Detected**
>
> - **qwen_image_2512_t2i_1024** (sglang): 13.03s -> 13.77s (+5.6%)
> - **ltx2.3_twostage_ti2v_2gpus** (sglang): 58.19s -> 69.23s (+19.0%)


## Cross-Framework Performance Comparison

| Model | Risk | sglang (s) |
|-------|------|---------|
| FLUX.1-dev | ✅ | **6.71** |
| FLUX.2-dev | ✅ | **22.81** |
| Qwen-Image-2512 | ⚠️ | **13.77** |
| Qwen-Image-Edit-2511 | ✅ | **23.83** |
| Z-Image-Turbo | ✅ | **0.89** |
| Wan2.2-T2V-A14B-Diffusers | ✅ | **211.38** |
| Wan2.2-TI2V-5B-Diffusers | ✅ | **62.22** |
| LTX-2 | ✅ | **16.09** |
| LTX-2.3 | ⚠️ | **69.23** |
| Wan2.2-I2V-A14B-Diffusers | ✅ | **209.67** |

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


### Latency Trend: ltx2.3_twostage_ti2v_2gpus

![Latency Trend ltx2.3_twostage_ti2v_2gpus](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_ltx2.3_twostage_ti2v_2gpus.png)


### Latency Trend: wan22_i2v_a14b_720p

![Latency Trend wan22_i2v_a14b_720p](https://raw.githubusercontent.com/sglang-bot/sglang-ci-data/main/diffusion-comparisons/charts/latency_wan22_i2v_a14b_720p.png)


## SGLang Performance Trend (Last 15 Runs)

| Date | Commit | flux1_dev_t2i_1024 (s) | flux2_dev_t2i_1024 (s) | qwen_image_2512_t2i_1024 (s) | qwen_image_edit_2511 (s) | zimage_turbo_t2i_1024 (s) | wan22_t2v_a14b_720p (s) | wan22_ti2v_5b_720p (s) | ltx2_twostage_t2v (s) | ltx2.3_twostage_ti2v_2gpus (s) | wan22_i2v_a14b_720p (s) | Trend |
|------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
| Apr 24 | `c016635` | 6.71 | 22.81 | 13.77 | 23.83 | 0.89 | 211.38 | 62.22 | 16.09 | 69.23 | 209.67 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:    :arrow_down:  :arrow_up:  |
| Apr 23 | `c689f77` | 6.60 | 22.62 | 13.03 | 23.68 | 0.88 | N/A | N/A | 16.75 | 58.19 | N/A | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:    :arrow_up:  :arrow_down:  |
| Apr 18 | `9c47bba` | 6.68 | 22.82 | 13.13 | 23.78 | 0.89 | 210.45 | 62.17 | 15.32 | 60.19 | 206.66 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow: |
| Apr 17 | `3d2d57c` | 6.69 | 22.82 | 13.12 | 23.80 | 0.89 | 210.28 | 62.19 | 15.47 | 62.20 | 205.63 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :arrow_up:  :left_right_arrow: |
| Apr 16 | `a4cf2ea` | 6.70 | 22.90 | 14.08 | 23.93 | 0.89 | 210.73 | 62.15 | 14.86 | 59.20 | 206.60 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 15 | `2c9e76d` | 6.69 | 22.78 | 13.14 | 23.82 | 0.89 | 209.49 | 63.18 | 14.97 | N/A | 206.58 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:   :left_right_arrow: |
| Apr 14 | `c456cba` | 6.70 | 22.85 | 13.17 | 23.83 | 0.89 | 210.52 | 62.16 | 14.58 | N/A | 206.54 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 13 | `37fc47c` | 6.60 | 22.61 | 12.99 | 23.60 | 0.88 | 209.76 | 61.17 | 14.74 | N/A | 207.63 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:   :left_right_arrow: |
| Apr 10 | `5638d40` | 6.68 | 22.78 | 13.16 | 23.76 | 0.88 | 209.41 | 62.16 | 14.21 | N/A | 204.58 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:   :left_right_arrow: |
| Apr 09 | `2c4e113` | 6.71 | 22.93 | 13.20 | 23.80 | 0.89 | 210.35 | 62.16 | 14.71 | N/A | 206.54 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:   :left_right_arrow: |
| Apr 08 | `dd73e9a` | 6.83 | 22.96 | 13.19 | 23.84 | 0.89 | 210.25 | 62.20 | 15.24 | N/A | 206.65 | :arrow_up:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 07 | `5cc246e` | 6.69 | 22.92 | 13.19 | 23.83 | 0.89 | 210.64 | 62.20 | 15.11 | N/A | 207.62 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:   :left_right_arrow: |
| Apr 06 | `93109cc` | 6.70 | 22.86 | 13.61 | 23.92 | 0.89 | 210.73 | 62.15 | 14.13 | N/A | 207.67 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 05 | `70658bf` | 6.68 | 22.80 | 14.12 | 23.80 | 0.88 | 209.31 | 62.15 | 14.06 | N/A | 204.55 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :arrow_down:  :left_right_arrow:  :arrow_up:  :arrow_down:    :arrow_up: |
| Apr 04 | `1ad6839` | 6.71 | 22.85 | 13.56 | 29.57 | 0.89 | 16.10 | 64.20 | N/A | N/A | 21.07 | -- |

> [!CAUTION]
> **Action Required — Performance Alert**
>
> The following cases need attention:
> - qwen_image_2512_t2i_1024: SGLang regression +5.1% vs 3-run avg (13.77s vs 13.10s)
> - ltx2.3_twostage_ti2v_2gpus: SGLang regression +15.0% vs 3-run avg (69.23s vs 60.19s)


---
*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*
