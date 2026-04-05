# Diffusion Cross-Framework Performance Dashboard

*Generated: Apr 05 | Commit: `70658bf`*

> [!WARNING]
> **Performance Regression Detected**
>
> - **wan22_t2v_a14b_720p** (sglang): 16.10s -> 209.31s (+1200.1%)
> - **wan22_i2v_a14b_720p** (sglang): 21.07s -> 204.55s (+870.7%)


## Cross-Framework Performance Comparison

| Model | Risk | sglang (s) |
|-------|------|---------|
| FLUX.1-dev | ✅ | **6.68** |
| FLUX.2-dev | ✅ | **22.80** |
| Qwen-Image-2512 | ⚠️ | **14.12** |
| Qwen-Image-Edit-2511 | ✅ | **23.80** |
| Z-Image-Turbo | ✅ | **0.88** |
| Wan2.2-T2V-A14B-Diffusers | ⚠️ | **209.31** |
| Wan2.2-TI2V-5B-Diffusers | ✅ | **62.15** |
| LTX-2 | ✅ | **14.06** |
| Wan2.2-I2V-A14B-Diffusers | ⚠️ | **204.55** |

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
| Apr 05 | `70658bf` | 6.68 | 22.80 | 14.12 | 23.80 | 0.88 | 209.31 | 62.15 | 14.06 | 204.55 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :arrow_down:  :left_right_arrow:  :arrow_up:  :arrow_down:   :arrow_up: |
| Apr 04 | `1ad6839` | 6.71 | 22.85 | 13.56 | 29.57 | 0.89 | 16.10 | 64.20 | N/A | 21.07 | :arrow_down:  :left_right_arrow:  :arrow_up:  :arrow_up:  :arrow_down:  :arrow_down:  :arrow_up:   :arrow_down: |
| Apr 04 | `c84f085` | 6.86 | 22.87 | 13.16 | 23.95 | 0.93 | 210.88 | 62.21 | 14.89 | 206.55 | :arrow_up:  :left_right_arrow:  :arrow_down:  :arrow_down:  :left_right_arrow:  :arrow_up:  :left_right_arrow:   :arrow_up: |
| Apr 04 | `95cdbce` | 6.68 | 22.80 | 13.51 | 29.41 | 0.93 | 16.14 | 63.19 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 03 | `90e8680` | 6.72 | 22.89 | 13.57 | 29.56 | 0.93 | 16.09 | 62.17 | N/A | 21.06 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:   :left_right_arrow: |
| Apr 03 | `29d8e95` | 6.69 | 22.88 | 13.55 | 29.62 | 0.93 | 16.57 | 62.18 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:   :arrow_down: |
| Apr 02 | `d7256eb` | 6.70 | 22.87 | 13.18 | 29.59 | 0.93 | 16.11 | 62.20 | N/A | 22.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 01 | `a315d74` | 6.70 | 22.89 | 13.30 | 29.70 | 0.92 | 16.20 | 62.19 | N/A | 22.08 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_up: |
| Apr 01 | `a8759dd` | 6.70 | 22.88 | 13.19 | 30.10 | 0.93 | 16.16 | 62.20 | N/A | 21.07 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_up: |
| Mar 31 | `3650bfb` | 6.69 | 22.84 | 13.19 | 29.64 | 0.93 | 16.10 | 63.18 | N/A | 9.03 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_down: |
| Mar 30 | `afb32d7` | 6.68 | 22.79 | 13.16 | 29.44 | 0.92 | 16.08 | 62.21 | N/A | 10.04 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Mar 29 | `5958f9d` | 6.70 | 22.86 | 13.19 | 29.74 | 0.93 | 16.31 | 62.18 | N/A | 10.04 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_up: |
| Mar 29 | `3ab9afd` | 6.70 | 22.90 | 13.19 | 29.52 | 0.93 | 16.17 | 62.17 | N/A | 9.04 | :left_right_arrow:  :left_right_arrow:  :arrow_down:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :arrow_down: |
| Mar 28 | `8399708` | 6.68 | 22.76 | 13.71 | 31.67 | 0.93 | 16.27 | 62.22 | N/A | 10.04 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :arrow_up:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Mar 27 | `c2b3e42` | 6.70 | 22.94 | 13.18 | 29.64 | 0.93 | 16.11 | 62.21 | N/A | 10.04 | -- |

> [!CAUTION]
> **Action Required — Performance Alert**
>
> The following cases need attention:
> - qwen_image_2512_t2i_1024: SGLang regression +5.3% vs 3-run avg (14.12s vs 13.41s)
> - wan22_t2v_a14b_720p: SGLang regression +158.3% vs 3-run avg (209.31s vs 81.04s)
> - wan22_i2v_a14b_720p: SGLang regression +146.7% vs 3-run avg (204.55s vs 82.90s)


---
*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*
