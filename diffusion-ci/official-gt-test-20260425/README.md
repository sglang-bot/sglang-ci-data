# Official GT Test Bundle 2026-04-25

This is a non-active test bundle for multimodal generation consistency work.
It intentionally does not replace the active `diffusion-ci/consistency_gt`
ground truth.

## Contents

- `consistency_gt/`: official-comparable GT files for selected CI cases.
- `repro_scripts/`: scripts and wrappers used to reproduce official outputs.
- `manifests/`: raw generation manifests grouped by source backend.
- `report/`: comparison report generated from SGLang native vs official GT.

## Comparable Cases

Diffusers official:

- `fast_hunyuan_video`
- `flux_2_image_t2i`
- `flux_2_klein_image_t2i`
- `flux_2_ti2i`
- `flux_image_t2i`
- `qwen_image_edit_2509_ti2i`
- `qwen_image_edit_2511_ti2i`
- `qwen_image_edit_ti2i`
- `qwen_image_layered_i2i`
- `qwen_image_t2i`
- `zimage_image_t2i`
- `zimage_image_t2i_fp8`

Official repo:

- `wan2_1_t2v_1.3b`: Wan2.1 official repo, MP4 encode plus CI key-frame extraction.
- `wan2_2_ti2v_5b`: Wan2.2 official repo TI2V path. Diffusers output is not used because it failed to inject the image.
- `ltx_2.3_one_stage_ti2v`: LTX-Video official repo script with `torch.inference_mode()`.
- `ltx_2.3_two_stage_t2v_2gpus`: LTX-Video official repo script with `torch.inference_mode()`.

## Notes

Native-only CI cases such as cache/postprocess/custom-VAE/dynamic-LoRA cases are
not included because there is no directly comparable official implementation.
HF tokens are not stored in this bundle.
