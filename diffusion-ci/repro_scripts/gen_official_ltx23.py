import gc
import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from urllib.request import urlretrieve

import imageio
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image

import ltx_core.layer_streaming as ltx_layer_streaming
import ltx_core.model.transformer.attention as ltx_attention
import ltx_pipelines.utils.denoisers as ltx_denoisers
import ltx_pipelines.utils.blocks as ltx_blocks
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.helpers import modality_from_latent_state
from sglang.multimodal_gen.test.test_utils import (
    _consistency_gt_filenames,
    extract_key_frames_from_video,
)

try:
    import flashinfer
except ImportError:
    flashinfer = None


REPO_ID = "Lightricks/LTX-2.3"
OUT_DIR = Path("/tmp/mmgen-official-ltx23-report")
IMAGE_URL = (
    "https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/"
    "5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg"
)
SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=["ltx_2.3_two_stage_t2v_2gpus", "ltx_2.3_one_stage_ti2v"],
    )
    parser.add_argument("--checkpoint-name", default="ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--streaming-prefetch-count", type=int, default=0)
    parser.add_argument(
        "--quantization",
        choices=["none", "fp8-cast", "fp8-scaled-mm"],
        default="fp8-cast",
    )
    parser.add_argument("--decode-audio", action="store_true")
    parser.add_argument(
        "--skip-v2a-cross-attn-for-video-gt",
        action="store_true",
        help="Disable the video-to-audio cross-attention branch to reproduce the legacy CI GT.",
    )
    return parser.parse_args()


def resolve_quantization(name: str):
    if name == "none":
        return None
    if name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    return QuantizationPolicy.fp8_scaled_mm()


class NoopAudioDecoder:
    def __call__(self, latent):
        return None

VIDEO_GUIDER = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[28],
)
AUDIO_GUIDER = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[28],
)


class LowMemoryLayerStreamingWrapper(ltx_layer_streaming.LayerStreamingWrapper):
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        self._model = model
        self._layers = ltx_layer_streaming._resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._prefetch_count = min(max(prefetch_count, 0), len(self._layers) - 1)
        self._hooks = []
        self._setup()

    def _setup(self) -> None:
        self._store = ltx_layer_streaming._LayerStore(
            self._layers, self._target_device
        )

        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for tensor in ltx_layer_streaming.itertools.chain(
                layer.parameters(), layer.buffers()
            ):
                layer_tensor_ids.add(id(tensor))

        for param in self._model.parameters():
            if id(param) not in layer_tensor_ids:
                param.data = param.data.to(self._target_device)
        for buffer in self._model.buffers():
            if id(buffer) not in layer_tensor_ids:
                buffer.data = buffer.data.to(self._target_device)

        if len(self._layers):
            self._store.move_to_gpu(0, self._layers[0])

        self._prefetcher = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {
            id(layer): idx for idx, layer in enumerate(self._layers)
        }

        def pre_hook(module: nn.Module, _args, *, idx: int) -> None:
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx, module)

        def post_hook(module: nn.Module, _args, _output, *, idx: int) -> None:
            torch.cuda.synchronize(device=self._target_device)
            self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            self._hooks.extend(
                [
                    layer.register_forward_pre_hook(
                        ltx_layer_streaming.functools.partial(pre_hook, idx=idx)
                    ),
                    layer.register_forward_hook(
                        ltx_layer_streaming.functools.partial(post_hook, idx=idx)
                    ),
                ]
            )


def enable_zero_prefetch_layer_streaming() -> None:
    ltx_layer_streaming.LayerStreamingWrapper = LowMemoryLayerStreamingWrapper
    ltx_blocks.LayerStreamingWrapper = LowMemoryLayerStreamingWrapper


def sequential_guided_denoise(
    transformer,
    video_state,
    audio_state,
    sigma,
    video_guider,
    audio_guider,
    v_context,
    a_context,
    *,
    last_denoised_video,
    last_denoised_audio,
    step_index: int,
):
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        return last_denoised_video, last_denoised_audio

    def maybe_skip_v2a(perturbation: PerturbationConfig) -> PerturbationConfig:
        if not SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT:
            return perturbation
        perturbations = list(perturbation.perturbations or [])
        perturbations.append(
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None)
        )
        return PerturbationConfig(perturbations)

    passes = [("cond", v_context, a_context, PerturbationConfig.empty())]
    if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=video_guider.params.stg_blocks,
            )
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                blocks=audio_guider.params.stg_blocks,
            )
        )
    if stg_perturbations:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(
                            type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                            blocks=None,
                        ),
                        Perturbation(
                            type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                            blocks=None,
                        ),
                    ]
                ),
            )
        )

    results = {}
    for name, video_context, audio_context, perturbation in passes:
        video = (
            modality_from_latent_state(video_state, video_context, sigma, enabled=not v_skip)
            if video_state is not None
            else None
        )
        audio = (
            modality_from_latent_state(audio_state, audio_context, sigma, enabled=not a_skip)
            if audio_state is not None
            else None
        )
        results[name] = transformer(
            video=video,
            audio=audio,
            perturbations=BatchedPerturbationConfig([maybe_skip_v2a(perturbation)]),
        )

    cond_v, cond_a = results["cond"]
    uncond_v, uncond_a = results.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = results.get("ptb", (0.0, 0.0))
    mod_v, mod_a = results.get("mod", (0.0, 0.0))

    denoised_video = (
        last_denoised_video
        if v_skip
        else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
    )
    denoised_audio = (
        last_denoised_audio
        if a_skip
        else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
    )
    return denoised_video, denoised_audio


def enable_low_memory_official_ltx() -> None:
    enable_zero_prefetch_layer_streaming()
    if SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT:
        # Legacy CI GT skipped V2A for all guidance passes. The official
        # default denoiser does not expose that knob, so keep the old helper
        # only for explicit legacy reproduction.
        ltx_denoisers._guided_denoise = sequential_guided_denoise
    if flashinfer is not None:
        original_to_callable = ltx_attention.AttentionFunction.to_callable

        class FlashInferAttention:
            def __call__(self, q, k, v, heads: int, mask=None):
                if mask is not None:
                    return ltx_attention.PytorchAttention()(q, k, v, heads, mask)
                batch, _, inner_dim = q.shape
                dim_head = inner_dim // heads
                q = q.view(batch, -1, heads, dim_head)
                k = k.view(batch, -1, heads, dim_head)
                v = v.view(batch, -1, heads, dim_head)
                outs = [
                    flashinfer.single_prefill_with_kv_cache(
                        q[i].contiguous(),
                        k[i].contiguous(),
                        v[i].contiguous(),
                        causal=False,
                    )
                    for i in range(batch)
                ]
                return torch.stack(outs, dim=0).reshape(batch, -1, inner_dim)

        def to_callable(self):
            if self is ltx_attention.AttentionFunction.DEFAULT:
                return FlashInferAttention()
            return original_to_callable(self)

        ltx_attention.AttentionFunction.to_callable = to_callable


def newest_materialized_ltx23_root() -> Path:
    base = Path("/root/.cache/sgl_diffusion/materialized_models")
    roots = [
        p
        for p in base.glob("Lightricks__LTX-2.3-*")
        if p.is_dir()
        and not p.name.endswith(".tmp")
        and (p / "text_encoder").exists()
        and (p / "ltx-2.3-22b-distilled-lora-384.safetensors").exists()
        and (p / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").exists()
    ]
    if not roots:
        raise FileNotFoundError("No complete materialized LTX-2.3 root found")
    return max(roots, key=lambda p: p.stat().st_mtime)


def prepare_gemma_root(materialized: Path) -> Path:
    bundle = OUT_DIR / "gemma_root"
    bundle.mkdir(parents=True, exist_ok=True)
    for src_dir in (materialized / "text_encoder", materialized / "tokenizer"):
        for src in src_dir.iterdir():
            dst = bundle / src.name
            if not dst.exists():
                dst.symlink_to(src)
    return bundle


def collect_video_frames(video_iter) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for chunk in video_iter:
        arr = chunk.detach().to("cpu").numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.extend([frame[..., :3] for frame in arr])
    return frames


def ci_key_frames(frames: list[np.ndarray], fps: int) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        imageio.mimsave(
            tmp_path,
            frames,
            fps=fps,
            format="mp4",
            codec="libx264",
            quality=5,
        )
        return extract_key_frames_from_video(Path(tmp_path).read_bytes())
    finally:
        os.unlink(tmp_path)


def save_case(case_id: str, frames: list[np.ndarray], fps: int) -> list[str]:
    selected = ci_key_frames(frames, fps)
    saved: list[str] = []
    for frame, filename in zip(
        selected,
        _consistency_gt_filenames(case_id, 2, is_video=True),
        strict=True,
    ):
        Image.fromarray(frame).save(OUT_DIR / filename)
        saved.append(filename)
    return saved


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_snapshot(device: torch.device) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    global OUT_DIR, SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT = args.skip_v2a_cross_attn_for_video_gt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enable_low_memory_official_ltx()
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    device = torch.device(args.device)
    materialized = newest_materialized_ltx23_root()
    checkpoint_path = hf_hub_download(REPO_ID, args.checkpoint_name)
    gemma_root = str(prepare_gemma_root(materialized))
    distilled_lora_path = str(materialized / "ltx-2.3-22b-distilled-lora-384.safetensors")
    upsampler_path = str(materialized / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    quantization = resolve_quantization(args.quantization)
    input_image_path = str(OUT_DIR / "ltx23_ti2v_input.jpg")
    if not Path(input_image_path).exists():
        urlretrieve(IMAGE_URL, input_image_path)

    manifest = {
        "generator": "official-ltx-pipelines",
        "repo_id": REPO_ID,
        "args": vars(args),
        "low_memory_overrides": {
            "torch_inference_mode": True,
            "fp8_quantization": args.quantization,
            "layer_streaming": "synchronous per-layer eviction",
            "guided_denoise": (
                "legacy sequential guidance with global V2A skip"
                if SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT
                else "official batched guidance passes (max_batch_size=1 default)"
            ),
            "skip_v2a_cross_attn_for_video_gt": SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT,
            "decode_audio": bool(args.decode_audio),
            "attention": (
                "flashinfer.single_prefill_with_kv_cache"
                if flashinfer is not None
                else "official default attention"
            ),
        },
        "checkpoint_path": checkpoint_path,
        "materialized_root": str(materialized),
        "cases": [],
        "failures": [],
        "time": time.time(),
    }

    if "ltx_2.3_two_stage_t2v_2gpus" in args.case_ids:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            print("[ltx-official] generating ltx_2.3_two_stage_t2v_2gpus", flush=True)
            pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    distilled_lora_path,
                    1.0,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            device=device,
            quantization=quantization,
        )
            if not args.decode_audio:
                pipe.audio_decoder = NoopAudioDecoder()
            tiling_config = TilingConfig.default()
            video, _audio = pipe(
            prompt="A curious raccoon",
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=42,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.fps,
            num_inference_steps=args.steps,
            video_guider_params=VIDEO_GUIDER,
            audio_guider_params=AUDIO_GUIDER,
            images=[],
            tiling_config=tiling_config,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=1,
        )
            frames = collect_video_frames(video)
            saved = save_case("ltx_2.3_two_stage_t2v_2gpus", frames, fps=args.fps)
            manifest["cases"].append(
                {
                    "case_id": "ltx_2.3_two_stage_t2v_2gpus",
                    "pipeline_class": "TI2VidTwoStagesPipeline",
                    "saved_files": saved,
                    "num_frames": len(frames),
                    "video_chunks_number": get_video_chunks_number(args.num_frames, tiling_config),
                    "cuda_memory": cuda_memory_snapshot(device),
                }
            )
            del pipe, video, _audio, frames
            cleanup_cuda()
        except Exception as exc:
            manifest["failures"].append(
                {
                    "case_id": "ltx_2.3_two_stage_t2v_2gpus",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "cuda_memory": cuda_memory_snapshot(device),
                }
            )
            cleanup_cuda()

    if "ltx_2.3_one_stage_ti2v" in args.case_ids:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            print("[ltx-official] generating ltx_2.3_one_stage_ti2v", flush=True)
            pipe = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            loras=[],
            device=device,
            quantization=quantization,
        )
            if not args.decode_audio:
                pipe.audio_decoder = NoopAudioDecoder()
            video, _audio = pipe(
            prompt=(
                "The man in the picture slowly turns his head, his expression enigmatic "
                "and otherworldly. The camera performs a slow, cinematic dolly out, "
                "focusing on his face. Moody lighting, neon signs glowing in the "
                "background, shallow depth of field."
            ),
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=42,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.fps,
            num_inference_steps=args.steps,
            video_guider_params=VIDEO_GUIDER,
            audio_guider_params=AUDIO_GUIDER,
            images=[
                ImageConditioningInput(
                    path=input_image_path,
                    frame_idx=0,
                    strength=1.0,
                    crf=33,
                )
            ],
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=1,
        )
            frames = collect_video_frames(video)
            saved = save_case("ltx_2.3_one_stage_ti2v", frames, fps=args.fps)
            manifest["cases"].append(
                {
                    "case_id": "ltx_2.3_one_stage_ti2v",
                    "pipeline_class": "TI2VidOneStagePipeline",
                    "saved_files": saved,
                    "num_frames": len(frames),
                    "image_path": input_image_path,
                    "cuda_memory": cuda_memory_snapshot(device),
                }
            )
            del pipe, video, _audio, frames
            cleanup_cuda()
        except Exception as exc:
            manifest["failures"].append(
                {
                    "case_id": "ltx_2.3_one_stage_ti2v",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "cuda_memory": cuda_memory_snapshot(device),
                }
            )
            cleanup_cuda()

    (OUT_DIR / "official_ltx23_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
