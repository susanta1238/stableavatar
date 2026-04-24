"""Refactored from inference.py: split into load_models() + generate_video()
so the model is loaded once and reused across requests.
Single-GPU only (no ulysses/ring parallelism)."""

import os
import torch
import librosa
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from diffusers import FlowMatchEulerDiscreteScheduler

from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline
from wan.utils.fp8_optimization import (
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
    convert_model_weight_to_float8,
)
from wan.utils.utils import get_image_to_video_latent, save_videos_grid


def _filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def load_models(
    config_path: str,
    pretrained_model_path: str,
    transformer_path: str,
    wav2vec_path: str,
    gpu_memory_mode: str = "model_full_load",
):
    """Load all models once. Returns a dict held by the worker for its lifetime."""
    device = "cuda"
    weight_dtype = torch.bfloat16
    config = OmegaConf.load(config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            pretrained_model_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            pretrained_model_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            pretrained_model_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
    wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_path).to("cpu")

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            pretrained_model_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    ).eval()

    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(
            pretrained_model_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        print(f"Loading transformer checkpoint: {transformer_path}")
        state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    scheduler = FlowMatchEulerDiscreteScheduler(
        **_filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )

    pipeline = WanI2VTalkingInferenceLongPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer3d,
        clip_image_encoder=clip_image_encoder,
        scheduler=scheduler,
        wav2vec_processor=wav2vec_processor,
        wav2vec=wav2vec,
    )

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation"], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation"])
        convert_weight_dtype_wrapper(transformer3d, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    return {
        "pipeline": pipeline,
        "vae": vae,
        "device": device,
        "config": config,
    }


def generate_video(
    models: dict,
    reference_path: str,
    audio_path: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    width: int,
    height: int,
    sample_steps: int,
    text_guidance: float,
    audio_guidance: float,
    overlap_window_length: int,
    clip_sample_n_frames: int,
    motion_frame: int,
    seed: int,
    fps: int = 25,
) -> str:
    """Run one generation. Returns path to the written silent MP4."""
    pipeline = models["pipeline"]
    vae = models["vae"]
    device = models["device"]

    generator = torch.Generator(device=device).manual_seed(seed)
    temporal_ratio = vae.config.temporal_compression_ratio
    if clip_sample_n_frames != 1:
        video_length = int((clip_sample_n_frames - 1) // temporal_ratio * temporal_ratio) + 1
    else:
        video_length = 1

    with torch.no_grad():
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            reference_path, None, video_length=video_length, sample_size=[height, width]
        )
        sr = 16000
        vocal_input, _ = librosa.load(audio_path, sr=sr)

        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=6.0,
            generator=generator,
            num_inference_steps=sample_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=text_guidance,
            audio_guide_scale=audio_guidance,
            vocal_input_values=vocal_input,
            motion_frame=motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=reference_path,
            seed=seed,
            overlap_window_length=overlap_window_length,
            overlapping_weight_scheme="uniform",
            clip_length=clip_sample_n_frames,
        ).videos

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_videos_grid(sample, output_path, fps=fps)

    return output_path
