import os
import subprocess
import traceback
from pathlib import Path

import redis
from celery.signals import worker_process_init

from api.celery_app import celery_app
from api.config import (
    CONFIG_PATH,
    PRETRAINED_MODEL_PATH,
    TRANSFORMER_PATH,
    WAV2VEC_PATH,
    GPU_MEMORY_MODE,
    OUTPUT_DIR,
    REDIS_URL,
    JOB_TTL_SECONDS,
    NEGATIVE_PROMPT,
)

_MODELS = None
_REDIS = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _set_status(job_id: str, status: str, **extra):
    key = f"job:{job_id}"
    payload = {"status": status, **{k: str(v) for k, v in extra.items() if v is not None}}
    _REDIS.hset(key, mapping=payload)
    _REDIS.expire(key, JOB_TTL_SECONDS)


@worker_process_init.connect
def _load_models_on_worker_init(**_kwargs):
    """Load all models once when the Celery worker process starts."""
    global _MODELS
    from api.inference_runtime import load_models
    print("[worker] loading StableAvatar models...")
    _MODELS = load_models(
        config_path=CONFIG_PATH,
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        transformer_path=TRANSFORMER_PATH,
        wav2vec_path=WAV2VEC_PATH,
        gpu_memory_mode=GPU_MEMORY_MODE,
    )
    print("[worker] models loaded; worker ready.")


def _mux_audio(silent_video: str, audio_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@celery_app.task(name="generate_video")
def generate_video_task(
    job_id: str,
    reference_path: str,
    audio_path: str,
    prompt: str,
    width: int,
    height: int,
    sample_steps: int,
    text_guidance: float,
    audio_guidance: float,
    overlap_window_length: int,
    clip_sample_n_frames: int,
    motion_frame: int,
    seed: int,
    fps: int,
):
    from api.inference_runtime import generate_video

    _set_status(job_id, "running")
    try:
        job_out_dir = Path(OUTPUT_DIR) / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)
        silent_path = str(job_out_dir / "video_without_audio.mp4")
        final_path = str(job_out_dir / "video.mp4")

        generate_video(
            models=_MODELS,
            reference_path=reference_path,
            audio_path=audio_path,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            output_path=silent_path,
            width=width,
            height=height,
            sample_steps=sample_steps,
            text_guidance=text_guidance,
            audio_guidance=audio_guidance,
            overlap_window_length=overlap_window_length,
            clip_sample_n_frames=clip_sample_n_frames,
            motion_frame=motion_frame,
            seed=seed,
            fps=fps,
        )

        _mux_audio(silent_path, audio_path, final_path)
        try:
            os.remove(silent_path)
        except OSError:
            pass

        _set_status(job_id, "done", video_path=final_path)
        return {"job_id": job_id, "status": "done", "video_path": final_path}
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[worker] job {job_id} failed: {err}")
        _set_status(job_id, "failed", error=str(e))
        raise
