from pathlib import Path

import redis
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.config import (
    ALLOWED_RESOLUTIONS,
    DEFAULTS,
    OUTPUT_DIR,
    REDIS_URL,
)
from api.schemas import HealthResponse, JobCreatedResponse, JobStatusResponse
from api.storage import new_job_id, save_uploads
from api.celery_app import celery_app

app = FastAPI(title="StableAvatar API")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        model_loaded=redis_ok,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/jobs", response_model=JobCreatedResponse)
def create_job(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(...),
    width: int = Form(DEFAULTS["width"]),
    height: int = Form(DEFAULTS["height"]),
    sample_steps: int = Form(DEFAULTS["sample_steps"]),
    text_guidance: float = Form(DEFAULTS["text_guidance"]),
    audio_guidance: float = Form(DEFAULTS["audio_guidance"]),
    overlap_window_length: int = Form(DEFAULTS["overlap_window_length"]),
    clip_sample_n_frames: int = Form(DEFAULTS["clip_sample_n_frames"]),
    motion_frame: int = Form(DEFAULTS["motion_frame"]),
    seed: int = Form(DEFAULTS["seed"]),
):
    if (width, height) not in ALLOWED_RESOLUTIONS:
        raise HTTPException(400, f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
    if not (20 <= sample_steps <= 60):
        raise HTTPException(400, "sample_steps must be in [20, 60]")
    if not (1.0 <= text_guidance <= 10.0):
        raise HTTPException(400, "text_guidance must be in [1.0, 10.0]")
    if not (1.0 <= audio_guidance <= 10.0):
        raise HTTPException(400, "audio_guidance must be in [1.0, 10.0]")
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")

    job_id = new_job_id()
    image_path, audio_path = save_uploads(job_id, image, audio)

    r.hset(f"job:{job_id}", mapping={"status": "queued"})
    r.expire(f"job:{job_id}", 60 * 60 * 24)

    celery_app.send_task(
        "generate_video",
        kwargs=dict(
            job_id=job_id,
            reference_path=image_path,
            audio_path=audio_path,
            prompt=prompt,
            width=width,
            height=height,
            sample_steps=sample_steps,
            text_guidance=text_guidance,
            audio_guidance=audio_guidance,
            overlap_window_length=overlap_window_length,
            clip_sample_n_frames=clip_sample_n_frames,
            motion_frame=motion_frame,
            seed=seed,
            fps=DEFAULTS["fps"],
        ),
    )
    return JobCreatedResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    data = r.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(404, "job not found")
    status = data.get("status", "unknown")
    resp = JobStatusResponse(job_id=job_id, status=status)
    if status == "done":
        resp.video_url = f"/videos/{job_id}"
    elif status == "failed":
        resp.error = data.get("error")
    return resp


@app.get("/videos/{job_id}")
def download_video(job_id: str):
    path = Path(OUTPUT_DIR) / job_id / "video.mp4"
    if not path.exists():
        raise HTTPException(404, "video not found")
    return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}.mp4")
