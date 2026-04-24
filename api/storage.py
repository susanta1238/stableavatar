import os
import shutil
import uuid
import wave
import contextlib
from pathlib import Path
from fastapi import HTTPException, UploadFile

from api.config import (
    UPLOAD_DIR,
    MAX_IMAGE_BYTES,
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    ALLOWED_IMAGE_EXTS,
    ALLOWED_AUDIO_EXTS,
)


def new_job_id() -> str:
    return uuid.uuid4().hex


def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file too large (>{max_bytes} bytes)")
            out.write(chunk)


def save_uploads(job_id: str, image: UploadFile, audio: UploadFile) -> tuple[str, str]:
    img_ext = Path(image.filename or "").suffix.lower()
    if img_ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(415, f"image must be one of {ALLOWED_IMAGE_EXTS}")
    aud_ext = Path(audio.filename or "").suffix.lower()
    if aud_ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(415, f"audio must be one of {ALLOWED_AUDIO_EXTS}")

    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / f"reference{img_ext}"
    audio_path = job_dir / f"audio{aud_ext}"

    _save_upload(image, image_path, MAX_IMAGE_BYTES)
    _save_upload(audio, audio_path, MAX_AUDIO_BYTES)

    _validate_wav_duration(audio_path)
    return str(image_path), str(audio_path)


def _validate_wav_duration(path: Path) -> None:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate) if rate else 0
    except wave.Error as e:
        raise HTTPException(400, f"invalid wav file: {e}")
    if duration <= 0:
        raise HTTPException(400, "audio has zero duration")
    if duration > MAX_AUDIO_SECONDS:
        raise HTTPException(400, f"audio longer than {MAX_AUDIO_SECONDS}s")


def cleanup_job(job_id: str) -> None:
    shutil.rmtree(Path(UPLOAD_DIR) / job_id, ignore_errors=True)
