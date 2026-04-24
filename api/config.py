import os
from pathlib import Path

REPO_ROOT = Path(os.getenv("STABLEAVATAR_ROOT", "/workspace/StableAvatar"))
WORK_ROOT = Path(os.getenv("STABLEAVATAR_WORK", "/workspace"))

UPLOAD_DIR = WORK_ROOT / "uploads"
OUTPUT_DIR = WORK_ROOT / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CONFIG_PATH = os.getenv(
    "STABLEAVATAR_CONFIG",
    str(REPO_ROOT / "deepspeed_config" / "wan2.1" / "wan_civitai.yaml"),
)
PRETRAINED_MODEL_PATH = os.getenv(
    "STABLEAVATAR_PRETRAINED",
    str(CHECKPOINTS_DIR / "Wan2.1-Fun-V1.1-1.3B-InP"),
)
TRANSFORMER_PATH = os.getenv(
    "STABLEAVATAR_TRANSFORMER",
    str(CHECKPOINTS_DIR / "StableAvatar-1.3B" / "transformer3d-square.pt"),
)
WAV2VEC_PATH = os.getenv(
    "STABLEAVATAR_WAV2VEC",
    str(CHECKPOINTS_DIR / "wav2vec2-base-960h"),
)
GPU_MEMORY_MODE = os.getenv("STABLEAVATAR_GPU_MODE", "model_full_load")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
API_KEY = os.getenv("API_KEY", "change-me")

JOB_TTL_SECONDS = 60 * 60 * 24
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_AUDIO_SECONDS = 300

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTS = {".wav"}

ALLOWED_RESOLUTIONS = {(512, 512), (480, 832), (832, 480)}

DEFAULTS = {
    "width": 512,
    "height": 512,
    "sample_steps": 30,
    "text_guidance": 3.0,
    "audio_guidance": 5.0,
    "overlap_window_length": 5,
    "clip_sample_n_frames": 81,
    "motion_frame": 25,
    "seed": 42,
    "fps": 25,
}

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
