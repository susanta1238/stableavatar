# StableAvatar API

FastAPI + Celery + Redis wrapper around `inference.py`.
Single-GPU, one job at a time. Model is loaded once and reused.

API deps (fastapi, celery, redis, uvicorn, python-multipart, pydantic) are included
in the root `requirements.txt`.

## Architecture

```
client ──HTTP──> FastAPI (uvicorn :8000) ──Redis──> Celery worker (concurrency=1)
                       │                                   │
                       └── /workspace/uploads/             └── loads model once
                       └── /workspace/outputs/             └── runs inference + ffmpeg mux
```

## Deploy on RunPod

1. Launch a RunPod **Pod** (not Serverless) with an A100 80GB, 200GB volume mounted at `/workspace`, PyTorch template.
2. SSH in and clone the repo into `/workspace/StableAvatar`.
3. Run the one-shot setup (installs deps + downloads ~30 GB of checkpoints):
   ```bash
   cd /workspace/StableAvatar
   bash api/setup.sh
   ```
4. Expose port **8000** in the RunPod pod config (HTTP). You'll get a URL like
   `https://<pod-id>-8000.proxy.runpod.net`.
5. Start the service:
   ```bash
   export API_KEY=your-long-random-key
   bash api/start.sh
   ```
   First request will wait ~60s for model load on worker boot. Check `/workspace/celery.log`.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `API_KEY` | (required) | Clients send this in `X-API-Key` header |
| `STABLEAVATAR_ROOT` | `/workspace/StableAvatar` | Repo root |
| `STABLEAVATAR_WORK` | `/workspace` | Where uploads/outputs go |
| `STABLEAVATAR_TRANSFORMER` | `.../transformer3d-square.pt` | Swap to `transformer3d-rec-vec.pt` for 480×832 / 832×480 |
| `STABLEAVATAR_GPU_MODE` | `model_full_load` | Or `model_cpu_offload` / `sequential_cpu_offload` |
| `REDIS_URL` | `redis://localhost:6379/0` | |

## Endpoints

### `POST /jobs`
Multipart form. Returns `{job_id}` immediately (inference runs async).

| Field | Type | Required | Default |
|---|---|---|---|
| `image` | file (png/jpg) | yes | — |
| `audio` | file (wav) | yes | — |
| `prompt` | string | yes | — |
| `width` | int | no | 512 |
| `height` | int | no | 512 |
| `sample_steps` | int | no | 30 |
| `text_guidance` | float | no | 3.0 |
| `audio_guidance` | float | no | 5.0 |
| `overlap_window_length` | int | no | 5 |
| `clip_sample_n_frames` | int | no | 81 |
| `motion_frame` | int | no | 25 |
| `seed` | int | no | 42 |

Allowed resolutions: **512×512**, **480×832**, **832×480**.

### `GET /jobs/{job_id}`
Returns `{job_id, status, video_url?, error?}` where `status ∈ {queued, running, done, failed}`.

### `GET /videos/{job_id}`
Streams the final muxed MP4 (video + audio).

### `GET /health`
No auth. Returns Redis + GPU status.

## Example with curl

```bash
URL=https://<pod-id>-8000.proxy.runpod.net
KEY=your-long-random-key

# submit
JOB=$(curl -s -X POST "$URL/jobs" \
    -H "X-API-Key: $KEY" \
    -F image=@face.png \
    -F audio=@speech.wav \
    -F prompt="A woman speaking in a bright office" \
    -F width=512 -F height=512 -F sample_steps=30 \
    | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# poll
while true; do
    STATUS=$(curl -s -H "X-API-Key: $KEY" "$URL/jobs/$JOB" \
        | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "status: $STATUS"
    [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ] && break
    sleep 10
done

# download
curl -s -H "X-API-Key: $KEY" "$URL/videos/$JOB" -o output.mp4
```

## Example with Python

```python
import requests, time

URL = "https://<pod-id>-8000.proxy.runpod.net"
KEY = "your-long-random-key"
H = {"X-API-Key": KEY}

job = requests.post(f"{URL}/jobs", headers=H, files={
    "image": open("face.png", "rb"),
    "audio": open("speech.wav", "rb"),
}, data={
    "prompt": "A woman speaking in a bright office",
    "width": 512, "height": 512, "sample_steps": 30,
}).json()["job_id"]

while True:
    s = requests.get(f"{URL}/jobs/{job}", headers=H).json()
    if s["status"] in ("done", "failed"):
        break
    time.sleep(10)

if s["status"] == "done":
    open("output.mp4", "wb").write(
        requests.get(f"{URL}/videos/{job}", headers=H).content
    )
```

## Limits enforced

- image ≤ 10 MB, png/jpg/jpeg
- audio ≤ 50 MB, wav only
- audio duration ≤ 300 s
- `sample_steps` in [20, 60]
- guidance scales in [1.0, 10.0]
- resolution must be one of the three allowed pairs

## Troubleshooting

- **First request hangs** — worker is still loading model (~60s). Watch `/workspace/celery.log`.
- **OOM** — switch `STABLEAVATAR_GPU_MODE=model_cpu_offload` (half VRAM, slower).
- **`ulysses_degree` / multi-GPU** — not supported in this API wrapper. Use the original `multiple_gpu_inference.sh` directly.
- **Resolution mismatch** — use the matching transformer checkpoint: `transformer3d-square.pt` for 512², `transformer3d-rec-vec.pt` for the rectangular ones.
- **Worker died** — `tail -f /workspace/celery.log`. Kill with `pkill -f 'celery.*worker'` and restart.
