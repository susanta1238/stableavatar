#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[setup] installing system packages..."
apt-get update -y
apt-get install -y redis-server ffmpeg git-lfs

echo "[setup] installing python deps..."
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir "huggingface_hub[cli]"

mkdir -p checkpoints
if [ ! -d "checkpoints/StableAvatar-1.3B" ]; then
    echo "[setup] downloading StableAvatar checkpoints (large, be patient)..."
    huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
else
    echo "[setup] checkpoints already present, skipping download."
fi

echo "[setup] done. Next: export API_KEY=... && bash api/start.sh"
