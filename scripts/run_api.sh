#!/usr/bin/env bash
# Run API (từ thư mục gốc repo). Đổi chỗ đọc artifact bằng MUSIC_APP_INDEX và MUSIC_ALS_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn api.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
