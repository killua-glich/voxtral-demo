#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Installing backend dependencies..."
pip install -r backend/requirements.txt

echo "==> Starting FastAPI backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

# Wait for backend to be ready
echo "==> Waiting for backend..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/voices > /dev/null 2>&1; then
    echo "    Backend ready."
    break
  fi
  sleep 1
done

echo "==> Starting Gradio frontend via Docker Compose..."
docker compose up --build
