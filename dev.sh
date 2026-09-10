#!/usr/bin/env bash
# Start the backend and frontend together; Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d backend/.venv ]; then
  echo "Creating the backend virtualenv..."
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -q -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

pids=()
cleanup() {
  echo
  echo "Shutting down..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Backend  → http://127.0.0.1:8000  (docs at /docs)"
(cd backend && exec .venv/bin/uvicorn app.main:app --reload --port 8000) &
pids+=($!)

echo "Frontend → http://localhost:3000"
(cd frontend && exec npm run dev) &
pids+=($!)

wait
