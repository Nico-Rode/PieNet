"""
PieNet Cloud Dashboard — receives data from Raspberry Pi, serves historical charts.

Run locally:
  pip install -r requirements.txt
  uvicorn app:app --reload --port 8000

Env vars:
  API_KEY            — shared secret for /api/ingest and /ws/feed
  SPACES_BUCKET      — DO Spaces bucket name for snapshot URLs
  SPACES_REGION      — DO Spaces region (default: nyc3)
"""
from __future__ import annotations

import asyncio
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PieNet Dashboard")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/pienet_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "history.csv"
API_KEY = os.environ.get("API_KEY", "changeme")

CSV_FIELDS = [
    "timestamp", "count", "wait_min", "status",
    "tracked_ids", "service_rate", "snapshot_url",
]


def _ensure_csv() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


_ensure_csv()


def _verify_key(x_api_key: str | None) -> None:
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# ── WebSocket video relay ──────────────────────────────────────────────────

_viewers: set[WebSocket] = set()
_latest_frame: bytes = b""
_frame_event: asyncio.Event = asyncio.Event()


@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket, api_key: str = ""):
    """Producer endpoint — the Pi connects here and pushes JPEG frames."""
    global _latest_frame
    if api_key != API_KEY:
        await ws.close(code=4001, reason="invalid api key")
        return
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            _latest_frame = data
            _frame_event.set()
            _frame_event.clear()
            dead: list[WebSocket] = []
            for viewer in list(_viewers):
                try:
                    await viewer.send_bytes(data)
                except Exception:
                    dead.append(viewer)
            for v in dead:
                _viewers.discard(v)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    """Consumer endpoint — browser clients connect here to watch the feed."""
    await ws.accept()
    _viewers.add(ws)
    try:
        if _latest_frame:
            await ws.send_bytes(_latest_frame)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _viewers.discard(ws)


# ── Ingest endpoint (Pi pushes data here) ──────────────────────────────────

@app.post("/api/ingest")
async def ingest(request: Request, x_api_key: str | None = Header(None)):
    _verify_key(x_api_key)
    body = await request.json()
    row = {k: body.get(k, "") for k in CSV_FIELDS}
    if not row.get("timestamp"):
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
    return {"ok": True}


# ── History endpoint ────────────────────────────────────────────────────────

@app.get("/api/history")
async def history(hours: float = 24):
    if not CSV_PATH.exists():
        return JSONResponse([])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows: list[dict] = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_str = row.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    row["count"] = int(row.get("count", 0))
                    row["wait_min"] = float(row.get("wait_min", 0))
                    rows.append(row)
            except (ValueError, TypeError):
                continue
    return JSONResponse(rows)


# ── Latest data point ──────────────────────────────────────────────────────

@app.get("/api/latest")
async def latest():
    if not CSV_PATH.exists():
        return JSONResponse({})
    last_row: dict = {}
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if last_row:
        last_row["count"] = int(last_row.get("count", 0))
        last_row["wait_min"] = float(last_row.get("wait_min", 0))
    return JSONResponse(last_row)


# ── Serve static dashboard ─────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>PieNet Dashboard</h1><p>static/index.html not found</p>")
