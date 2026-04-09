#!/usr/bin/env python3
"""
PieNet Line Monitor — detect and track people in camera feed, estimate queue wait times.

Uses SSD MobileNet V2 (COCO) on Edge TPU for real-time person detection.
Includes centroid tracking, service rate learning, DO Spaces snapshot upload,
cloud dashboard push, and a local MJPEG stream.

  source ~/coral-tpu-venv/bin/activate
  python3 line_monitor.py --config config.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tflite_runtime.interpreter as tflite

log = logging.getLogger("pienet")

MODEL_URL = (
    "https://github.com/google-coral/test_data/raw/master/"
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)
LABELS_URL = (
    "https://github.com/google-coral/test_data/raw/master/"
    "coco_labels.txt"
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def ensure_file(url: str, path: str) -> None:
    if os.path.isfile(path):
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_labels(path: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[i] = line
    return labels


def load_config(path: str | None) -> dict:
    defaults = {
        "camera": {"width": 640, "height": 480},
        "detection": {"confidence_threshold": 0.4, "person_class_id": 0},
        "tracking": {"max_disappeared": 15},
        "roi": {"enabled": False, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "wait_estimation": {
            "minutes_per_person": 3.0,
            "auto_learn": True,
            "learning_interval_seconds": 300,
            "status_thresholds": {"no_line": 0, "short": 3, "moderate": 8, "long": 15},
        },
        "stream": {"port": 8080},
        "logging": {"enabled": True, "csv_path": "~/pienet_log.csv", "interval_seconds": 30},
        "snapshots": {
            "enabled": False,
            "interval_seconds": 300,
            "spaces_bucket": "",
            "spaces_region": "nyc3",
            "spaces_endpoint": "https://nyc3.digitaloceanspaces.com",
        },
        "cloud": {
            "enabled": False,
            "endpoint": "",
            "api_key": "",
            "push_interval_seconds": 30,
        },
    }
    if path and os.path.isfile(path):
        with open(path) as f:
            user = json.load(f)
        _deep_merge(defaults, user)
    return defaults


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ── Camera capture ──────────────────────────────────────────────────────────


def capture_frame(width: int, height: int) -> Image.Image | None:
    try:
        proc = subprocess.run(
            [
                "rpicam-still", "--immediate", "--nopreview",
                "--output", "-",
                "--width", str(width), "--height", str(height),
                "--encoding", "jpg", "-t", "1",
            ],
            capture_output=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except Exception as e:
        log.warning("capture error: %s", e)
        return None


# ── Detection ───────────────────────────────────────────────────────────────


class PersonDetector:
    def __init__(self, model_path: str, edgetpu_lib: str = "libedgetpu.so.1"):
        delegate = tflite.load_delegate(edgetpu_lib)
        self.interpreter = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate],
        )
        self.interpreter.allocate_tensors()
        self._inp = self.interpreter.get_input_details()[0]
        self._outs = self.interpreter.get_output_details()
        self._h = int(self._inp["shape"][1])
        self._w = int(self._inp["shape"][2])

    def detect(self, image: Image.Image, threshold: float, person_class: int) -> list[dict]:
        resized = image.resize((self._w, self._h), Image.LANCZOS)
        input_data = np.asarray(resized, dtype=np.uint8)[np.newaxis]
        self.interpreter.set_tensor(self._inp["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self._outs[0]["index"])[0]
        classes = self.interpreter.get_tensor(self._outs[1]["index"])[0]
        scores = self.interpreter.get_tensor(self._outs[2]["index"])[0]
        count = int(self.interpreter.get_tensor(self._outs[3]["index"])[0])

        results = []
        for i in range(min(count, len(scores))):
            if scores[i] < threshold:
                continue
            if int(classes[i]) != person_class:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            results.append({
                "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "score": float(scores[i]),
            })
        return results


# ── Centroid Tracker ────────────────────────────────────────────────────────


class CentroidTracker:
    """Track detected people across frames using centroid distance matching."""

    def __init__(self, max_disappeared: int = 15):
        self._next_id = 0
        self._objects: dict[int, np.ndarray] = {}
        self._disappeared: dict[int, int] = {}
        self._max_disappeared = max_disappeared

    @property
    def object_count(self) -> int:
        return len(self._objects)

    @property
    def total_ids_assigned(self) -> int:
        return self._next_id

    def _register(self, centroid: np.ndarray) -> int:
        oid = self._next_id
        self._objects[oid] = centroid
        self._disappeared[oid] = 0
        self._next_id += 1
        return oid

    def _deregister(self, oid: int) -> None:
        del self._objects[oid]
        del self._disappeared[oid]

    def update(self, detections: list[dict]) -> dict[int, dict]:
        """Match detections to tracked objects; return {object_id: detection_dict}."""
        if not detections:
            for oid in list(self._disappeared):
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self._max_disappeared:
                    self._deregister(oid)
            return {}

        input_centroids = np.array([
            [
                (d["box"][0] + d["box"][2]) / 2,
                (d["box"][1] + d["box"][3]) / 2,
            ]
            for d in detections
        ])

        if not self._objects:
            tracked = {}
            for i, det in enumerate(detections):
                oid = self._register(input_centroids[i])
                tracked[oid] = det
            return tracked

        object_ids = list(self._objects.keys())
        object_centroids = np.array(list(self._objects.values()))

        # Pairwise distance matrix
        dist = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy assignment: for each existing object, find closest input
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        tracked: dict[int, dict] = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self._objects[oid] = input_centroids[col]
            self._disappeared[oid] = 0
            tracked[oid] = detections[col]
            used_rows.add(row)
            used_cols.add(col)

        # Mark unmatched existing objects as disappeared
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self._disappeared[oid] += 1
            if self._disappeared[oid] > self._max_disappeared:
                self._deregister(oid)

        # Register new detections that weren't matched
        for col in set(range(len(detections))) - used_cols:
            oid = self._register(input_centroids[col])
            tracked[oid] = detections[col]

        return tracked


# ── Service Rate Learner ────────────────────────────────────────────────────


class ServiceRateLearner:
    """Auto-calibrate minutes_per_person from observed count trends."""

    def __init__(
        self,
        default_mpp: float = 3.0,
        learning_interval: float = 300.0,
        window_size: int = 60,
        alpha: float = 0.3,
    ):
        self.minutes_per_person = default_mpp
        self._default_mpp = default_mpp
        self._learning_interval = learning_interval
        self._alpha = alpha
        self._samples: collections.deque[tuple[float, int]] = collections.deque(maxlen=window_size)
        self._last_learn_time = time.time()

    def record(self, timestamp: float, count: int) -> None:
        self._samples.append((timestamp, count))

    def maybe_learn(self) -> float | None:
        now = time.time()
        if now - self._last_learn_time < self._learning_interval:
            return None
        self._last_learn_time = now
        if len(self._samples) < 10:
            return None
        return self._compute()

    def _compute(self) -> float | None:
        samples = list(self._samples)
        decreases: list[float] = []
        for i in range(1, len(samples)):
            t0, c0 = samples[i - 1]
            t1, c1 = samples[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            dc = c0 - c1
            if dc > 0 and c0 > 0:
                rate = dc / (dt / 60.0)
                decreases.append(rate)

        if not decreases:
            return None

        avg_rate = sum(decreases) / len(decreases)
        if avg_rate <= 0:
            return None

        learned_mpp = 1.0 / avg_rate
        learned_mpp = max(0.5, min(learned_mpp, 30.0))
        self.minutes_per_person = (
            self._alpha * learned_mpp + (1 - self._alpha) * self.minutes_per_person
        )
        log.info("Service rate learned: %.2f min/person", self.minutes_per_person)
        return self.minutes_per_person


# ── ROI filtering ───────────────────────────────────────────────────────────


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def filter_roi(detections: list[dict], polygon: list[list[float]]) -> list[dict]:
    filtered = []
    for d in detections:
        xmin, ymin, xmax, ymax = d["box"]
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if point_in_polygon(cx, cy, polygon):
            filtered.append(d)
    return filtered


# ── Wait estimation ─────────────────────────────────────────────────────────


def estimate_wait(count: int, minutes_per_person: float) -> float:
    return count * minutes_per_person


def queue_status(count: int, thresholds: dict) -> str:
    if count <= thresholds.get("no_line", 0):
        return "No line"
    if count <= thresholds.get("short", 3):
        return "Short wait"
    if count <= thresholds.get("moderate", 8):
        return "Moderate wait"
    return "Long wait"


# ── Annotation ──────────────────────────────────────────────────────────────


def annotate_frame(
    image: Image.Image,
    tracked: dict[int, dict],
    count: int,
    wait_min: float,
    status: str,
    roi_polygon: list[list[float]] | None = None,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_big = font

    if roi_polygon:
        pts = [(int(px * w), int(py * h)) for px, py in roi_polygon]
        draw.polygon(pts, outline=(255, 255, 0, 180))

    for oid, d in tracked.items():
        xmin, ymin, xmax, ymax = d["box"]
        x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        label = f"#{oid} {d['score']:.0%}"
        draw.rectangle([x0, y0 - 18, x0 + len(label) * 9, y0], fill=(0, 0, 0, 180))
        draw.text((x0 + 2, y0 - 17), label, fill=(0, 255, 0), font=font)

    status_color = {
        "No line": (0, 200, 0), "Short wait": (200, 200, 0),
        "Moderate wait": (255, 140, 0), "Long wait": (255, 40, 40),
    }.get(status, (255, 255, 255))

    panel_h = 80
    draw.rectangle([0, h - panel_h, w, h], fill=(0, 0, 0, 200))
    draw.text((10, h - panel_h + 8), f"People: {count}", fill=(255, 255, 255), font=font_big)
    draw.text((10, h - panel_h + 36), f"Est. wait: {wait_min:.0f} min", fill=(255, 255, 255), font=font_big)
    draw.text((w // 2, h - panel_h + 18), status, fill=status_color, font=font_big)
    draw.text((10, h - panel_h + 62), datetime.now().strftime("%H:%M:%S"), fill=(180, 180, 180), font=font)

    return img


# ── CSV Logging ─────────────────────────────────────────────────────────────


class CSVLogger:
    def __init__(self, csv_path: str, interval: float):
        self.path = os.path.expanduser(csv_path)
        self.interval = interval
        self._last_write = 0.0
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "person_count", "est_wait_min", "status",
                    "minutes_per_person", "total_tracked",
                ])

    def maybe_log(self, count: int, wait_min: float, status: str,
                  mpp: float = 0.0, total_tracked: int = 0) -> None:
        now = time.time()
        if now - self._last_write < self.interval:
            return
        self._last_write = now
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                count, f"{wait_min:.1f}", status,
                f"{mpp:.2f}", total_tracked,
            ])


# ── Snapshot Uploader ───────────────────────────────────────────────────────


class SnapshotUploader:
    """Upload annotated frames to DigitalOcean Spaces on an interval."""

    def __init__(self, cfg: dict):
        self._interval = cfg.get("interval_seconds", 300)
        self._bucket = cfg["spaces_bucket"]
        self._region = cfg.get("spaces_region", "nyc3")
        self._endpoint = cfg.get("spaces_endpoint", f"https://{self._region}.digitaloceanspaces.com")
        self._last_upload = 0.0
        self._last_url: str = ""
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "s3",
                region_name=self._region,
                endpoint_url=self._endpoint,
                aws_access_key_id=os.environ.get("DO_SPACES_ACCESS_KEY", ""),
                aws_secret_access_key=os.environ.get("DO_SPACES_SECRET_KEY", ""),
            )
        return self._client

    def maybe_upload(self, frame_jpeg: bytes) -> str | None:
        now = time.time()
        if now - self._last_upload < self._interval:
            return None
        self._last_upload = now
        try:
            ts = datetime.now()
            key = f"snapshots/{ts.strftime('%Y-%m-%d')}/{ts.strftime('%H-%M-%S')}.jpg"
            self._get_client().put_object(
                Bucket=self._bucket, Key=key, Body=frame_jpeg,
                ContentType="image/jpeg", ACL="public-read",
            )
            url = f"{self._endpoint}/{self._bucket}/{key}"
            self._last_url = url
            log.info("Snapshot uploaded: %s", url)
            return url
        except Exception as e:
            log.warning("Snapshot upload failed: %s", e)
            return None

    @property
    def last_url(self) -> str:
        return self._last_url


# ── Cloud Data Pusher ───────────────────────────────────────────────────────


class CloudPusher:
    """Push stats to cloud dashboard API in a background thread."""

    def __init__(self, cfg: dict):
        self._endpoint = cfg["endpoint"]
        self._api_key = cfg.get("api_key", "")
        self._interval = cfg.get("push_interval_seconds", 30)
        self._last_push = 0.0
        self._lock = threading.Lock()

    def maybe_push(self, data: dict) -> None:
        now = time.time()
        if now - self._last_push < self._interval:
            return
        self._last_push = now
        t = threading.Thread(target=self._do_push, args=(data.copy(),), daemon=True)
        t.start()

    def _do_push(self, data: dict) -> None:
        import urllib.request
        import urllib.error
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            self._endpoint, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status < 300:
                        return
            except (urllib.error.URLError, OSError) as e:
                log.warning("Cloud push attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 ** attempt)


# ── MJPEG HTTP stream ──────────────────────────────────────────────────────

latest_frame: bytes = b""
frame_lock = threading.Lock()
current_stats: dict = {"count": 0, "wait": 0.0, "status": "No line"}


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._serve_dashboard()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/api/status":
            self._serve_json()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PieNet Line Monitor</title>
<style>
  *{margin:0;box-sizing:border-box}
  body{background:#111;color:#eee;font-family:system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:8px}
  h1{padding:8px;font-size:1.2em;color:#8f8}
  img{max-width:100%;max-height:70vh;border:2px solid #333;border-radius:6px}
  #stats{padding:10px;font-size:1em;display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
  .stat{text-align:center;min-width:80px}
  .stat .val{font-size:1.6em;font-weight:bold}
</style>
</head><body>
  <h1>PieNet Line Monitor</h1>
  <img src="/stream">
  <div id="stats">
    <div class="stat"><div class="val" id="count">-</div>People</div>
    <div class="stat"><div class="val" id="wait">-</div>Est. wait</div>
    <div class="stat"><div class="val" id="status">-</div>Status</div>
    <div class="stat"><div class="val" id="mpp">-</div>Min/person</div>
  </div>
  <script>
    setInterval(async()=>{
      try{
        const r=await fetch('/api/status');
        const d=await r.json();
        document.getElementById('count').textContent=d.count;
        document.getElementById('wait').textContent=d.wait.toFixed(0)+' min';
        document.getElementById('status').textContent=d.status;
        if(d.mpp)document.getElementById('mpp').textContent=d.mpp.toFixed(1);
      }catch(e){}
    },2000);
  </script>
</body></html>"""
        self.wfile.write(html.encode())

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_frame
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_json(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(current_stats).encode())

    def log_message(self, format, *args):
        pass


def start_server(port: int):
    server = HTTPServer(("0.0.0.0", port), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ── Main loop ───────────────────────────────────────────────────────────────


def main() -> None:
    global latest_frame, current_stats

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="PieNet — restaurant line monitor")
    p.add_argument("--config", default=None, help="Path to config.json")
    p.add_argument("--model-dir", default=os.path.expanduser("~/.cache/pienet_models"))
    args = p.parse_args()

    cfg = load_config(args.config)
    cam = cfg["camera"]
    det = cfg["detection"]
    trk_cfg = cfg["tracking"]
    roi_cfg = cfg["roi"]
    wait_cfg = cfg["wait_estimation"]
    stream_cfg = cfg["stream"]
    log_cfg = cfg["logging"]
    snap_cfg = cfg["snapshots"]
    cloud_cfg = cfg["cloud"]

    edgetpu_lib = os.environ.get("EDGETPU_DELEGATE", "libedgetpu.so.1")

    model_path = os.path.join(args.model_dir, "ssd_mobilenet_v2_coco_edgetpu.tflite")
    labels_path = os.path.join(args.model_dir, "coco_labels.txt")
    ensure_file(MODEL_URL, model_path)
    ensure_file(LABELS_URL, labels_path)

    labels = read_labels(labels_path)
    detector = PersonDetector(model_path, edgetpu_lib)
    print(f"Model loaded. Person class: {labels.get(det['person_class_id'], 'person')}")

    tracker = CentroidTracker(max_disappeared=trk_cfg["max_disappeared"])

    rate_learner = None
    if wait_cfg.get("auto_learn", False):
        rate_learner = ServiceRateLearner(
            default_mpp=wait_cfg["minutes_per_person"],
            learning_interval=wait_cfg.get("learning_interval_seconds", 300),
        )

    csv_logger = CSVLogger(log_cfg["csv_path"], log_cfg["interval_seconds"]) if log_cfg["enabled"] else None

    snap_uploader = None
    if snap_cfg.get("enabled", False) and snap_cfg.get("spaces_bucket"):
        snap_uploader = SnapshotUploader(snap_cfg)

    cloud_pusher = None
    if cloud_cfg.get("enabled", False) and cloud_cfg.get("endpoint"):
        cloud_pusher = CloudPusher(cloud_cfg)

    start_server(stream_cfg["port"])
    print(f"Dashboard: http://0.0.0.0:{stream_cfg['port']}/")
    print(f"MJPEG:     http://0.0.0.0:{stream_cfg['port']}/stream")
    print(f"API:       http://0.0.0.0:{stream_cfg['port']}/api/status")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            image = capture_frame(cam["width"], cam["height"])
            if image is None:
                time.sleep(0.5)
                continue

            detections = detector.detect(
                image, det["confidence_threshold"], det["person_class_id"]
            )

            if roi_cfg["enabled"]:
                detections = filter_roi(detections, roi_cfg["polygon"])

            tracked = tracker.update(detections)
            count = tracker.object_count

            mpp = wait_cfg["minutes_per_person"]
            if rate_learner:
                rate_learner.record(time.time(), count)
                rate_learner.maybe_learn()
                mpp = rate_learner.minutes_per_person

            wait_min = estimate_wait(count, mpp)
            status = queue_status(count, wait_cfg["status_thresholds"])

            current_stats = {
                "count": count, "wait": wait_min, "status": status,
                "mpp": mpp, "total_tracked": tracker.total_ids_assigned,
            }

            roi_poly = roi_cfg["polygon"] if roi_cfg["enabled"] else None
            annotated = annotate_frame(image, tracked, count, wait_min, status, roi_poly)

            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=80)
            frame_bytes = buf.getvalue()
            with frame_lock:
                latest_frame = frame_bytes

            snap_url = None
            if snap_uploader:
                snap_url = snap_uploader.maybe_upload(frame_bytes)

            if cloud_pusher:
                push_data = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "count": count,
                    "wait_min": round(wait_min, 1),
                    "status": status,
                    "tracked_ids": tracker.total_ids_assigned,
                    "service_rate": round(mpp, 2),
                    "snapshot_url": snap_url or (snap_uploader.last_url if snap_uploader else ""),
                }
                cloud_pusher.maybe_push(push_data)

            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] People: {count}  Wait: {wait_min:.0f} min  "
                f"Status: {status}  MPP: {mpp:.1f}",
                flush=True,
            )

            if csv_logger:
                csv_logger.maybe_log(count, wait_min, status, mpp, tracker.total_ids_assigned)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
