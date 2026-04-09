#!/usr/bin/env python3
"""
Live camera -> Edge TPU classification with browser preview.

Uses rpicam-still (subprocess) for capture — no picamera2 Python binding needed.
Serves an MJPEG stream at http://<pi-ip>:8080 so you can watch from any browser.

  source ~/coral-tpu-venv/bin/activate
  python3 coral_camera_classify.py
  # then open http://192.168.0.7:8080 in your browser
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tflite_runtime.interpreter as tflite

DEFAULT_MODEL = (
    "https://github.com/google-coral/test_data/raw/master/"
    "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
)
DEFAULT_LABELS = (
    "https://github.com/google-coral/test_data/raw/master/"
    "inat_bird_labels.txt"
)

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480


def ensure_file(url: str, path: str) -> None:
    if os.path.isfile(path):
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_label_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def capture_frame() -> Image.Image | None:
    """Grab one JPEG frame via rpicam-still and return as PIL Image."""
    try:
        proc = subprocess.run(
            [
                "rpicam-still",
                "--immediate",
                "--nopreview",
                "--output", "-",
                "--width", str(CAPTURE_WIDTH),
                "--height", str(CAPTURE_HEIGHT),
                "--encoding", "jpg",
                "-t", "1",
            ],
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except Exception as e:
        print(f"capture error: {e}")
        return None


def classify_image(interpreter, image: Image.Image, input_mean: float, input_std: float):
    inp = interpreter.get_input_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    resized = image.resize((w, h), Image.LANCZOS)

    params = inp.get("quantization_parameters") or {}
    scales = params.get("scales")
    zps = params.get("zero_points")
    if scales is not None and len(scales):
        scale = float(scales[0])
        zero_point = int(zps[0]) if zps is not None and len(zps) else 0
    else:
        scale, zero_point = 1.0, 0

    arr = np.asarray(resized, dtype=np.float32)
    if abs(scale * input_std - 1) < 1e-5 and abs(input_mean - zero_point) < 1e-5:
        input_data = np.asarray(resized, dtype=np.uint8)
    else:
        normalized = (arr - input_mean) / (input_std * scale) + zero_point
        np.clip(normalized, 0, 255, out=normalized)
        input_data = normalized.astype(np.uint8)

    interpreter.set_tensor(inp["index"], input_data[np.newaxis])
    interpreter.invoke()

    out = interpreter.get_output_details()[0]
    raw = np.asarray(interpreter.get_tensor(out["index"]), dtype=np.float32).reshape(-1)
    return raw


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, scores.size)
    if k <= 0:
        return []
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


def annotate(image: Image.Image, results: list[tuple[str, float]]) -> Image.Image:
    """Burn classification labels onto the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = 10
    for label, score in results:
        text = f"{label}: {score:.2f}"
        draw.rectangle([8, y - 2, 8 + len(text) * 11, y + 22], fill=(0, 0, 0, 180))
        draw.text((10, y), text, fill=(0, 255, 0), font=font)
        y += 28
    return img


# --- MJPEG HTTP server ---

latest_frame: bytes = b""
frame_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#111;display:flex;"
                b"justify-content:center;align-items:center;height:100vh'>"
                b"<img src='/stream' style='max-width:100%;max-height:100vh'>"
                b"</body></html>"
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def start_server(port: int):
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def main() -> None:
    global latest_frame

    p = argparse.ArgumentParser(description="Coral + Pi camera classification with live preview")
    p.add_argument("--model-dir", default=os.path.expanduser("~/.cache/coral_test_models"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--input-mean", type=float, default=128.0)
    p.add_argument("--input-std", type=float, default=128.0)
    p.add_argument("--port", type=int, default=8080, help="HTTP port for MJPEG stream")
    args = p.parse_args()

    edgetpu_lib = os.environ.get("EDGETPU_DELEGATE", "libedgetpu.so.1")

    model_path = os.path.join(args.model_dir, "model_edgetpu.tflite")
    labels_path = os.path.join(args.model_dir, "labels.txt")
    ensure_file(DEFAULT_MODEL, model_path)
    ensure_file(DEFAULT_LABELS, labels_path)
    labels = read_label_file(labels_path)

    delegate = tflite.load_delegate(edgetpu_lib)
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[delegate],
    )
    interpreter.allocate_tensors()
    print("Model loaded on Edge TPU.")

    start_server(args.port)
    print(f"MJPEG stream at http://0.0.0.0:{args.port}/")
    print("Open that URL in your browser. Ctrl+C to stop.")

    try:
        while True:
            image = capture_frame()
            if image is None:
                time.sleep(0.5)
                continue

            scores = classify_image(interpreter, image, args.input_mean, args.input_std)
            ranked = top_k(scores, args.top_k)
            results = [
                (labels[i] if i < len(labels) else str(i), float(s))
                for i, s in ranked
            ]

            line = " | ".join(f"{lbl} ({sc:.2f})" for lbl, sc in results)
            print(line, flush=True)

            annotated = annotate(image, results)
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=80)
            with frame_lock:
                latest_frame = buf.getvalue()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
