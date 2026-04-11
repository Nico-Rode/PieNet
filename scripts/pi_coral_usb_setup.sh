#!/usr/bin/env bash
# PieNet setup — run ON the Raspberry Pi (3B+, 4, or 5).
# Installs libedgetpu (apt), Python 3.11 venv with tflite-runtime, and downloads models.
set -euo pipefail

PI_MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown")
echo "==> Detected board: ${PI_MODEL}"

VENV="${HOME}/coral-tpu-venv"
MODEL_DIR="${HOME}/.cache/pienet_models"

# ── System packages ─────────────────────────────────────────────────────────
echo "==> apt update..."
sudo apt-get update

echo "==> Edge TPU runtime..."
DEBIAN_VERSION=$(grep VERSION_CODENAME /etc/os-release | cut -d= -f2)
if [ "${DEBIAN_VERSION}" = "trixie" ]; then
  echo "    Trixie detected — using feranick's compatible build..."
  EDGETPU_DEB="libedgetpu1-std_16.0TF2.17.1-1.trixie_arm64.deb"
  [ -f "/tmp/${EDGETPU_DEB}" ] || wget -q -O "/tmp/${EDGETPU_DEB}" \
    "https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/${EDGETPU_DEB}"
  sudo dpkg -i "/tmp/${EDGETPU_DEB}"
else
  echo "    Using standard Coral apt repo..."
  sudo install -d /usr/share/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg 2>/dev/null || true
  echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  sudo apt-get update
  sudo apt-get install -y libedgetpu1-std
fi

echo "==> rpicam-apps (camera capture)..."
sudo apt-get install -y rpicam-apps-lite || sudo apt-get install -y rpicam-apps || true

# ── Python 3.11 (needed for tflite-runtime wheels) ─────────────────────────
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "==> python3.11 not found; checking if it needs to be built from source..."
  if ! sudo apt-get install -y python3.11 python3.11-venv 2>/dev/null; then
    echo "==> Building Python 3.11 from source (Pi 5 ~15 min, Pi 3B+ ~45 min)..."
    sudo apt-get install -y build-essential libssl-dev libffi-dev libbz2-dev \
      libreadline-dev libsqlite3-dev libncursesw5-dev liblzma-dev tk-dev zlib1g-dev wget
    cd /tmp
    [ -f Python-3.11.13.tar.xz ] || wget -q https://www.python.org/ftp/python/3.11.13/Python-3.11.13.tar.xz
    [ -d Python-3.11.13 ] || tar xf Python-3.11.13.tar.xz
    cd Python-3.11.13
    ./configure --enable-optimizations --quiet
    make -j"$(nproc)"
    sudo make altinstall
    cd ~
  fi
fi
echo "==> $(python3.11 --version)"

# ── Python venv ─────────────────────────────────────────────────────────────
echo "==> Creating venv at ${VENV}..."
python3.11 -m venv "${VENV}" --system-site-packages
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
pip install --upgrade pip
pip install tflite-runtime 'numpy<2' pillow boto3 websockets

# ── udev / plugdev ──────────────────────────────────────────────────────────
sudo udevadm control --reload-rules
sudo udevadm trigger
if ! groups "${USER}" | grep -q '\bplugdev\b'; then
  echo "Adding ${USER} to plugdev — log out/in or reboot after this."
  sudo usermod -aG plugdev "${USER}"
fi

# ── Download models ─────────────────────────────────────────────────────────
echo "==> Downloading person detection model..."
mkdir -p "${MODEL_DIR}"
[ -f "${MODEL_DIR}/ssd_mobilenet_v2_coco_edgetpu.tflite" ] || \
  wget -q -O "${MODEL_DIR}/ssd_mobilenet_v2_coco_edgetpu.tflite" \
    "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
[ -f "${MODEL_DIR}/coco_labels.txt" ] || \
  wget -q -O "${MODEL_DIR}/coco_labels.txt" \
    "https://github.com/google-coral/test_data/raw/master/coco_labels.txt"
echo "==> Models in ${MODEL_DIR}"

# ── Smoke test ──────────────────────────────────────────────────────────────
echo "==> Quick Edge TPU check..."
python3 -c "
import tflite_runtime.interpreter as t
d = t.load_delegate('libedgetpu.so.1')
i = t.Interpreter(model_path='${MODEL_DIR}/ssd_mobilenet_v2_coco_edgetpu.tflite', experimental_delegates=[d])
i.allocate_tensors()
print('Edge TPU + person detection model: OK')
"

echo ""
echo "=== Setup complete ==="
echo "  source ${VENV}/bin/activate"
echo "  python3 ~/line_monitor.py --config ~/config.json"
echo "  Then open http://<pi-ip>:8080 in your browser."
