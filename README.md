# PieNet — Restaurant Line Monitor

Real-time queue monitoring for NYC restaurants using a Raspberry Pi 5, Google Coral Edge TPU, and a wide-angle camera. Detects people, tracks them across frames, estimates wait times, and streams results to a cloud dashboard.

## Architecture

```
Raspberry Pi 5 + Coral USB ──► DigitalOcean
  Camera → Detection → Tracker    App Platform (FastAPI dashboard)
  → Rate Learner → Data Push ──►  + DO Spaces (snapshot storage)
  → Snapshot Upload ──────────►
  → Local MJPEG stream
```

## Repo Structure

```
scripts/
  line_monitor.py        # Pi-side: detection, tracking, streaming, push
  pi_coral_usb_setup.sh  # Pi setup: Edge TPU runtime, Python 3.11, models
  coral_camera_classify.py # (legacy) initial classification test

dashboard/
  app.py                 # Cloud FastAPI server: ingest, history, latest
  static/index.html      # Chart.js dashboard UI
  requirements.txt       # Python deps for dashboard

config.json              # Default config (no secrets)
```

## Quick Start

### Raspberry Pi

```bash
# 1. Copy files to Pi
scp -r scripts/ config.json admin@<pi-ip>:~/pienet/

# 2. SSH in and run setup
ssh admin@<pi-ip>
cd ~/pienet && bash scripts/pi_coral_usb_setup.sh

# 3. Start monitoring
source ~/coral-tpu-venv/bin/activate
python3 scripts/line_monitor.py --config config.json
# Open http://<pi-ip>:8080 in your browser
```

### Cloud Dashboard

```bash
cd dashboard
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Set environment variables: `API_KEY`, `DO_SPACES_ACCESS_KEY`, `DO_SPACES_SECRET_KEY`.

## Configuration

Edit `config.json` to adjust detection thresholds, ROI polygon, wait estimation parameters, snapshot uploads, and cloud push settings. Secrets (API keys, Spaces credentials) should be set via environment variables, never in the config file.

## License

MIT
