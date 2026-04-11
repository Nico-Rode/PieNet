#!/usr/bin/env bash
# PieNet deployment optimization — run ON the Raspberry Pi (3B+, 4, or 5).
# Reduces power draw for unattended street deployment.
# Also installs a systemd service for auto-start on boot.
set -euo pipefail

PIENET_DIR="${HOME}/pienet"
VENV="${HOME}/coral-tpu-venv"
CONFIG="${PIENET_DIR}/config.json"

PI_MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown")
echo "=== PieNet Deployment Optimizer ==="
echo "==> Detected board: ${PI_MODEL}"
echo ""

# Detect boot config location (older images use /boot, newer use /boot/firmware)
if [ -f /boot/firmware/config.txt ]; then
  BOOT_CFG="/boot/firmware/config.txt"
elif [ -f /boot/config.txt ]; then
  BOOT_CFG="/boot/config.txt"
else
  echo "WARNING: Could not find config.txt, skipping boot config."
  BOOT_CFG=""
fi

# ── Layer 1: Boot config (HDMI, BT, CPU freq, LEDs) ─────────────────────────

if [ -n "${BOOT_CFG}" ]; then
echo "==> Optimizing boot config (${BOOT_CFG})..."

if echo "${PI_MODEL}" | grep -qi "Pi 5"; then
  BOOT_ADDITIONS="
# ── PieNet power optimizations (Pi 5) ──
dtoverlay=vc4-kms-v3d,nohdmi0
dtoverlay=vc4-kms-v3d,nohdmi1
dtoverlay=disable-bt
arm_freq=1000
dtparam=act_led_trigger=none
dtparam=act_led_activelow=off
dtparam=pwr_led_trigger=none
dtparam=pwr_led_activelow=off"
else
  BOOT_ADDITIONS="
# ── PieNet power optimizations (Pi 3B+/4) ──
# Disable HDMI output
hdmi_blanking=2
# Disable Bluetooth
dtoverlay=disable-bt
# Cap CPU at 1.0 GHz (ML runs on Coral, CPU just captures frames)
arm_freq=1000
# Disable activity LED
dtparam=act_led_trigger=none
dtparam=act_led_activelow=on"
fi

if grep -q "PieNet power optimizations" "${BOOT_CFG}" 2>/dev/null; then
  echo "    Boot optimizations already applied, skipping."
else
  echo "${BOOT_ADDITIONS}" | sudo tee -a "${BOOT_CFG}" >/dev/null
  echo "    Added power optimizations to ${BOOT_CFG}"
fi
fi

# ── Layer 2: Disable desktop & unnecessary services ──────────────────────────

echo "==> Switching to headless (multi-user) mode..."
sudo systemctl set-default multi-user.target 2>/dev/null || true

echo "==> Disabling unnecessary services..."
for svc in avahi-daemon cups cups-browsed triggerhappy ModemManager; do
  if systemctl is-enabled "${svc}" 2>/dev/null | grep -q enabled; then
    sudo systemctl disable --now "${svc}" 2>/dev/null || true
    echo "    Disabled ${svc}"
  fi
done

# Disable bluetooth service (matches dtoverlay above)
sudo systemctl disable --now bluetooth 2>/dev/null || true
sudo systemctl disable --now hciuart 2>/dev/null || true
echo "    Disabled bluetooth services"

# ── Layer 3: Systemd service for auto-start ──────────────────────────────────

echo "==> Installing pienet.service..."

sudo tee /etc/systemd/system/pienet.service >/dev/null <<UNIT
[Unit]
Description=PieNet Line Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PIENET_DIR}
Environment=PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${VENV}/bin/python3 ${PIENET_DIR}/scripts/line_monitor.py --config ${CONFIG}
Restart=always
RestartSec=10
WatchdogSec=120

StandardOutput=journal
StandardError=journal
SyslogIdentifier=pienet

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable pienet.service
echo "    pienet.service installed and enabled"
echo "    Start now:  sudo systemctl start pienet"
echo "    View logs:  journalctl -u pienet -f"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Optimization complete (${PI_MODEL}) ==="
echo ""
echo "  Power savings applied:"
echo "    - HDMI disabled"
echo "    - Bluetooth disabled"
echo "    - CPU capped at 1.0 GHz"
echo "    - LEDs disabled"
echo "    - Desktop GUI disabled (headless mode)"
echo "    - Unnecessary services disabled"
echo ""
echo "  Deployment:"
echo "    - pienet.service will auto-start on boot"
echo "    - Auto-restarts on crash (10s delay)"
echo "    - Watchdog: reboots if hung for 120s"
echo ""
echo "  REBOOT REQUIRED for boot config changes:"
echo "    sudo reboot"
echo ""
if echo "${PI_MODEL}" | grep -qi "Pi 5"; then
  echo "  After reboot, check power draw:"
  echo "    vcgencmd pmic_read_adc"
else
  echo "  After reboot, check temperature:"
  echo "    vcgencmd measure_temp"
fi
