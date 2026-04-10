#!/usr/bin/env bash
# PieNet deployment optimization — run ON the Raspberry Pi.
# Reduces power draw from ~9W to ~5-6W for unattended street deployment.
# Also installs a systemd service for auto-start on boot.
set -euo pipefail

PIENET_DIR="${HOME}/pienet"
VENV="${HOME}/coral-tpu-venv"
CONFIG="${PIENET_DIR}/config.json"
BOOT_CFG="/boot/firmware/config.txt"

echo "=== PieNet Deployment Optimizer ==="
echo ""

# ── Layer 1: Boot config (HDMI, BT, CPU freq, LEDs) ─────────────────────────

echo "==> Optimizing boot config (${BOOT_CFG})..."

BOOT_ADDITIONS="
# ── PieNet power optimizations ──
# Disable HDMI (saves ~0.5-1W)
dtoverlay=vc4-kms-v3d,nohdmi0
dtoverlay=vc4-kms-v3d,nohdmi1
# Disable Bluetooth (saves ~0.2W)
dtoverlay=disable-bt
# Cap CPU at 1.0 GHz (ML runs on Coral, CPU just captures frames)
arm_freq=1000
# Disable onboard LEDs
dtparam=act_led_trigger=none
dtparam=act_led_activelow=off
dtparam=pwr_led_trigger=none
dtparam=pwr_led_activelow=off"

if grep -q "PieNet power optimizations" "${BOOT_CFG}" 2>/dev/null; then
  echo "    Boot optimizations already applied, skipping."
else
  echo "${BOOT_ADDITIONS}" | sudo tee -a "${BOOT_CFG}" >/dev/null
  echo "    Added power optimizations to ${BOOT_CFG}"
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
echo "=== Optimization complete ==="
echo ""
echo "  Power savings applied:"
echo "    - HDMI disabled (both ports)"
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
echo "  After reboot, check power draw:"
echo "    vcgencmd pmic_read_adc"
