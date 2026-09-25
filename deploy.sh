#!/usr/bin/env bash
# ====================================================================
# DigitalOcean Droplet Automated Setup Script for MEXC Williams Scanner
# ====================================================================
set -e

echo "=== [1/5] Updating system packages ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

APP_DIR="/opt/mexc-williams-system"

echo "=== [2/5] Setting up project directory at $APP_DIR ==="
sudo mkdir -p "$APP_DIR"
# If running inside git repo clone, copy files; else setup current directory
if [ -d "./scanner" ]; then
    sudo cp -r ./* "$APP_DIR/"
fi

cd "$APP_DIR"

echo "=== [3/5] Creating Python virtual environment ==="
sudo python3 -m venv venv
sudo ./venv/bin/pip install --upgrade pip
sudo ./venv/bin/pip install -r requirements.txt

echo "=== [4/5] Checking .env configuration ==="
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    sudo cp .env.example .env
    echo "⚠️ IMPORTANT: Please edit /opt/mexc-williams-system/.env with your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID!"
fi

echo "=== [5/5] Configuring systemd service ==="
sudo cp mexc-scanner.service /etc/systemd/system/mexc-scanner.service
sudo systemctl daemon-reload
sudo systemctl enable mexc-scanner.service
sudo systemctl restart mexc-scanner.service

echo ""
echo "===================================================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "Check scanner logs anytime with:"
echo "    sudo journalctl -u mexc-scanner.service -f"
echo ""
echo "Check service status with:"
echo "    sudo systemctl status mexc-scanner.service"
echo "===================================================================="
