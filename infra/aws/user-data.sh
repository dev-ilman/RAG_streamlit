#!/bin/bash
# EC2 user-data for a free-tier single-host deployment (t2.micro / t3.micro,
# Amazon Linux 2023). Installs Docker, pulls the repository, and brings up the
# production compose stack with Postgres running alongside the app.
#
# Pass the repository and secrets as environment substitutions before launch,
# or edit the variables below.
set -euxo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOUR_USER/rag-platform.git}"
APP_DIR=/opt/rag-platform

dnf update -y
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

# Compose v2 as a Docker CLI plugin.
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL \
  "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone --depth 1 "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

# t2.micro has 1 GB of RAM and no swap; the frontend build step needs more.
# A 2 GB swapfile keeps the build from being OOM-killed.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

cat > .env <<ENVFILE
ENVIRONMENT=production
SECRET_KEY=$(openssl rand -hex 32)
COOKIE_SECURE=false
POSTGRES_USER=rag
POSTGRES_PASSWORD=$(openssl rand -hex 16)
POSTGRES_DB=rag
POSTGRES_HOST=db
LLM_PROVIDER=${LLM_PROVIDER:-gemini}
GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
EMBEDDING_DIM=768
STORAGE_DIR=/app/data/uploads
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
ENVFILE
chmod 600 .env

docker compose -f docker-compose.prod.yml up -d --build

# Restart the stack on reboot.
cat > /etc/systemd/system/rag-platform.service <<UNIT
[Unit]
Description=RAG Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${APP_DIR}
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable rag-platform.service
