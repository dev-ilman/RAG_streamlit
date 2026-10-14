#!/usr/bin/env bash
# Provision the whole stack on Azure, using services that fit the free tier:
#
#   * Azure Database for PostgreSQL Flexible Server - B1ms burstable, free for
#     12 months on a new subscription. pgvector ships with the service but must
#     be allow-listed via the azure.extensions server parameter.
#   * Azure Container Apps - consumption plan with a monthly free grant, and
#     scale-to-zero so an idle demo costs nothing.
#
# Idempotent: re-running updates in place rather than failing.
#
#   ./provision.sh            # uses the defaults below
#   LOCATION=westeurope ./provision.sh
set -euo pipefail

RG="${RESOURCE_GROUP:-rag-platform-rg}"
LOCATION="${LOCATION:-centralindia}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-rag-platform-env}"
PG_SERVER="${PG_SERVER:-rag-platform-pg-$RANDOM}"
PG_ADMIN="${PG_ADMIN:-ragadmin}"
PG_DB="${PG_DB:-rag}"
BACKEND_APP="${BACKEND_APP:-rag-backend}"
FRONTEND_APP="${FRONTEND_APP:-rag-frontend}"
IMAGE_PREFIX="${IMAGE_PREFIX:?set IMAGE_PREFIX, e.g. ghcr.io/you/rag-platform}"

: "${PG_PASSWORD:?set PG_PASSWORD to a strong password}"
: "${SECRET_KEY:?set SECRET_KEY, e.g. openssl rand -hex 32}"
: "${GOOGLE_API_KEY:?set GOOGLE_API_KEY (or export LLM_PROVIDER=fake)}"

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

say "Resource group ${RG}"
az group create --name "$RG" --location "$LOCATION" --output none

say "PostgreSQL flexible server ${PG_SERVER}"
az postgres flexible-server create \
  --resource-group "$RG" \
  --name "$PG_SERVER" \
  --location "$LOCATION" \
  --admin-user "$PG_ADMIN" \
  --admin-password "$PG_PASSWORD" \
  --database-name "$PG_DB" \
  --tier Burstable \
  --sku-name Standard_B1ms \
  --storage-size 32 \
  --version 16 \
  --public-access 0.0.0.0 \
  --yes --output none 2>/dev/null || echo "server already exists, continuing"

say "Allow-listing the vector extension"
az postgres flexible-server parameter set \
  --resource-group "$RG" --server-name "$PG_SERVER" \
  --name azure.extensions --value vector --output none

say "Allowing Azure services to reach the database"
az postgres flexible-server firewall-rule create \
  --resource-group "$RG" --name "$PG_SERVER" \
  --rule-name allow-azure-services \
  --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0 --output none 2>/dev/null || true

PG_HOST="${PG_SERVER}.postgres.database.azure.com"
DATABASE_URL="postgresql+asyncpg://${PG_ADMIN}:${PG_PASSWORD}@${PG_HOST}:5432/${PG_DB}?ssl=require"

say "Container Apps environment ${ENVIRONMENT_NAME}"
az extension add --name containerapp --upgrade --only-show-errors --output none
az containerapp env create \
  --name "$ENVIRONMENT_NAME" --resource-group "$RG" --location "$LOCATION" \
  --output none 2>/dev/null || echo "environment already exists, continuing"

say "Backend container app ${BACKEND_APP}"
# Migrations run as part of the container command so every revision reaches a
# schema it understands before serving traffic.
az containerapp create \
  --name "$BACKEND_APP" \
  --resource-group "$RG" \
  --environment "$ENVIRONMENT_NAME" \
  --image "${IMAGE_PREFIX}-backend:latest" \
  --target-port 8000 \
  --ingress internal \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 0.5 --memory 1.0Gi \
  --secrets "db-url=${DATABASE_URL}" "secret-key=${SECRET_KEY}" "google-key=${GOOGLE_API_KEY}" \
  --env-vars \
    "DATABASE_URL=secretref:db-url" \
    "SECRET_KEY=secretref:secret-key" \
    "GOOGLE_API_KEY=secretref:google-key" \
    "ENVIRONMENT=production" \
    "LLM_PROVIDER=${LLM_PROVIDER:-gemini}" \
    "COOKIE_SECURE=true" \
    "LOG_LEVEL=INFO" \
    "STORAGE_DIR=/app/data/uploads" \
  --command "/bin/sh" \
  --args "-c","alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2" \
  --output none 2>/dev/null || {
    say "Backend app exists - updating instead"
    az containerapp update --name "$BACKEND_APP" --resource-group "$RG" \
      --image "${IMAGE_PREFIX}-backend:latest" --output none
  }

say "Frontend container app ${FRONTEND_APP}"
az containerapp create \
  --name "$FRONTEND_APP" \
  --resource-group "$RG" \
  --environment "$ENVIRONMENT_NAME" \
  --image "${IMAGE_PREFIX}-frontend:latest" \
  --target-port 8080 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 0.25 --memory 0.5Gi \
  --output none 2>/dev/null || {
    say "Frontend app exists - updating instead"
    az containerapp update --name "$FRONTEND_APP" --resource-group "$RG" \
      --image "${IMAGE_PREFIX}-frontend:latest" --output none
  }

FQDN="$(az containerapp show --name "$FRONTEND_APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)"

cat <<SUMMARY

------------------------------------------------------------------
Deployed.

  App          https://${FQDN}
  API docs     https://${FQDN}/docs
  Health       https://${FQDN}/api/v1/health/ready
  Database     ${PG_HOST}

The frontend image proxies /api to "backend:8000". Inside a Container Apps
environment that name resolves to the backend app, so set the backend app name
to "backend" or edit frontend/nginx.conf before building the image.

Add these to the GitHub repository so the Deploy workflow can roll out updates:

  gh variable set AZURE_RESOURCE_GROUP --body "${RG}"
  gh variable set AZURE_BACKEND_APP    --body "${BACKEND_APP}"
  gh variable set AZURE_FRONTEND_APP   --body "${FRONTEND_APP}"

Tear everything down with:

  az group delete --name ${RG} --yes --no-wait
------------------------------------------------------------------
SUMMARY
