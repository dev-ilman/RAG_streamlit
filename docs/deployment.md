# Deployment

Two supported targets, both achievable inside a free tier. Azure is the
recommended one: managed Postgres, scale-to-zero containers, and a rollout path
already wired into GitHub Actions.

## Before you start

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # SECRET_KEY
```

Production requires:

| Variable         | Value                                                  |
| ---------------- | ------------------------------------------------------ |
| `SECRET_KEY`     | A real secret. Startup refuses the dev placeholder.     |
| `ENVIRONMENT`    | `production` — hides internal errors from API responses |
| `COOKIE_SECURE`  | `true` whenever you serve over HTTPS                    |
| `DATABASE_URL`   | Managed Postgres URL; `postgres://` is normalised to asyncpg |
| `GOOGLE_API_KEY` | Unless you run with `LLM_PROVIDER=fake`                 |

---

## Azure — Container Apps + PostgreSQL Flexible Server

**What is free.** A new subscription includes 12 months of PostgreSQL Flexible
Server B1ms with 32 GB of storage, and Container Apps has a monthly free grant
of vCPU-seconds and requests. Both apps scale to zero, so an idle demo costs
nothing.

### One-shot provisioning

```bash
export IMAGE_PREFIX=ghcr.io/<you>/rag-platform
export PG_PASSWORD='<a strong password>'
export SECRET_KEY="$(openssl rand -hex 32)"
export GOOGLE_API_KEY='<your key>'

./infra/azure/provision.sh
```

The script is idempotent — re-running updates in place. It creates the resource
group, the database, an internal-ingress backend app, and an external-ingress
frontend app, then prints the public URL.

### The two things that catch people out

**pgvector must be allow-listed.** The extension ships with the service but is
disabled until you add it to the `azure.extensions` server parameter. The
script does this; if you provision by hand:

```bash
az postgres flexible-server parameter set \
  --resource-group rag-platform-rg --server-name <server> \
  --name azure.extensions --value vector
```

`CREATE EXTENSION vector` then succeeds, which the first migration runs for you.

**The frontend proxies to the name `backend`.** `frontend/nginx.conf` sends
`/api` to `http://backend:8000`. Inside a Container Apps environment that
resolves to an app literally named `backend`, so either name the backend app
`backend` or edit `nginx.conf` before building the image.

### Continuous deployment

Register the targets once, and every green build of `main` rolls out a new
revision:

```bash
gh variable set AZURE_RESOURCE_GROUP --body rag-platform-rg
gh variable set AZURE_BACKEND_APP    --body backend
gh variable set AZURE_FRONTEND_APP   --body rag-frontend

# Federated credentials — no stored password.
gh secret set AZURE_CLIENT_ID       --body "<app registration client id>"
gh secret set AZURE_TENANT_ID       --body "<tenant id>"
gh secret set AZURE_SUBSCRIPTION_ID --body "<subscription id>"
```

The deploy job is skipped automatically when `AZURE_RESOURCE_GROUP` is unset,
so a fork still gets image builds without an Azure account.

### Migrations

The backend container runs `alembic upgrade head` before starting uvicorn, so
each revision reaches a schema it understands. With `--min-replicas 0`, the
first request after an idle period pays a cold start.

### Tear down

```bash
az group delete --name rag-platform-rg --yes --no-wait
```

---

## AWS — one free-tier EC2 instance

**What is free.** 750 hours a month of t2.micro or t3.micro for 12 months.
Postgres runs in a container alongside the app rather than on RDS, which keeps
everything inside that one instance.

```bash
aws ec2 run-instances \
  --image-id <latest Amazon Linux 2023 AMI> \
  --instance-type t3.micro \
  --key-name <your-key> \
  --security-group-ids <sg allowing 22 and 80> \
  --user-data file://infra/aws/user-data.sh
```

The user-data script installs Docker and Compose, clones the repository, writes
a `.env` with generated secrets, brings up `docker-compose.prod.yml`, and
installs a systemd unit so the stack returns after a reboot.

Edit `REPO_URL` and `GOOGLE_API_KEY` at the top of the script before launching.

**Give it swap.** t2.micro has 1 GB of RAM and no swap by default, which is not
enough for the frontend production build; the script adds a 2 GB swapfile
before building.

### Backups

Data lives in the `postgres-data` volume on the instance.

```bash
docker compose -f docker-compose.prod.yml exec -T db \
  pg_dump -U rag rag | gzip > backup-$(date +%F).sql.gz
```

---

## Hardening checklist

- [ ] `SECRET_KEY` generated, not the placeholder
- [ ] `ENVIRONMENT=production` and `COOKIE_SECURE=true`
- [ ] TLS terminated in front of the app (Container Apps does this for you)
- [ ] `CORS_ORIGINS` narrowed to your real domain
- [ ] Database not publicly reachable, or firewalled to known addresses
- [ ] `RATE_LIMIT_PER_MINUTE` set — and moved to Redis or the gateway if you
      run more than one replica, since the built-in limiter is per-process
- [ ] Scheduled backups
- [ ] Log aggregation pointed at the JSON output on stdout

## Troubleshooting

| Symptom                                | Cause and fix                                                     |
| -------------------------------------- | ----------------------------------------------------------------- |
| `type "vector" does not exist`          | The extension is not allow-listed. See the Azure note above.       |
| Documents stick in `failed`             | Read `error` on the document. Usually a missing or invalid API key, or a scanned PDF with no text layer. |
| `embedding model returned N dimensions` | `EMBEDDING_DIM` disagrees with the model. See docs/architecture.md. |
| Login works, reload signs you out       | The refresh cookie was rejected — `COOKIE_SECURE=true` over plain HTTP, or the SPA is not same-origin with the API. |
| Streamed answers arrive all at once     | A proxy is buffering. Keep `proxy_buffering off` and the `X-Accel-Buffering: no` header. |
| `SECRET_KEY must be set` on startup     | Working as intended in production. Set a real secret.              |
