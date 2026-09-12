# PostgreSQL Setup Guide — VAPT-AI v3.2

> **When to run this:** After W1-E (migration script ready), before W1-L (verify on VM).
> This guide sets up PostgreSQL 16 on Ubuntu 26.04 LTS with the `vapt_ai` database,
> `vapt` user, pgvector + pgcrypto extensions, and runs the Alembic migration.

## Prerequisites

- Ubuntu 26.04 LTS VM (already installed per user setup)
- sudo access
- Python 3.12 + venv (installed in W1-A)
- VAPT-AI repo cloned to `~/VAPT-AI/` (or wherever you placed it)

## Step 1 — Install PostgreSQL 16

```bash
# Add PostgreSQL official repo (Ubuntu's default may be older)
sudo apt install -y curl ca-certificates
sudo install -d /usr/share/postgresql-common/pgdg
sudo curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc
sudo sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Update + install
sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib-16

# Verify
psql --version
# Expected: psql (PostgreSQL) 16.x
```

## Step 2 — Install pgvector extension

```bash
# pgvector is in PostgreSQL's contrib packages on Ubuntu 26.04
sudo apt install -y postgresql-16-pgvector

# Verify the .so exists
ls /usr/share/postgresql/16/extension/vector*
# Expected: vector.control, vector--*.sql files
```

## Step 3 — Start PostgreSQL service

```bash
# Start + enable on boot
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Verify running
sudo systemctl status postgresql
# Expected: active (running)
```

## Step 4 — Create database + user

```bash
# Switch to postgres user
sudo -u postgres psql

# In the psql prompt, run these commands:
```

```sql
-- Create user vapt with password vapt@NhatVKU
CREATE USER vapt WITH PASSWORD 'vapt@NhatVKU';

-- Create database vapt_ai owned by vapt
CREATE DATABASE vapt_ai OWNER vapt;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE vapt_ai TO vapt;

-- Connect to the new database
\c vapt_ai

-- Create extensions (needed by migration)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Grant schema privileges to vapt
GRANT ALL ON SCHEMA public TO vapt;

-- Exit psql
\q
```

## Step 5 — Verify connection

```bash
# Test that vapt user can connect
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "SELECT version();"
# Expected: PostgreSQL 16.x on x86_64-pc-linux-gnu...

# Test extensions are installed
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "SELECT extname, extversion FROM pg_extension;"
# Expected: pgcrypto, vector, plpgsql (at minimum)
```

## Step 6 — Configure VAPT-AI .env

```bash
# Navigate to your VAPT-AI repo
cd ~/VAPT-AI    # or wherever you cloned it

# Copy env template
cp .env.example .env

# Edit .env — verify these lines:
# VAPT_AI_POSTGRES_DB=postgresql://vapt:vapt%40NhatVKU@localhost:5432/vapt_ai
# (note: @ is URL-encoded as %40)

# Verify the URL works by testing with python:
python3 -c "
from urllib.parse import urlparse
url = 'postgresql://vapt:vapt%40NhatVKU@localhost:5432/vapt_ai'
p = urlparse(url)
print(f'user: {p.username}')
print(f'pass: {p.password}')  # should print vapt@NhatVKU
print(f'host: {p.hostname}')
print(f'port: {p.port}')
print(f'db:   {p.path[1:]}')
"
```

Expected output:
```
user: vapt
pass: vapt@NhatVKU
host: localhost
port: 5432
db:   vapt_ai
```

## Step 7 — Install Python dependencies

```bash
cd ~/VAPT-AI

# Create venv (if not already)
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify key packages
python3 -c "import sqlalchemy, asyncpg, psycopg2, alembic, pgvector; print('DB deps OK')"
python3 -c "import fastapi, uvicorn, pydantic; print('FastAPI deps OK')"
```

## Step 8 — Run Alembic migration

```bash
cd ~/VAPT-AI
source venv/bin/activate

# Check current migration state (should be empty — fresh DB)
alembic current
# Expected: (empty — no migrations applied yet)

# Apply ALL migrations (0001 baseline + 0002 user_token_usage + 0003 vapt_ai_baseline)
alembic upgrade head

# Expected output:
# INFO  [alembic.runtime.migration] Running upgrade  -> 0001, baseline schema snapshot
# INFO  [alembic.runtime.migration] Running upgrade 0001 -> 0002, add user_token_usage + quota overrides
# INFO  [alembic.runtime.migration] Running upgrade 0002 -> 0003_vapt_ai_baseline, VAPT-AI v3.2 baseline...
```

## Step 9 — Verify tables created

```bash
# Count all tables in vapt_ai database
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "
SELECT count(*) as total_tables
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
"
# Expected: 40+ tables (legacy EVVO tables + 24 new vapt_ tables)

# List VAPT-AI tables specifically
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public' AND tablename LIKE 'vapt_%'
ORDER BY tablename;
"
# Expected: 24 rows (vapt_assets, vapt_attack_chains, vapt_audit_log, ...)

# Verify extensions
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pgcrypto');
"
# Expected: pgcrypto + vector
```

## Step 10 — Verify migration history

```bash
cd ~/VAPT-AI
source venv/bin/activate

alembic current
# Expected: 0003_vapt_ai_baseline (head)

alembic history --verbose
# Expected: 3 revisions (0001, 0002, 0003_vapt_ai_baseline)
```

## Troubleshooting

### Error: `connection refused`

PostgreSQL not running or not listening on localhost:
```bash
sudo systemctl status postgresql
sudo systemctl restart postgresql
# Check pg_hba.conf allows localhost connections
sudo cat /etc/postgresql/16/main/pg_hba.conf | grep -v "^#" | grep -v "^$"
```

### Error: `password authentication failed`

User/password mismatch. Reset password:
```bash
sudo -u postgres psql -c "ALTER USER vapt WITH PASSWORD 'vapt@NhatVKU';"
```

### Error: `permission denied for schema public`

Grant privileges:
```bash
sudo -u postgres psql -d vapt_ai -c "GRANT ALL ON SCHEMA public TO vapt;"
sudo -u postgres psql -d vapt_ai -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vapt;"
sudo -u postgres psql -d vapt_ai -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vapt;"
```

### Error: `extension "vector" does not exist`

pgvector not installed:
```bash
sudo apt install -y postgresql-16-pgvector
sudo systemctl restart postgresql
sudo -u postgres psql -d vapt_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Error: `gen_random_uuid() does not exist`

pgcrypto not installed:
```bash
sudo -u postgres psql -d vapt_ai -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
```

### Error: `alembic: command not found`

venv not activated or alembic not installed:
```bash
source venv/bin/activate
pip install alembic
# Or reinstall all deps:
pip install -r requirements.txt
```

### Error: `ModuleNotFoundError: No module named 'app'`

PYTHONPATH not set. Run from project root:
```bash
cd ~/VAPT-AI
export PYTHONPATH=$(pwd)
alembic upgrade head
```

Or set in .env:
```bash
echo "PYTHONPATH=$(pwd)" >> .env
```

## Next Steps

After PostgreSQL is set up + migration applied:
- **W1-F**: AI writes `app/auth/manager.py` (JWT auth) + `app/main.py` (FastAPI factory)
- **W1-G**: AI verifies MCP server skeleton
- **W1-H**: AI writes `tests/test_auth.py`
- **W1-K**: AI self-verifies in sandbox (uvicorn + /docs + /health + MCP server)
- **W1-L**: You verify on VM: `uvicorn app.main:app --reload` + browser → `http://localhost:8000/docs`

## Quick Reference — All Commands in Order

```bash
# 1. Install PostgreSQL
sudo apt install -y curl ca-certificates
sudo install -d /usr/share/postgresql-common/pgdg
sudo curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc
sudo sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib-16 postgresql-16-pgvector
sudo systemctl enable postgresql
sudo systemctl start postgresql

# 2. Create DB + user
sudo -u postgres psql -c "CREATE USER vapt WITH PASSWORD 'vapt@NhatVKU';"
sudo -u postgres psql -c "CREATE DATABASE vapt_ai OWNER vapt;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE vapt_ai TO vapt;"
sudo -u postgres psql -d vapt_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d vapt_ai -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
sudo -u postgres psql -d vapt_ai -c "GRANT ALL ON SCHEMA public TO vapt;"

# 3. Setup VAPT-AI
cd ~/VAPT-AI
cp .env.example .env
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run migration
alembic upgrade head

# 5. Verify
alembic current
PGPASSWORD='vapt@NhatVKU' psql -U vapt -d vapt_ai -h localhost -c "SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'vapt_%';"
```