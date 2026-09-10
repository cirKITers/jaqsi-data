# Dependencies only.  docker-compose.yml bind-mounts the repository at /app, so
# the source comes from the checkout on the executing machine and is reached
# through PYTHONPATH.  The venv lives at /opt/venv, which that mount would hide.
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

# git fetches the jaqsi and dynamiqs commits the lock pins, and the runner reads
# it to tag each results file with the commit it ran from.  libgomp is the
# OpenMP runtime qulacs and qiskit-aer link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

# Match the host user that owns the checkout, so what a run writes into the
# mount does not come back owned by root.  Values come from .env.
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g ${GID} bench \
    && useradd -u ${UID} -g ${GID} -m -d /home/bench bench

WORKDIR /app

# --no-install-project is what keeps the source out of the image; the dev group
# comes along so `docker compose run tests` has pytest.  Cleaning the cache in
# the same layer keeps the downloaded wheels out of it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=8
COPY pyproject.toml uv.lock ./
RUN ulimit -n 65536 2>/dev/null || true; \
    uv sync --frozen --no-install-project && uv cache clean

# PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the mounted checkout.
ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/bench \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/home/bench/.cache/matplotlib \
    JAX_PLATFORMS=cpu

USER bench

ENTRYPOINT ["python", "-m", "benchmark"]
