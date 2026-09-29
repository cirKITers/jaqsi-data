# Install dependencies; Compose mounts the source at /app.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

# Install Git and the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

# Match the checkout owner.
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g ${GID} bench \
    && useradd -u ${UID} -g ${GID} -m -d /home/bench bench

WORKDIR /app

# Install locked dependencies, including test tools.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=8
COPY pyproject.toml uv.lock ./
RUN ulimit -n 65536 2>/dev/null || true; \
    uv sync --frozen --no-install-project && uv cache clean

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
