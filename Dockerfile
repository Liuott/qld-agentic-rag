FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# libgomp is needed by torch/onnxruntime wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY serve_api.py ./

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

# The index and the model cache are mounted, not baked into the image
VOLUME ["/app/storage", "/models"]

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8008/healthz || exit 1

CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8008"]
