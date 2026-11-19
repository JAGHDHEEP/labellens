# syntax=docker/dockerfile:1
# Serving image. No PyTorch: embeddings and generation are remote calls and the
# reranker runs on ONNX Runtime, which keeps the image small and cold starts fast.
FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    LABELLENS_DATA_DIR=/data \
    FASTEMBED_CACHE_PATH=/models
RUN useradd --create-home --uid 10001 app && mkdir -p /data /models && chown app /data /models
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/labellens /usr/local/bin/labellens
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz').status==200 else 1)"
CMD ["labellens", "serve", "--host", "0.0.0.0", "--port", "8000", "--json-logs"]
