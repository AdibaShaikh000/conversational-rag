# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN pip install --no-cache-dir hatchling

COPY pyproject.toml .
COPY src/ src/

# Install runtime dependencies into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY app.py .
COPY resources/ resources/

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
