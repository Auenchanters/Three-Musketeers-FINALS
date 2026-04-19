FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# HF Spaces best practice: run as non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

USER user

# Expose port for HF Spaces
EXPOSE 7860

# Health check for automated validator (using Python since slim image lacks curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7860/health'); assert r.status_code == 200" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
