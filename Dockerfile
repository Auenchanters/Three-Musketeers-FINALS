FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Health check for automated validator (using Python since slim image lacks curl)
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7860/health'); assert r.status_code == 200" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
