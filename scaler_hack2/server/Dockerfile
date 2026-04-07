FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY smart_radio_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full package
COPY smart_radio_env/ /app/smart_radio_env/
COPY pyproject.toml /app/pyproject.toml

# Install the package itself
RUN pip install --no-cache-dir -e .

# Expose the FastAPI port
EXPOSE 8000

# TASK_ID controls which of the 3 tasks the server runs (1, 2, or 3)
ENV TASK_ID=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the server
CMD uvicorn smart_radio_env.server.app:app --host 0.0.0.0 --port ${PORT}
