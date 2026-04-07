FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir -e . uvicorn

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV TASK_ID=1

EXPOSE 7860

# Start the OpenEnv server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
