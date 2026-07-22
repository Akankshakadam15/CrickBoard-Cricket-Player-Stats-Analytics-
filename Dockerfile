# CrickBoard — Dockerfile
# Build:  docker build -t crickboard .
# Run:    docker run -p 8501:8501 crickboard
# Then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# curl is needed for the HEALTHCHECK below; python:slim doesn't include it
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching — only re-installs
# when requirements.txt actually changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# The SQLite database (accounts/favorites) is created at runtime inside
# the container. For data to survive container restarts, mount a volume
# over /app when running, e.g.:
#   docker run -p 8501:8501 -v crickboard_data:/app/db_data crickboard

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]