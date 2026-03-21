# ── GitLab Handbook AI — Dockerfile ─────────────────────────────────────────
# Multi-stage: build index → runtime server
# Build: docker build -t gitlab-handbook-ai .
# Run:   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... gitlab-handbook-ai

FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Pre-build the index at image build time (optional — can also be done at runtime)
# Comment out if you prefer to build on first request (slower cold start)
ARG BUILD_INDEX=false
RUN if [ "$BUILD_INDEX" = "true" ]; then \
      cd backend && python -c "from rag.pipeline import RAGPipeline; p=RAGPipeline(); p.build_index()"; \
    fi

WORKDIR /app/backend

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
