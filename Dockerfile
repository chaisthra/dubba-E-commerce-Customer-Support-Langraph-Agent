# Assignment 2, section 2.6.1 -- the image built and pushed to ECR (SHA-tagged,
# not latest-only, per .github/workflows/ci-cd.yml) and run as an ECS Fargate task
# behind the ALB. Runs api.py (FastAPI), not main.py (the CLI) -- main.py stays
# the local-dev entry point, unrelated to this image.
#
# python:3.11-slim, not the 3.14 used in local dev -- broader wheel availability
# for torch/sentence-transformers (used by rag/retriever.py) on a still-new Python
# minor version isn't worth the risk for a deployment image; nothing in this
# codebase depends on a 3.14-only feature.
FROM python:3.11-slim

WORKDIR /app

# psycopg[binary] already bundles libpq -- no build-essential/libpq-dev needed.
# curl only for illustrating the container's own healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Not consulted by the ALB target group (that's configured separately in the ECS
# task definition / target group health check path) -- this is Docker/ECS's own
# container-level healthcheck, a second, independent signal.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
