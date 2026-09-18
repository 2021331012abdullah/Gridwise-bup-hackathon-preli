# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Non-root runtime user
RUN groupadd -r gridwise && useradd -r -g gridwise -d /app -s /sbin/nologin gridwise

WORKDIR /app

# Dependencies first, for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source only. No .env, no tests, no secrets in any layer.
COPY app ./app

RUN chown -R gridwise:gridwise /app
USER gridwise

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# /health must answer with no API key present, so the probe needs no credentials.
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
