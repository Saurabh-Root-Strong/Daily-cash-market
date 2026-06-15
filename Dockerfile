# Daily_Cash_Market — Streamlit dashboard (self-host image).
#
# Read-only analytics app over a DuckDB snapshot. The database is deliberately
# NOT baked into the image (.dockerignore excludes data/) — either:
#   • mount it as a volume:   -v /host/data:/app/data
#   • or run with CLOUD_MODE=true to pull the snapshot from GitHub Releases.
# Put Caddy (separate container) in front for TLS + auth, same topology as the
# Tradebot image. Single process — Streamlit holds in-process cache state.
FROM python:3.12-slim

# Pinned base + UTF-8 + IST. PYTHONUTF8 avoids the cp1252 encode errors seen on
# Windows; TZ keeps trade-date logic aligned with the NSE session.
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    TZ=Asia/Kolkata \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Deps first for layer caching. requirements.txt caps pandas<3 / numpy<2 so the
# build is reproducible (the bare `>=` set crashed Streamlit Cloud on rebuild).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit's built-in health endpoint — lets the orchestrator restart a hung app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "src/dashboard/app.py"]
