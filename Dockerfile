# Fallback deployment path only (PRD §17.4, decisions.md Round 4 N4).
#
# The PRIMARY target is Streamlit Community Cloud, which needs no Dockerfile —
# it builds from `requirements.txt` + `packages.txt`. This image exists for the
# documented fallback, Render Pro (4 GB / 2 CPU), in case Community Cloud OOMs.
#
# Render Standard ($25, 2 GB) is deliberately NOT a target: it has LESS RAM than
# the free Community Cloud tier (2.7 GB), so it is strictly worse.

FROM python:3.11-slim

# PRD R17.8 — Python 3.11 exactly.
# Apt packages mirror packages.txt: every one is a subprocess this system shells
# out to. `default-jre` is required by OPSIN; without a real JRE py2opsin fails
# confusingly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        default-jre-headless \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PRD R9.8 — torch MUST come from the CPU index. The default PyPI torch pulls
# the nvidia-* CUDA packages: several GB of libraries useless on a CPU host.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps MolScribe

COPY . .

# PRD R9.9 / R17.9 — the slim checkpoint is produced at BUILD time and baked in.
# The published checkpoint is 1,134 MB, of which ~751 MB is optimizer state that
# is useless for inference; stripping it takes the image cost to 384 MB and cuts
# model load from 16.0 s to 1.5 s. A 1.13 GB cold-start download is unacceptable.
RUN python -m sarmine.cli slim-checkpoint --out /app/models/molscribe_slim.pth || \
    echo "slim-checkpoint failed at build time; the app will start without the image channel"

ENV SARMINE_MOLSCRIBE_CKPT=/app/models/molscribe_slim.pth \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OMP_NUM_THREADS=2

EXPOSE 8501

# PRD R9.10 — MolScribe's DataLoader re-executes the entry script, so every
# entry point must be import-safe and `__main__`-guarded.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
