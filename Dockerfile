FROM python:3.11-slim

ARG INSTALL_ESSENTIA=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MER_DATA_DIR=/workspace/data/raw \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ARCHITECTURE.md ./
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install numpy scipy librosa soundfile scikit-learn pyyaml pandas tqdm matplotlib pytest ruff && \
    if [ "$INSTALL_ESSENTIA" = "true" ]; then pip install essentia; fi

COPY . .
RUN pip install --no-deps -e .

CMD ["pytest", "-v"]
