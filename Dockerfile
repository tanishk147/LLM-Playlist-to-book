# Pinned to bookworm-based slim image. Bump the tag deliberately when
# upgrading Python or Debian base, since both affect reproducibility.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    pandoc \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-recommended \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the project metadata first so pip layer caches across code edits.
COPY pyproject.toml ./
# Setuptools needs at least a stub src/ to resolve the package set during
# `pip install -e .`. We populate the full tree in the next COPY step.
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir -e .

COPY . .

CMD ["make", "book"]
