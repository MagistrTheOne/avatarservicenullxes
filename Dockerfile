# syntax=docker/dockerfile:1.7
#
# Production image for avatarservicenullxes on RunPod H200.
#
# The image comes up without weights. At pod start we expect a RunPod Network
# Volume mounted at /models containing ARACHNE-X-ULTRA-AVATAR (use
# scripts/download_weights.sh once per volume to populate it).
#
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        ffmpeg libopus0 libvpx7 libsrtp2-1 libnss3 \
        libgl1 libglib2.0-0 \
        git curl ca-certificates tini build-essential \
        libavdevice-dev libavfilter-dev libavformat-dev \
        libavcodec-dev libswresample-dev libswscale-dev libavutil-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching.
COPY requirements.txt requirements-gpu.txt ./

# Core + GPU packages (torch etc. are pulled from the PyTorch CUDA index).
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements-gpu.txt

# FlashAttention-2 requires --no-build-isolation because it compiles CUDA
# extensions against the installed torch version. Version matches upstream
# ARACHNE-X requirements.
RUN pip install ninja psutil packaging \
    && pip install flash-attn==2.7.4.post1 --no-build-isolation

# Copy source.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install -e .

# Non-root user for RunPod templates that enforce it.
RUN useradd -m -u 1000 avatar && chown -R avatar:avatar /app
USER avatar

EXPOSE 8080
ENV HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8080 \
    ARACHNE_MODE=real \
    ARACHNE_WEIGHTS_DIR=/models/ARACHNE-X-ULTRA-AVATAR

# tini so aiortc / torch threads get clean signals on pod stop.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["avatar-service", "serve"]
