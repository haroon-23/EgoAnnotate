FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Enforce read-only permissions on gates/ inside the container (Rule R1)
RUN chmod -R a-w gates/

CMD ["python", "gates/g12_tests_hygiene.py"]
