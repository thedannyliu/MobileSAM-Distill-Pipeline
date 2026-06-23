ARG CUDA_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
FROM ${CUDA_IMAGE}

ARG MOBILE_SAM_REPO=https://github.com/ChaoningZhang/MobileSAM.git
ARG MOBILE_SAM_COMMIT=01ea8d0f5590082f0c1ceb0a3e2272593f20154b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MOBILE_SAM_ROOT=/opt/MobileSAM

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone "${MOBILE_SAM_REPO}" "${MOBILE_SAM_ROOT}" \
  && cd "${MOBILE_SAM_ROOT}" \
  && git checkout "${MOBILE_SAM_COMMIT}" \
  && pip install -e .

WORKDIR /repo
COPY requirements /repo/requirements
COPY requirements.txt pyproject.toml README.md /repo/
RUN pip install -r requirements/train.txt

COPY . /repo
RUN pip install -e .

ENV ARTIFACT_ROOT=/artifacts \
    DATA_ROOT=/artifacts/data \
    FEATURE_ROOT=/artifacts/features \
    CHECKPOINT_ROOT=/artifacts/checkpoints \
    OUTPUT_ROOT=/artifacts/outputs \
    CACHE_ROOT=/artifacts/cache

CMD ["python", "-m", "mobilesam_distill.cli.smoke_import"]
