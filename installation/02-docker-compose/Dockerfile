# ARG lets callers swap CPU/GPU base at build time:
#   docker compose build                        → CPU image (default)
#   BASE_IMAGE=rayproject/ray:2.47.0-py311-gpu docker compose build → GPU image
ARG BASE_IMAGE=rayproject/ray:2.47.0-py311
FROM ${BASE_IMAGE}

# Extra ML/observability packages not bundled in the base Ray image
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    huggingface_hub \
    outlines \
    tensorboard \
    sentencepiece \
    protobuf \
    mlflow \
    scikit-learn \
    matplotlib \
    jupyter \
    ipython

WORKDIR /app

# Ray ports: GCS | Dashboard | Client
EXPOSE 6379 8265 10001
