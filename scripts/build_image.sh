#!/usr/bin/env bash
set -euo pipefail

PROJECT="${GCP_PROJECT:?GCP_PROJECT must be set}"
REGION="${GCP_REGION:-us-central1}"
TAG="${IMAGE_TAG:-latest}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/genomic-pipeline/agent-service:${TAG}"

echo "==> Building and pushing ${IMAGE}"
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT}" \
  .
echo "==> Done: ${IMAGE}"
