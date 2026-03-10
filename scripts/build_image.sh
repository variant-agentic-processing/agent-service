#!/usr/bin/env bash
set -euo pipefail

PROJECT="${GCP_PROJECT:?GCP_PROJECT must be set}"
REGION="${GCP_REGION:-us-central1}"
TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/genomic-pipeline/agent-service:${TAG}"

echo "==> Building and pushing ${IMAGE}"
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT}" \
  .

echo "==> Updating Pulumi image_tag to ${TAG}"
(cd "$(dirname "$0")/../deploy" && pulumi config set --stack dev image_tag "${TAG}")

echo "==> Done: ${IMAGE}"
