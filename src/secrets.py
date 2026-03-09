"""Secret Manager helpers — fetch secrets with env var fallback."""

import logging
import os

logger = logging.getLogger(__name__)


def get_secret(secret_id: str, env_var: str) -> str:
    """Return secret value from Secret Manager, falling back to env var."""
    value = os.environ.get(env_var)
    if value:
        return value

    try:
        from google.cloud import secretmanager

        project = os.environ["GCP_PROJECT"]
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch secret '{secret_id}' and {env_var} env var is not set: {exc}"
        ) from exc
