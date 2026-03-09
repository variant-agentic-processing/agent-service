"""Entry point for `poe login` — logs into the GCS Pulumi backend."""
import subprocess
import sys
from pathlib import Path


def _load_env() -> dict[str, str]:
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print(f"ERROR: .env not found at {env_file}")
        print("       Copy .env.example to .env and fill in your values.")
        sys.exit(1)
    values: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def main() -> None:
    env = _load_env()
    project = env.get("GCP_PROJECT")
    if not project:
        print("ERROR: GCP_PROJECT is not set in .env")
        sys.exit(1)
    bucket = f"gs://vap-pulumi-state-{project}"
    print(f"==> Logging into {bucket}")
    result = subprocess.run(["pulumi", "login", bucket], check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
