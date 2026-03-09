"""CLI script for `poe ask` — send a question to the agent and stream the answer.

Usage:
    poe ask -- --question "What pathogenic variants does HG002 have in BRCA2?"
    poe ask -- --question "..." --base-url https://agent-service-xxx-uc.a.run.app
"""

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent a question")
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AGENT_SERVICE_URL", "http://localhost:8080"),
        help="Agent service base URL",
    )
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is required. Run: pip install httpx")
        sys.exit(1)

    url = f"{args.base_url.rstrip('/')}/query"
    print(f"Asking: {args.question}\n{'─' * 60}")

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", url, json={"question": args.question}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "tool_call":
                    print(f"  [tool] {event['tool']}({json.dumps(event.get('args', {}), separators=(',', ':'))})")
                elif etype == "tool_result":
                    status = " ⚠ error" if event.get("is_error") else ""
                    print(f"  [result] {event['tool']} — {event['chars']} chars{status}")
                elif etype == "answer":
                    print(f"\n{'─' * 60}\n{event['text']}")
                elif etype == "error":
                    print(f"\n[ERROR] {event['message']}", file=sys.stderr)
                    sys.exit(1)
                elif etype == "done":
                    break


if __name__ == "__main__":
    main()
