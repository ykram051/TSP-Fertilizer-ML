from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    path = Path(os.getenv("HEALTH_FILE", "/runtime/health.json"))
    if not path.exists():
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    age = datetime.now(timezone.utc) - datetime.fromisoformat(payload["updated_at"])
    if age.total_seconds() > 120:
        return 1
    if payload.get("status") in {"DISCONNECTED", "DATA_ERROR"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
