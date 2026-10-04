from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    """Return 0 when the service published a fresh, non-failed health file."""
    path = Path(os.getenv("HEALTH_FILE", "/runtime/health.json"))
    try:
        maximum_age = float(os.getenv("HEALTH_MAX_AGE_SECONDS", "120"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated_at = datetime.fromisoformat(payload["updated_at"])
    except (OSError, ValueError, KeyError, TypeError):
        # Missing, unreadable or malformed health data means "not healthy".
        return 1
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - updated_at).total_seconds() > maximum_age:
        return 1
    if payload.get("status") in {"DISCONNECTED", "DATA_ERROR"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
