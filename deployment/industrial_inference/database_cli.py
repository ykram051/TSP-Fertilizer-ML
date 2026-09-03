from __future__ import annotations

import argparse
import json

from .configuration import load_configuration
from .storage import PredictionRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or export inference SQLite outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary")
    latest = subparsers.add_parser("latest")
    latest.add_argument("--limit", type=int, default=10)
    export = subparsers.add_parser("export")
    export.add_argument("--output", default="/runtime/predictions_export.csv")
    arguments = parser.parse_args()

    config, _ = load_configuration()
    repository = PredictionRepository(
        config["outputs"]["sqlite_database"], config["outputs"]["sqlite_schema"]
    )
    if arguments.command == "summary":
        print(json.dumps(repository.summary(), indent=2))
    elif arguments.command == "latest":
        print(json.dumps(repository.latest(arguments.limit), indent=2))
    else:
        count = repository.export_csv(arguments.output)
        print(f"Exported {count} prediction rows to {arguments.output}")


if __name__ == "__main__":
    main()
