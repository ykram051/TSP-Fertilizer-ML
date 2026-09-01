from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from asyncua import Client

from .buffer import RollingProcessBuffer
from .configuration import load_configuration
from .models import ModelRuntime
from .quality import DataQualityShield


class SequenceHandler:
    def __init__(self, queue: asyncio.Queue[int]):
        self.queue = queue

    def datachange_notification(self, node, value, data) -> None:
        try:
            self.queue.put_nowait(int(value))
        except (ValueError, TypeError, asyncio.QueueFull):
            pass


class InferenceService:
    def __init__(self, config: dict[str, Any], tags: dict[str, Any]):
        self.config, self.tags = config, tags
        self.runtime = ModelRuntime(os.getenv("ARTIFACT_ROOT", "/artifacts"))
        service = config["service"]
        self.buffer = RollingProcessBuffer(
            maximum_rows=service["maximum_buffer_rows"],
            expected_cadence_seconds=service["cadence_seconds"],
            tolerance_seconds=service["cadence_tolerance_seconds"],
        )
        self.shield = DataQualityShield(config, tags)
        self.queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1000)
        self.health_path = Path(config["outputs"]["health_file"])
        self.prediction_path = Path(config["outputs"]["jsonl_predictions"])
        self.latest_prediction_path = Path(
            config["outputs"].get("latest_prediction_file", "/runtime/latest_prediction.json")
        )
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        # The history file exists from startup, even while no prediction is possible.
        self.prediction_path.touch(exist_ok=True)
        self.last_processed_sequence: int | None = None

    def health(self, status: str, **details: Any) -> None:
        payload = {
            "service": self.config["service"]["name"],
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "buffer_rows": len(self.buffer),
            **details,
        }
        temporary = self.health_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.health_path)

    def append_prediction(self, prediction: dict[str, Any]) -> None:
        with self.prediction_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(prediction, allow_nan=False) + "\n")
        temporary = self.latest_prediction_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(prediction, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.latest_prediction_path)

    async def read_snapshot(
        self, client: Client
    ) -> tuple[int, dict[str, Any], dict[str, bool]]:
        """Read one coherent committed OPC snapshot.

        The simulator writes the sequence tag last. Reading it before and after all
        values implements a small seqlock: a snapshot is accepted only if no new CSV
        row was committed while its values were being read.
        """
        sequence_node = client.get_node(self.tags["system"]["sequence"]["node_id"])
        source_node = client.get_node(self.tags["system"]["source_timestamp"]["node_id"])
        names = list(self.tags["inputs"])
        nodes = [client.get_node(self.tags["inputs"][name]["node_id"]) for name in names]

        for _ in range(10):
            sequence_before = int(await sequence_node.read_value())
            values = await asyncio.gather(
                source_node.read_value(), *(node.read_data_value() for node in nodes)
            )
            sequence_after = int(await sequence_node.read_value())
            if sequence_before != sequence_after:
                await asyncio.sleep(0)
                continue

            source_time = pd.Timestamp(values[0])
            if pd.isna(source_time):
                raise ValueError("OPC source timestamp is missing or invalid")
            snapshot: dict[str, Any] = {"Date": source_time}
            qualities: dict[str, bool] = {}
            for name, data_value in zip(names, values[1:]):
                snapshot[name] = data_value.Value.Value
                qualities[name] = bool(data_value.StatusCode.is_good())
            return sequence_after, snapshot, qualities
        raise RuntimeError("OPC values changed too quickly to capture one coherent snapshot")

    async def write_advisory(self, client: Client, prediction: dict[str, Any]) -> None:
        if not self.config["outputs"]["advisory_write_enabled"]:
            return
        mapping = self.tags["advisory_outputs"]
        values = {
            "prediction": float(prediction["predicted_slurry_free_acid"]),
            "predicted_change": float(prediction["predicted_change"] or 0.0),
            "generated_at": prediction["generated_at"],
            "prediction_for": prediction["prediction_for"],
            "model_status": prediction["model_status"],
            "process_familiarity": prediction["process_familiarity"],
            "input_quality": prediction["input_quality"],
            "model_version": prediction["model_version"],
        }
        for name, value in values.items():
            await client.get_node(mapping[name]["node_id"]).write_value(value)

    async def process_sequences(self, client: Client) -> None:
        while True:
            notified_sequence = await self.queue.get()
            sequence = notified_sequence
            try:
                sequence, snapshot, qualities = await self.read_snapshot(client)
                if self.last_processed_sequence is not None and sequence <= self.last_processed_sequence:
                    continue
                self.last_processed_sequence = sequence
                decision = self.shield.evaluate(snapshot, qualities)
                if not decision.accepted:
                    self.health("DATA_ERROR", sequence=sequence, reasons=decision.reasons)
                    continue
                try:
                    self.buffer.append(snapshot)
                except ValueError as error:
                    self.health("DATA_ERROR", sequence=sequence, reasons=[str(error)])
                    continue
                if not self.buffer.ready(self.runtime.minimum_rows):
                    remaining = self.runtime.minimum_rows - len(self.buffer)
                    self.health(
                        "WARMING_UP", sequence=sequence,
                        rows_required=self.runtime.minimum_rows,
                        rows_remaining=remaining,
                        message=(
                            f"Collecting process history: {len(self.buffer)}/"
                            f"{self.runtime.minimum_rows} rows; {remaining} remaining."
                        ),
                    )
                    continue

                mode = self.config["service"]["mode"]
                frame = self.buffer.frame()
                if mode == "target_anchored":
                    laboratory = self.shield.laboratory_target_is_fresh(
                        snapshot, pd.Timestamp(snapshot["Date"])
                    )
                    if not laboratory.accepted:
                        self.health("DATA_ERROR", sequence=sequence, reasons=laboratory.reasons)
                        continue
                    prediction = self.runtime.predict_target_anchored(
                        frame, self.config["service"]["forecast_horizon_minutes"]
                    )
                else:
                    prediction = self.runtime.predict_target_free(frame)

                prediction.update({
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model_status": "RUNNING",
                    "input_quality": "GOOD",
                    "sequence": sequence,
                })
                self.append_prediction(prediction)
                await self.write_advisory(client, prediction)
                self.health(
                    "RUNNING", sequence=sequence,
                    last_prediction_for=prediction["prediction_for"],
                    process_familiarity=prediction["process_familiarity"],
                )
            except Exception as error:  # keep the subscriber alive; expose failure in health/log
                logging.exception("Snapshot processing failed")
                self.health("DATA_ERROR", sequence=sequence, reasons=[repr(error)])
            finally:
                self.queue.task_done()

    async def run_connected(self) -> None:
        endpoint = self.config["opc"]["endpoint"]
        async with Client(url=endpoint) as client:
            # This laboratory reconnect starts a fresh consistency window.
            self.buffer.clear()
            self.last_processed_sequence = None
            self.health("CONNECTED", endpoint=endpoint)
            sequence = client.get_node(self.tags["system"]["sequence"]["node_id"])
            subscription = await client.create_subscription(
                self.config["opc"]["subscription_period_ms"], SequenceHandler(self.queue)
            )
            await subscription.subscribe_data_change(sequence)
            await self.process_sequences(client)

    async def run(self) -> None:
        delay = self.config["opc"]["reconnect_delay_seconds"]
        while True:
            try:
                await self.run_connected()
            except Exception as error:
                logging.exception("OPC connection failed")
                self.health("DISCONNECTED", reasons=[repr(error)])
                await asyncio.sleep(delay)


def main() -> None:
    config, tags = load_configuration()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # asyncua INFO output contains full protocol packets and obscures service status.
    logging.getLogger("asyncua").setLevel(logging.WARNING)
    service = InferenceService(config, tags)
    service.health("STARTING")
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
