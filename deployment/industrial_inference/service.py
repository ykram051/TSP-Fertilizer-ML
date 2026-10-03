from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from asyncua import Client

from . import __version__
from .buffer import RollingProcessBuffer
from .configuration import load_configuration
from .models import ModelRuntime
from .quality import DataQualityShield
from .storage import PredictionRepository


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
        self.run_id: str | None = None
        self.repository = PredictionRepository(
            config["outputs"]["sqlite_database"], config["outputs"]["sqlite_schema"]
        )
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        # The history file exists from startup, even while no prediction is possible.
        self.prediction_path.touch(exist_ok=True)
        self.last_processed_sequence: int | None = None
        self.last_recorded_status: str | None = None
        self.current_status = "STARTING"
        self.current_health_details: dict[str, Any] = {}
        self.heartbeat_count = 0
        self.last_snapshot_at: str | None = None
        self.last_prediction_at: str | None = None
    def start_audit_run(self) -> None:
        """Open a new audit/timeline segment after each OPC session is established.

        Controller or gateway restarts can reset their sequence counter. A new
        run_id keeps the SQLite uniqueness contract valid and makes the dashboard
        show a clear new timeline rather than mixing two stream sessions.
        """
        service = self.config["service"]
        self.run_id = str(uuid.uuid4())
        self.last_recorded_status = None
        self.repository.start_run(
            run_id=self.run_id,
            service_name=service["name"],
            model_version=__version__,
            inference_mode=service["mode"],
            horizon_minutes=(service["forecast_horizon_minutes"]
                             if service["mode"] == "target_anchored" else None),
            started_at=datetime.now(timezone.utc).isoformat(),
            configuration={"service": service, "opc_endpoint": self.config["opc"]["endpoint"]},
        )

    def health(self, status: str, *, heartbeat: bool = False, **details: Any) -> None:
        """Atomically publish service state and retain it for periodic heartbeats."""
        if heartbeat:
            details = dict(self.current_health_details)
        else:
            self.current_status = status
            self.current_health_details = dict(details)
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "service": self.config["service"]["name"],
            "run_id": self.run_id,
            "status": status,
            "updated_at": updated_at,
            "buffer_rows": len(self.buffer),
            "buffer_reset_count": self.buffer.reset_count,
            "last_buffer_reset_reason": self.buffer.last_reset_reason,
            "last_snapshot_at": self.last_snapshot_at,
            "last_prediction_at": self.last_prediction_at,
            "heartbeat_count": self.heartbeat_count,
            "heartbeat_interval_seconds": self.config["service"]["heartbeat_interval_seconds"],
            **details,
        }
        temporary = self.health_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.health_path)
        if self.run_id is not None and not heartbeat and status != self.last_recorded_status:
            self.repository.add_event(
                self.run_id, updated_at, status, details.get("sequence"),
                details.get("message"), details,
            )
            self.last_recorded_status = status

    async def heartbeat_loop(self) -> None:
        """Publish liveness even while the stream is quiet or reconnecting."""
        interval = float(self.config["service"].get("heartbeat_interval_seconds", 5))
        while True:
            await asyncio.sleep(max(interval, 1.0))
            self.heartbeat_count += 1
            self.health(self.current_status, heartbeat=True)

    def append_prediction(self, prediction: dict[str, Any]) -> None:
        if self.run_id is None:
            raise RuntimeError("Cannot persist a prediction before an OPC audit run is started")
        prediction["run_id"] = self.run_id
        # The database is the authoritative audit record; JSON remains a readable mirror.
        self.repository.add_prediction(self.run_id, prediction)
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

    async def acknowledge_replay_row(self, client: Client, sequence: int) -> None:
        """Acknowledge a CSV row only in the laboratory replay environment.

        A real plant must never wait for an ML service before publishing process
        data.  This handshake exists solely to prevent a fast simulator from
        coalescing rows faster than the model can read them.
        """
        simulator = self.config.get("simulator", {})
        if not simulator.get("flow_control_acknowledgement_enabled", False):
            return
        node_id = self.tags.get("system", {}).get("replay_acknowledgement", {}).get("node_id")
        if not node_id:
            return
        await client.get_node(node_id).write_value(int(sequence))

    @staticmethod
    def is_connection_failure(error: Exception) -> bool:
        if isinstance(error, (ConnectionError, BrokenPipeError, ConnectionResetError,
                              asyncio.IncompleteReadError)):
            return True
        description = repr(error).lower()
        return any(token in description for token in (
            "connection is closed", "connection closed", "connection lost",
            "session closed", "not connected",
        ))

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
            "input_quality": prediction["input_quality"],
            "model_version": prediction["model_version"],
        }
        for name, value in values.items():
            await client.get_node(mapping[name]["node_id"]).write_value(value)

    async def process_sequences(self, client: Client, sequence_node) -> None:
        watchdog_seconds = float(self.config["opc"].get("sequence_watchdog_poll_seconds", 5))
        while True:
            from_subscription = False
            try:
                notified_sequence = await asyncio.wait_for(
                    self.queue.get(), timeout=max(watchdog_seconds, 1.0)
                )
                from_subscription = True
            except asyncio.TimeoutError:
                # A quiet process is normal. A sequence reset or unreadable node
                # is not: reconnect and begin a clean temporal-history window.
                observed_sequence = int(await sequence_node.read_value())
                if self.last_processed_sequence is not None and observed_sequence < self.last_processed_sequence:
                    raise ConnectionError(
                        f"OPC sequence reset detected: {observed_sequence} < "
                        f"{self.last_processed_sequence}"
                    )
                if self.last_processed_sequence is None or observed_sequence > self.last_processed_sequence:
                    notified_sequence = observed_sequence
                else:
                    continue
            sequence = notified_sequence
            acknowledgement_sequence: int | None = None
            try:
                sequence, snapshot, qualities = await self.read_snapshot(client)
                acknowledgement_sequence = sequence
                if self.last_processed_sequence is not None and sequence <= self.last_processed_sequence:
                    continue
                self.last_processed_sequence = sequence
                self.last_snapshot_at = datetime.now(timezone.utc).isoformat()
                decision = self.shield.evaluate(snapshot, qualities)
                if not decision.accepted:
                    self.health("DATA_ERROR", sequence=sequence, reasons=decision.reasons)
                    continue
                try:
                    history_reset = self.buffer.append(snapshot)
                except ValueError as error:
                    self.health("DATA_ERROR", sequence=sequence, reasons=[str(error)])
                    continue
                if not self.buffer.ready(self.runtime.minimum_rows):
                    remaining = self.runtime.minimum_rows - len(self.buffer)
                    reset_detail = ""
                    if history_reset:
                        reset_detail = f" History reset: {self.buffer.last_reset_reason}."
                    self.health(
                        "WARMING_UP", sequence=sequence,
                        rows_required=self.runtime.minimum_rows,
                        rows_remaining=remaining,
                        message=(
                            f"Collecting process history: {len(self.buffer)}/"
                            f"{self.runtime.minimum_rows} rows; {remaining} remaining."
                            + reset_detail
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
                    "input_quality": decision.status,
                    "imputed_inputs": [
                        warning.removeprefix("imputed_inputs:")
                        for warning in decision.warnings
                        if warning.startswith("imputed_inputs:")
                    ],
                    "sequence": sequence,
                })
                self.append_prediction(prediction)
                self.last_prediction_at = prediction["generated_at"]
                await self.write_advisory(client, prediction)
                self.health(
                    "RUNNING", sequence=sequence,
                    last_prediction_for=prediction["prediction_for"],
                    input_quality=decision.status,
                    imputed_inputs=prediction["imputed_inputs"],
                    rows_required=self.runtime.minimum_rows,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self.is_connection_failure(error):
                    self.health("DISCONNECTED", sequence=sequence, reasons=[repr(error)])
                    raise
                # Data and model errors are visible but do not stop the subscriber.
                logging.exception("Snapshot processing failed")
                self.health("DATA_ERROR", sequence=sequence, reasons=[repr(error)])
            finally:
                # In the CSV laboratory only, release the next row after this one
                # has completed its full quality/model path. This is intentionally
                # after prediction, not merely after the OPC read.
                if from_subscription:
                    self.queue.task_done()
                if acknowledgement_sequence is not None:
                    try:
                        await self.acknowledge_replay_row(client, acknowledgement_sequence)
                    except Exception as acknowledgement_error:
                        self.health("DISCONNECTED", sequence=acknowledgement_sequence,
                                    reasons=[repr(acknowledgement_error)])
                        raise

    async def run_connected(self) -> None:
        endpoint = self.config["opc"]["endpoint"]
        async with Client(url=endpoint) as client:
            # This laboratory reconnect starts a fresh consistency window.
            self.start_audit_run()
            self.buffer.clear()
            self.last_processed_sequence = None
            self.health("CONNECTED", endpoint=endpoint, message="OPC UA session established")
            sequence = client.get_node(self.tags["system"]["sequence"]["node_id"])
            subscription = await client.create_subscription(
                self.config["opc"]["subscription_period_ms"], SequenceHandler(self.queue)
            )
            await subscription.subscribe_data_change(sequence)
            # Bootstrap explicitly as well as subscribing. This closes a race where
            # the initial data-change notification is delivered before the handler
            # is fully registered on a freshly connected client.
            self.queue.put_nowait(int(await sequence.read_value()))
            await self.process_sequences(client, sequence)

    async def run(self) -> None:
        delay = self.config["opc"]["reconnect_delay_seconds"]
        heartbeat = asyncio.create_task(self.heartbeat_loop(), name="service-heartbeat")
        try:
            while True:
                try:
                    await self.run_connected()
                except Exception as error:
                    logging.warning("OPC endpoint unavailable; retrying in %s seconds: %s", delay, error)
                    self.health("DISCONNECTED", reasons=[repr(error)],
                                message=f"Reconnecting in {delay}s")
                    await asyncio.sleep(delay)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass


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
