from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
from asyncua import Server, ua

from industrial_inference.configuration import load_configuration


async def create_variable(parent, node_id: str, name: str, value):
    node = await parent.add_variable(ua.NodeId.from_string(node_id), name, value)
    return node


async def run() -> None:
    config, tags = load_configuration()
    simulator = config["simulator"]
    frame = pd.read_csv(Path(simulator["csv_path"]))
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, errors="raise")

    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/tsp-simulator/")
    server.set_server_name("TSP CSV Replay Laboratory Simulator")
    namespace_index = await server.register_namespace(tags["namespace_uri"])
    if namespace_index != 2:
        raise RuntimeError(
            f"Simulator expected namespace 2 but received {namespace_index}; update laboratory Node IDs"
        )

    root = await server.nodes.objects.add_folder(ua.NodeId("TSP", namespace_index), "TSP")
    input_folder = await root.add_folder(ua.NodeId("TSP.Input", namespace_index), "Input")
    system_folder = await root.add_folder(ua.NodeId("TSP.System", namespace_index), "System")
    output_folder = await root.add_folder(ua.NodeId("TSP.ML_Advisory", namespace_index), "ML_Advisory")

    input_nodes = {}
    for name, metadata in tags["inputs"].items():
        initial = "" if name.endswith("TIMESTAMP") else 0.0
        input_nodes[name] = await create_variable(input_folder, metadata["node_id"], name, initial)

    sequence = await create_variable(system_folder, tags["system"]["sequence"]["node_id"], "Sequence", 0)
    acknowledgement = await create_variable(
        system_folder,
        tags["system"]["replay_acknowledgement"]["node_id"],
        "ReplayAcknowledgement",
        0,
    )
    await acknowledgement.set_writable()
    source_timestamp = await create_variable(
        system_folder, tags["system"]["source_timestamp"]["node_id"], "SourceTimestamp", ""
    )
    output_nodes = {}
    for name, metadata in tags["advisory_outputs"].items():
        initial = 0.0 if name in {"prediction", "predicted_change"} else ""
        output_nodes[name] = await create_variable(output_folder, metadata["node_id"], name, initial)
        await output_nodes[name].set_writable()

    laboratory_every = int(simulator["laboratory_update_every_rows"])
    flow_control = bool(simulator.get("flow_control_acknowledgement_enabled", False))
    acknowledgement_timeout = float(simulator.get("acknowledgement_timeout_seconds", 120))
    latest_lab_value, latest_lab_time = None, None
    async with server:
        logging.info("OPC UA simulator listening on port 4840 with %s CSV rows", len(frame))
        while True:
            for position, row in frame.iterrows():
                for name, node in input_nodes.items():
                    if name in {"SLURRY_FREE_ACID", "SLURRY_FREE_ACID_LAB_TIMESTAMP"}:
                        continue
                    if name in row:
                        value = row[name]
                        await node.write_value(float(value) if pd.notna(value) else float("nan"))

                if position % laboratory_every == 0:
                    latest_lab_value = row.get("SLURRY_FREE_ACID")
                    latest_lab_time = row["Date"]
                if latest_lab_value is not None:
                    await input_nodes["SLURRY_FREE_ACID"].write_value(float(latest_lab_value))
                    await input_nodes["SLURRY_FREE_ACID_LAB_TIMESTAMP"].write_value(latest_lab_time.isoformat())

                await source_timestamp.write_value(row["Date"].isoformat())
                await sequence.write_value(int(position + 1))  # commit marker written last
                if flow_control:
                    # Laboratory-only back-pressure: do not make historical time
                    # advance faster than the subscriber can read each row.
                    deadline = asyncio.get_running_loop().time() + acknowledgement_timeout
                    while int(await acknowledgement.read_value()) < int(position + 1):
                        if asyncio.get_running_loop().time() >= deadline:
                            raise TimeoutError(
                                f"Inference acknowledgement did not reach sequence {position + 1} "
                                f"within {acknowledgement_timeout:.0f}s"
                            )
                        await asyncio.sleep(0.02)
                await asyncio.sleep(float(simulator["replay_interval_seconds"]))
            if not simulator["loop"]:
                logging.info("CSV replay completed; server remains available")
                await asyncio.Event().wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
