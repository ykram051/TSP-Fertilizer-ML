# 02 — Translation layer: Kepware and OPC UA

## Kepware

Kepware uses vendor/device drivers on its southbound side and exposes a common
OPC UA address space on its northbound side.

```text
PLC/DCS protocol → Kepware channel/device/tag → OPC UA node → Python client
```

The real endpoint, security policy, namespace indexes and Node IDs are not yet
known. `config/opc_tags.yaml` therefore uses explicit laboratory placeholders.

## Client/server roles

- Kepware or the Stage-1 simulator is the OPC UA **server**.
- The Dockerized Python inference service is the OPC UA **client**.
- The client subscribes to input nodes and receives data-change notifications.

## Subscription policy

The service subscribes to input tags and creates a one-minute snapshot only when
all required variables are available. Native changes can arrive more often; the
model cadence remains one minute.

The Stage-1 simulator uses `TSP.System.Sequence` as an atomic commit marker: it
updates all values first and the sequence last. Production deployment requires
either an equivalent DCS/Kepware heartbeat or a time-based snapshot collector.
This trigger must be agreed with IT/OT; the laboratory sequence tag must not be
assumed to exist in the plant.

## Production security requirements

- OPC UA `SignAndEncrypt`.
- Trusted client and server application certificates.
- Anonymous login disabled.
- Dedicated least-privilege service identity.
- Read-only access to plant input tags.
- Optional writes restricted to a dedicated advisory namespace.
- Firewall rule only between the inference host and approved OPC endpoint.

The Stage-1 simulator permits an unsecured local endpoint only because it runs
on an isolated Docker laboratory network. That configuration must not be copied
to the plant.
