# 01 — Physical layer: sensors, actuators and the automate

## Sensors (captors)

Sensors convert physical conditions such as flow, temperature, pressure and
level into analog or digital signals. A 4–20 mA signal is common, but the
specific signal and scaling belong to the instrumentation configuration.

## Actuators

Valves, pumps, motors and drives change the process. They remain controlled by
approved PLC/DCS logic. The ML service has no actuator interface.

## PLC / automate

The PLC performs a deterministic cycle: read inputs, execute control logic,
update outputs and communicate diagnostics. It must keep the process safe if
the inference container, network or OPC connection fails.

## PLC versus DCS

| Feature | PLC | DCS |
|---|---|---|
| Scope | Machine or process section | Plant-wide coordination |
| Logic | Fast local logic | Integrated control strategies |
| Connectivity | Direct physical I/O | Many controllers/subsystems |
| Operator view | Limited/local | HMI, alarms and trends |

The actual plant topology must be confirmed; some instruments may terminate in
a DCS controller rather than a separate PLC.
