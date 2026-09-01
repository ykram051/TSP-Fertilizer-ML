# 03 — Docker isolated inference environment

## Images and containers

- An image packages Python, libraries, code and configuration.
- A container is a running instance of that image.

The package uses two images so that protocol simulation and inference remain
independent:

1. `opc-simulator`: replays the historical CSV as OPC UA nodes.
2. `inference-service`: subscribes, validates, buffers and predicts.

## Windows prototype

The first host is Windows with Docker Desktop using Linux containers. A future
traditional KEPServerEX installation is expected to remain on an approved
Windows host or VM and be reached over the OT network; it is not bundled into
the Python image.

## Persistence

Containers are disposable. The `runtime/` host directory is mounted as a volume
for predictions and health state. Model artifacts are mounted read-only.

## Reproducibility

`requirements.txt` pins the direct Python dependencies. Before plant testing,
the exact transitive environment must be locked and the image scanned/approved
under the site's IT/OT process.

LightGBM also requires the Linux OpenMP runtime `libgomp1`. It is installed as
an operating-system package in `Dockerfile.inference`, and the image build runs
an explicit `import lightgbm` smoke test before accepting the image.
