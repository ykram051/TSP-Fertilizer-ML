# 07 — Plant information still required

## Architecture and hardware

- [ ] DCS vendor, product and version.
- [ ] Meaning/vendor/model of the “automate” in this installation.
- [ ] Whether each required tag originates in a PLC, DCS or historian.
- [ ] Available historian and retention policy.

## Connectivity and security

- [ ] OPC UA or OPC DA.
- [ ] Existing Kepware installation, version and license.
- [ ] Kepware-to-controller driver/protocol.
- [ ] Production and laboratory OPC endpoints.
- [ ] OPC security policy and message mode.
- [ ] Authentication method and certificate approval process.
- [ ] Firewall route between Windows Kepware host and Docker host.
- [ ] Approved advisory-output mechanism, if any.

## Data

- [ ] Exact Node IDs for all variables in `config/opc_tags.yaml`.
- [ ] Engineering units and physical limits.
- [x] Nominal process sampling cadence: one minute.
- [x] `SLURRY_FREE_ACID` is a manual laboratory value.
- [ ] Laboratory sampling frequency.
- [ ] Whether the DCS repeats the last laboratory value every minute.
- [ ] A tag containing the actual laboratory sample/result timestamp.
- [ ] Maximum acceptable age of a laboratory result for anchored forecasting.
- [ ] Meaning of OPC bad/uncertain quality codes used by the site.

## Environment and audit

- [x] First Docker host: Windows/Docker Desktop.
- [ ] Final Docker host and approved operating-system version.
- [ ] Log/prediction retention location.
- [ ] Clock synchronization source.
- [ ] Image registry, vulnerability scanning and release approval process.
- [ ] Named model owner and rollback procedure.
