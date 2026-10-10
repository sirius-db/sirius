# Sirius as a StarRocks compute node

A Rust process (`sirius-starrocks-cn`) that joins a StarRocks FE as a compute node and runs the
plan fragments it receives on the embedded Sirius engine. The `starrocks/` submodule pins the
StarRocks release the CN is built against; its thrift and protobuf definitions are compiled into
the CN.

## StarRocks versions

The CN works with one StarRocks release: the one the submodule pins (4.1.3 today). The heartbeat
reports it, so `SHOW COMPUTE NODES` shows e.g.
`sirius-starrocks-cn/0.1.0 (starrocks 4.1.3, 8a8e186)`.

- Build the CN for each StarRocks release.
- Run it with an FE of the same release.
- Upgrade the FE and the CN together.

Nothing in the protocol checks versions, and the thrift plan structures change between minor
releases. StarRocks upgrades backends before the FE and expects them to accept the previous
release's plans; the CN doesn't, so it may reject or misread plans from an FE of another release.
