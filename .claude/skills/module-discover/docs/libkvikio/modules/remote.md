# Remote File Access

**Status**: UNUSED
**Path**: `kvikio/remote_handle.hpp`

## Summary
Enables reading remote files (S3, HTTP, WebHDFS) directly into GPU memory. Supports multiple endpoint types with auto-detection, parallel reads via thread pool, and configurable retry/timeout behavior.

## Key APIs
- `RemoteHandle` — Handle for remote file access; `read()` and `pread()` methods
- `RemoteHandle::open()` — Factory method with auto-detection of endpoint type
- `RemoteEndpointType` — Enum: `AUTO`, `S3`, `S3_PUBLIC`, `S3_PRESIGNED_URL`, `WEBHDFS`, `HTTP`
- `S3Endpoint` — AWS S3 endpoint with credentials and region support
- `HttpEndpoint` — Generic HTTP/HTTPS endpoint
- `S3PublicEndpoint` — Public S3 bucket access (no credentials)

## Potential Relevance
Medium. Could enable Sirius to query Parquet files directly from S3/cloud storage with GPU-accelerated reads, avoiding the need to download data to local storage first.
