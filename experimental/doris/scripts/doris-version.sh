# Pinned Apache Doris release. Must match the tag the `doris/` submodule is checked out at:
# the FE binary and the thrift/proto IDL the backend is generated from are the same version
# (ADR-011 D-2), so the enum surface the translator sees is exactly what the FE sends.
DORIS_VERSION="4.1.4"

# Directory the fetched FE is unpacked into (relative to experimental/doris).
DORIS_FE_DIR=".doris-fe"

# Directory the fetched official BE (the benchmark's native reference, scripts/fetch-be.sh)
# is unpacked into (relative to experimental/doris).
DORIS_BE_DIR=".doris-be"
