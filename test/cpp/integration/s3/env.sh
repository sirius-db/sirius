# Source this file to point the [s3][integration] / [s3][ioctx][integration]
# tests at the local MinIO container:
#
#   source test/cpp/integration/s3/env.sh
#   build/release/extension/sirius/test/cpp/sirius_unittest "[s3][integration]"
#
# The variable names match what read_env() consumes in
# test/cpp/io/s3/test_s3_ioctx.cpp and test_s3_integration.cpp.

export SIRIUS_TEST_S3_ENDPOINT="${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}"
export SIRIUS_TEST_S3_HTTPS_ENDPOINT="${SIRIUS_TEST_S3_HTTPS_ENDPOINT:-https://127.0.0.1:9443}"
export SIRIUS_TEST_S3_REGION="${SIRIUS_TEST_S3_REGION:-us-east-1}"
export SIRIUS_TEST_S3_ACCESS_KEY="${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}"
export SIRIUS_TEST_S3_SECRET_KEY="${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}"
export SIRIUS_TEST_S3_BUCKET="${SIRIUS_TEST_S3_BUCKET:-sirius-test}"
export SIRIUS_TEST_S3_KEY="${SIRIUS_TEST_S3_KEY:-hello.txt}"
# Default to best-effort skips for manual runs; make s3-test overrides this to 1
# so runtime S3 failures turn the job red instead of going green with a skip.
export SIRIUS_TEST_S3_STRICT="${SIRIUS_TEST_S3_STRICT:-0}"

# Integration tests also want the local fixtures dir so they can bit-compare
# S3 bytes against the local copy.
SIRIUS_TEST_S3_LOCAL_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures/local"
export SIRIUS_TEST_S3_LOCAL_DIR="${SIRIUS_TEST_S3_LOCAL_DIR:-${SIRIUS_TEST_S3_LOCAL_DIR_DEFAULT}}"

# The TLS MinIO service uses a generated self-signed cert. Tests pass this CA
# bundle to s3_ioctx so TLS verification remains enabled.
SIRIUS_TEST_S3_CA_BUNDLE_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs/public.crt"
export SIRIUS_TEST_S3_CA_BUNDLE="${SIRIUS_TEST_S3_CA_BUNDLE:-${SIRIUS_TEST_S3_CA_BUNDLE_DEFAULT}}"

# Point SiriusContextExtensionCallback at a tiny memory config so `require sirius`
# in s3_debug_size.test doesn't try to reserve ~90% of the GPU on load. The
# debug path (sirius_debug_datasource_size) never touches the reservation
# manager, so 256/128 MiB is enough headroom.
SIRIUS_TEST_S3_CONFIG_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sirius.yaml"
export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-${SIRIUS_TEST_S3_CONFIG_DEFAULT}}"
