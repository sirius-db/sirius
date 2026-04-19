# Source this file to point the [s3][integration] / [s3][ioctx][integration]
# tests at the local MinIO container:
#
#   source test/integration/s3/env.sh
#   build/release/extension/sirius/test/cpp/sirius_unittest "[s3][integration]"
#
# The variable names match what read_env() consumes in
# test/cpp/io/s3/test_s3_ioctx.cpp and test_s3_integration.cpp.

export SIRIUS_TEST_S3_ENDPOINT="${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}"
export SIRIUS_TEST_S3_REGION="${SIRIUS_TEST_S3_REGION:-us-east-1}"
export SIRIUS_TEST_S3_ACCESS_KEY="${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}"
export SIRIUS_TEST_S3_SECRET_KEY="${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}"
export SIRIUS_TEST_S3_BUCKET="${SIRIUS_TEST_S3_BUCKET:-sirius-test}"
export SIRIUS_TEST_S3_KEY="${SIRIUS_TEST_S3_KEY:-hello.txt}"

# The new integration test also wants the local fixtures dir so it can
# bit-compare S3 bytes against the local copy.
SIRIUS_TEST_S3_LOCAL_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures/local"
export SIRIUS_TEST_S3_LOCAL_DIR="${SIRIUS_TEST_S3_LOCAL_DIR:-${SIRIUS_TEST_S3_LOCAL_DIR_DEFAULT}}"

# Point SiriusContextExtensionCallback at a tiny memory config so `require sirius`
# in s3_debug_size.test doesn't try to reserve ~90% of the GPU on load. The
# debug path (sirius_debug_datasource_size) never touches the reservation
# manager, so 256/128 MiB is enough headroom.
SIRIUS_TEST_S3_CONFIG_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sirius.yaml"
export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-${SIRIUS_TEST_S3_CONFIG_DEFAULT}}"
