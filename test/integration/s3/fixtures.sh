#!/usr/bin/env bash
# Generate local fixtures and upload them to the MinIO container.
#
# Prerequisites (handled by `make s3-up`):
#   - MinIO container running at http://127.0.0.1:9000 with credentials
#     minioadmin/minioadmin (see docker-compose.yml).
#   - Python 3 (stdlib only — no extra deps).
#
# Usage:
#   test/integration/s3/fixtures.sh            # generate + upload
#   BUCKET=other test/integration/s3/fixtures.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/local"
ENDPOINT="${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}"
ACCESS_KEY="${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}"
SECRET_KEY="${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}"
BUCKET="${SIRIUS_TEST_S3_BUCKET:-sirius-test}"
MC_IMAGE="${MC_IMAGE:-minio/mc:RELEASE.2025-04-16T18-13-26Z}"

echo "[fixtures] generating local fixtures under ${FIXTURE_DIR}"
python3 "${SCRIPT_DIR}/generate_fixtures.py" --out "${FIXTURE_DIR}"

echo "[fixtures] waiting for MinIO at ${ENDPOINT}"
for _ in $(seq 1 30); do
  if curl -sf "${ENDPOINT}/minio/health/ready" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -sf "${ENDPOINT}/minio/health/ready" >/dev/null; then
  echo "[fixtures] MinIO did not become ready at ${ENDPOINT}; is it up?" >&2
  exit 1
fi

# mc talks to the host-network MinIO. host.docker.internal works on macOS;
# on Linux we use --network=host + the loopback endpoint.
if [[ "$(uname -s)" == "Linux" ]]; then
  MC_NET=(--network=host)
  MC_ENDPOINT="${ENDPOINT}"
else
  MC_NET=()
  MC_ENDPOINT="${ENDPOINT/127.0.0.1/host.docker.internal}"
fi

mc_run() {
  docker run --rm "${MC_NET[@]}" \
    -v "${FIXTURE_DIR}:/fixtures:ro" \
    --entrypoint /bin/sh \
    "${MC_IMAGE}" \
    -c "mc alias set local ${MC_ENDPOINT} ${ACCESS_KEY} ${SECRET_KEY} >/dev/null && $*"
}

echo "[fixtures] ensuring bucket ${BUCKET} exists"
mc_run "mc mb --ignore-existing local/${BUCKET}"

echo "[fixtures] uploading fixtures to s3://${BUCKET}/"
mc_run "mc cp --recursive /fixtures/ local/${BUCKET}/"

echo "[fixtures] listing s3://${BUCKET}/"
mc_run "mc ls --recursive local/${BUCKET}"

echo "[fixtures] done"
