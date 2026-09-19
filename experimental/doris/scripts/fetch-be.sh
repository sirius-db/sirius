#!/usr/bin/env bash
# Downloads the official Apache Doris release tarball for the pinned version and unpacks only
# its `be/` directory into `.doris-be/be`: the native BE the benchmark (plan-doc
# experiments/sf10-bench) runs as the CPU reference next to this directory's Sirius backend.
# Same streaming extraction as fetch-fe.sh (the multi-GB tarball is never stored); idempotent,
# exits early when `.doris-be/be/lib/doris_be` already exists.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=doris-version.sh
source scripts/doris-version.sh

if [ -x "${DORIS_BE_DIR}/be/lib/doris_be" ]; then
    echo "Doris BE ${DORIS_VERSION} already present at ${DORIS_BE_DIR}/be"
    exit 0
fi

# Refuse a silent version drift between the BE binary and the IDL submodule (the FE and the
# native BE must be the same release, see fetch-fe.sh).
if submodule_tag=$(git -C doris describe --tags --exact-match 2>/dev/null); then
    if [ "${submodule_tag}" != "${DORIS_VERSION}" ]; then
        echo "error: doris/ submodule is at tag ${submodule_tag} but DORIS_VERSION=${DORIS_VERSION}" >&2
        echo "       (scripts/doris-version.sh); align the two before fetching the BE" >&2
        exit 1
    fi
fi

case "$(uname -m)" in
    x86_64 | amd64) arch="x64" ;;
    aarch64 | arm64) arch="arm64" ;;
    *)
        echo "error: unsupported architecture $(uname -m)" >&2
        exit 1
        ;;
esac

tarball="apache-doris-${DORIS_VERSION}-bin-${arch}.tar.gz"
url="${DORIS_DOWNLOAD_BASE:-https://apache-doris-releases.oss-accelerate.aliyuncs.com}/${tarball}"
member="apache-doris-${DORIS_VERSION}-bin-${arch}/be"

echo "Fetching ${url} (extracting only ${member})"
mkdir -p "${DORIS_BE_DIR}"
# --strip-components=1 drops the versioned top-level directory so the BE lands at
# ${DORIS_BE_DIR}/be regardless of arch/version.
curl -fL --retry 3 --progress-bar "${url}" | tar -xzf - -C "${DORIS_BE_DIR}" --strip-components=1 "${member}"

if [ ! -x "${DORIS_BE_DIR}/be/lib/doris_be" ]; then
    echo "error: ${DORIS_BE_DIR}/be/lib/doris_be missing after extraction" >&2
    exit 1
fi
echo "Doris BE ${DORIS_VERSION} unpacked at ${DORIS_BE_DIR}/be"
