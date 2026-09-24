#!/usr/bin/env bash
# Downloads the official Apache Doris release tarball for the pinned version and unpacks only
# its `fe/` directory into `.doris-fe/fe` (the tarball also carries the BE, which is never
# used here). Idempotent: exits early when `.doris-fe/fe/lib/doris-fe.jar` already exists.
#
# The archive is streamed straight into tar, so the multi-GB tarball is never stored.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=doris-version.sh
source scripts/doris-version.sh

if [ -f "${DORIS_FE_DIR}/fe/lib/doris-fe.jar" ]; then
    echo "Doris FE ${DORIS_VERSION} already present at ${DORIS_FE_DIR}/fe"
    exit 0
fi

# Refuse a silent version drift between the FE binary and the IDL submodule.
if submodule_tag=$(git -C doris describe --tags --exact-match 2>/dev/null); then
    if [ "${submodule_tag}" != "${DORIS_VERSION}" ]; then
        echo "error: doris/ submodule is at tag ${submodule_tag} but DORIS_VERSION=${DORIS_VERSION}" >&2
        echo "       (scripts/doris-version.sh); align the two before fetching the FE" >&2
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
member="apache-doris-${DORIS_VERSION}-bin-${arch}/fe"

echo "Fetching ${url} (extracting only ${member})"
mkdir -p "${DORIS_FE_DIR}"
# --strip-components=1 drops the versioned top-level directory so the FE lands at
# ${DORIS_FE_DIR}/fe regardless of arch/version.
curl -fL --retry 3 --progress-bar "${url}" | tar -xzf - -C "${DORIS_FE_DIR}" --strip-components=1 "${member}"

if [ ! -f "${DORIS_FE_DIR}/fe/lib/doris-fe.jar" ]; then
    echo "error: ${DORIS_FE_DIR}/fe/lib/doris-fe.jar missing after extraction" >&2
    exit 1
fi
echo "Doris FE ${DORIS_VERSION} unpacked at ${DORIS_FE_DIR}/fe"
