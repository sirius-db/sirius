#!/usr/bin/env bash
# Generate the local self-signed certificate used by the TLS MinIO service.
#
# The generated certs are intentionally not checked in. They are recreated by
# `make s3-up` before docker compose starts MinIO so the TLS service boots with
# /root/.minio/certs/{public.crt,private.key} already present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/certs"
PUBLIC_CERT="${CERT_DIR}/public.crt"
PRIVATE_KEY="${CERT_DIR}/private.key"

if [[ -s "${PUBLIC_CERT}" && -s "${PRIVATE_KEY}" ]]; then
  echo "[s3-tls] using existing certificate under ${CERT_DIR}"
  exit 0
fi

mkdir -p "${CERT_DIR}"
echo "[s3-tls] generating self-signed MinIO TLS certificate under ${CERT_DIR}"
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout "${PRIVATE_KEY}" \
  -out "${PUBLIC_CERT}" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost,DNS:host.docker.internal"
