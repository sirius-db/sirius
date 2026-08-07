#!/usr/bin/env bash
# Engine B (baseline): stock StarRocks, 1 FE + 2 BEs on one host, from the prebuilt
# artifacts Docker image (the release tarball URLs 403). Safe to re-run after rm -rf.
#
# The two backend processes MUST run as BEs, not CNs: shared-nothing CNs are not
# schedulable for FILES() external scans and every query fails with "No available
# backends". Registration is `ALTER SYSTEM ADD BACKEND`, launch is start_be.sh.
#
# Environment: B_DIR (default ~/starrocks-bench), SR_IMAGE (default 3.5.20), JAVA_HOME.
set -eu
B=${B_DIR:-$HOME/starrocks-bench}
IMG=${SR_IMAGE:-starrocks/artifacts-ubuntu:3.5.20}
export JAVA_HOME=${JAVA_HOME:?set JAVA_HOME to a JDK 17+ (the FE needs it)}

if [ ! -d $B/fe ]; then
  mkdir -p $B
  docker rm -f sr-extract >/dev/null 2>&1 || true
  docker create --name sr-extract $IMG true
  # artifacts image layout: /release/{fe_artifacts,be_artifacts,broker_artifacts}
  docker cp sr-extract:/release/fe_artifacts/fe $B/fe 2>/dev/null || docker cp sr-extract:/fe $B/fe
  docker cp sr-extract:/release/be_artifacts/be $B/be 2>/dev/null || docker cp sr-extract:/be $B/be
  docker rm sr-extract >/dev/null
fi

mkdir -p $B/fe/meta
cat > $B/fe/conf/fe.conf <<'EOF'
priority_networks = 127.0.0.1/32
meta_dir = ${STARROCKS_HOME}/meta
http_port = 8030
rpc_port = 9020
query_port = 9030
edit_log_port = 9010
sys_log_level = INFO
EOF

# Two BE trees; port pairs chosen to mirror the Sirius demo CNs so the same host
# can run either engine (never both at once -- they also share the FE port).
for i in 1 2; do
  [ -d $B/be$i ] || cp -r $B/be $B/be$i
  mkdir -p $B/be$i/storage
done
cat > $B/be1/conf/be.conf <<'EOF'
priority_networks = 127.0.0.1/32
mem_limit = 16G
be_port = 9060
heartbeat_service_port = 9050
brpc_port = 8060
be_http_port = 8040
starlet_port = 9070
storage_root_path = ${STARROCKS_HOME}/storage
sys_log_level = INFO
EOF
cat > $B/be2/conf/be.conf <<'EOF'
priority_networks = 127.0.0.1/32
mem_limit = 16G
be_port = 9062
heartbeat_service_port = 9052
brpc_port = 8062
be_http_port = 8042
starlet_port = 9072
storage_root_path = ${STARROCKS_HOME}/storage
sys_log_level = INFO
EOF

echo "engine B laid out at $B. To run:"
echo "  $B/fe/bin/start_fe.sh --daemon"
echo "  $B/be1/bin/start_be.sh --daemon && $B/be2/bin/start_be.sh --daemon"
echo "  mysql -h127.0.0.1 -P9030 -uroot -e 'ALTER SYSTEM ADD BACKEND \"127.0.0.1:9050\"; ALTER SYSTEM ADD BACKEND \"127.0.0.1:9052\";'"
