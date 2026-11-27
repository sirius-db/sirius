#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_SCRIPT="$SCRIPT_DIR/sweep_threads.sh"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/sirius_sweep_test.XXXXXX")

cleanup() {
    rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

QUERY_DIR="$TMP_ROOT/queries"
PARQUET_DIR="$TMP_ROOT/parquet"
OUTPUT_DIR="$TMP_ROOT/output"
CAPTURE_FILE="$TMP_ROOT/configurations.txt"
CONFIG_PATHS="$TMP_ROOT/config_paths.txt"
FAKE_DUCKDB="$TMP_ROOT/duckdb"
mkdir -p "$QUERY_DIR" "$PARQUET_DIR"
printf 'SELECT 1;\n' > "$QUERY_DIR/q1.sql"

for table in customer lineitem nation orders part partsupp region supplier; do
    : > "$PARQUET_DIR/${table}.parquet"
done

cat > "$FAKE_DUCKDB" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ "${SWEEP_FAKE_FAIL:-0}" = "1" ]; then
    echo "synthetic DuckDB failure" >&2
    exit 42
fi

read_threads() {
    local section=$1
    awk -v wanted="$section" '
        $1 == wanted ":" { active = 1; next }
        active && $1 == "num_threads:" { print $2; exit }
        active && $1 ~ /:$/ { exit 1 }
    ' "$SIRIUS_CONFIG_FILE"
}

pipeline=$(read_threads pipeline)
scan=$(read_threads scan_manager)
task_creator=$(read_threads task_creator)
printf '%s/%s/%s\n' "$pipeline" "$scan" "$task_creator" >> "$SWEEP_CAPTURE_FILE"
printf '%s\n' "$SIRIUS_CONFIG_FILE" >> "$SWEEP_CONFIG_PATHS"
printf 'Run Time (s): real 1.25 user 0.00 sys 0.00\n'
printf 'Run Time (s): real 0.75 user 0.00 sys 0.00\n'
EOF
chmod +x "$FAKE_DUCKDB"

INTEGRATION_CONFIG="$PROJECT_DIR/test/cpp/integration/integration.yaml"
INTEGRATION_COPY="$TMP_ROOT/integration.yaml"
cp "$INTEGRATION_CONFIG" "$INTEGRATION_COPY"

SWEEP_DUCKDB="$FAKE_DUCKDB" \
SWEEP_QUERY_DIR="$QUERY_DIR" \
SWEEP_PARQUET_DIR="$PARQUET_DIR" \
SWEEP_OUTPUT_DIR="$OUTPUT_DIR" \
SWEEP_QUERIES=1 \
SWEEP_CAPTURE_FILE="$CAPTURE_FILE" \
SWEEP_CONFIG_PATHS="$CONFIG_PATHS" \
bash "$SWEEP_SCRIPT" > "$TMP_ROOT/success.log"

cat > "$TMP_ROOT/expected_configurations.txt" <<'EOF'
8/4/4
4/4/4
12/4/4
16/4/4
8/3/4
8/8/4
8/4/2
8/4/8
EOF
diff -u "$TMP_ROOT/expected_configurations.txt" "$CAPTURE_FILE"

CONFIG_PATH=$(head -n 1 "$CONFIG_PATHS")
[ -n "$CONFIG_PATH" ]
[ "${CONFIG_PATH#*XXXXXX}" = "$CONFIG_PATH" ]
[ "$(sort -u "$CONFIG_PATHS" | wc -l | tr -d ' ')" = "1" ]
[ ! -e "$CONFIG_PATH" ]
cmp "$INTEGRATION_COPY" "$INTEGRATION_CONFIG"

# The config parser requires scan_manager.num_threads > 2.
(
    # shellcheck source=sweep_threads.sh
    source "$SWEEP_SCRIPT"
    CONFIG_FILE="$TMP_ROOT/invalid.yaml"
    if update_config 8 2 4 > "$TMP_ROOT/invalid.out" 2>&1; then
        echo "scan thread validation unexpectedly accepted 2" >&2
        exit 1
    fi
)
grep -q "scan threads must be at least 3" "$TMP_ROOT/invalid.out"

if SWEEP_DUCKDB="$FAKE_DUCKDB" \
    SWEEP_QUERY_DIR="$QUERY_DIR" \
    SWEEP_PARQUET_DIR="$PARQUET_DIR" \
    SWEEP_OUTPUT_DIR="$TMP_ROOT/failure-output" \
    SWEEP_QUERIES=1 \
    SWEEP_CAPTURE_FILE="$TMP_ROOT/failure-configurations.txt" \
    SWEEP_CONFIG_PATHS="$TMP_ROOT/failure-config-paths.txt" \
    SWEEP_FAKE_FAIL=1 \
    bash "$SWEEP_SCRIPT" > "$TMP_ROOT/failure.log" 2>&1; then
    echo "DuckDB failure unexpectedly returned success" >&2
    exit 1
fi
grep -q "DuckDB failed for baseline_p8_s4_t4 query 1" "$TMP_ROOT/failure.log"
[ ! -e "$TMP_ROOT/failure-output/baseline_p8_s4_t4.csv" ]
[ ! -e "$TMP_ROOT/failure-output/baseline_p8_s4_t4.csv.partial" ]
if grep -q "N/A" "$TMP_ROOT/failure.log"; then
    echo "failure path emitted misleading N/A timing evidence" >&2
    exit 1
fi
cmp "$INTEGRATION_COPY" "$INTEGRATION_CONFIG"

echo "thread sweep harness tests passed"
