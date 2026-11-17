# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

has_parquet_data() {
    local parquet_dir="$1"
    local parquet_file
    for parquet_file in "$parquet_dir"/*.parquet "$parquet_dir"/*/*.parquet; do
        [ -f "$parquet_file" ] && return 0
    done
    return 1
}

ensure_parquet_data() {
    local sf="$1" parquet_dir="$2"
    if has_parquet_data "$parquet_dir"; then
        return 0
    fi

    echo "Parquet data not found: $parquet_dir"
    echo "Generating TPC-H SF${sf} dataset..."
    local status=0
    pixi run bash "$SCRIPT_DIR/generate_tpch_data.sh" "$sf" --format parquet --output "$parquet_dir" || status=$?
    if [ "$status" -ne 0 ]; then
        echo "ERROR: failed to generate parquet data for SF${sf} (exit code ${status})." >&2
        return "$status"
    fi

    if ! has_parquet_data "$parquet_dir"; then
        echo "ERROR: parquet data is still unavailable after generation: $parquet_dir" >&2
        return 1
    fi
}
