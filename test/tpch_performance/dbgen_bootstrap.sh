# Shared bootstrap for the classic TPC-H tools (dbgen, qgen).
#
# Source this file, then call:
#   ensure_tpch_tools <dbgen_dir> <project_dir> <tool>...

ensure_tpch_tools() {
    local dbgen_dir="$1" project_dir="$2"
    shift 2

    if [ ! -d "$dbgen_dir" ] && [ -f "$project_dir/test_datasets/tpch-dbgen.zip" ]; then
        echo "TPC-H tools not found; unzipping test_datasets/tpch-dbgen.zip"
        (cd "$project_dir/test_datasets" && unzip -nq tpch-dbgen.zip)
    fi

    local tool
    for tool in "$@"; do
        if [ ! -x "$dbgen_dir/$tool" ] && [ -f "$dbgen_dir/makefile" ]; then
            echo "Building $tool in $dbgen_dir"
            make -C "$dbgen_dir" "$tool" >/dev/null
        fi
        if [ ! -x "$dbgen_dir/$tool" ]; then
            echo "ERROR: $tool not found or not executable at $dbgen_dir/$tool"
            return 1
        fi
    done
}
