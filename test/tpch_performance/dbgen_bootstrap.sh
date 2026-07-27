# Shared bootstrap for the classic TPC-H tools (dbgen, qgen).
#
# Source this file, then call:
#   ensure_tpch_tools <dbgen_dir> <project_dir> <tool>...

ensure_tpch_tools() {
    local dbgen_dir="$1" project_dir="$2"
    shift 2
    local archive="$project_dir/test_datasets/tpch-dbgen.zip"

    if [ ! -d "$dbgen_dir" ]; then
        if [ -f "$archive" ]; then
            echo "TPC-H tools not found; unzipping test_datasets/tpch-dbgen.zip"
            (cd "$project_dir/test_datasets" && unzip -nq tpch-dbgen.zip)
        fi
    elif [ -f "$archive" ] && ! grep -q '3\.0\.1' "$dbgen_dir/NOTICE" 2>/dev/null; then
        # An extraction from an older bundle is on disk, and its qgen draws
        # substitution parameters outside the ranges the spec defines. Refresh
        # the sources in place so generated .tbl data alongside them survives,
        # then force a rebuild.
        echo "Extracted TPC-H tools predate the bundled 3.0.1; refreshing sources"
        (cd "$project_dir/test_datasets" && unzip -oq tpch-dbgen.zip)
        find "$dbgen_dir" -maxdepth 1 -name '*.o' -delete
        rm -f "$dbgen_dir/dbgen" "$dbgen_dir/qgen"
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
