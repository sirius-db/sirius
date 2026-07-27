# Shared bootstrap for the classic TPC-H tools (dbgen, qgen).
#
# Source this file, then call:
#   ensure_tpch_tools <dbgen_dir> <project_dir> <tool>...

# The bundled dbgen is 2.14.0, which draws two substitution parameters outside
# the ranges the spec defines. TPC fixed both in 3.0.1; apply the same fixes
# before building qgen so generated queries are compliant.
#
#   q4  month offset   UnifInt(1,58) -> UnifInt(0,57)  (1993-01 .. 1997-10)
#   q22 country codes  {10..34} + 10 -> {0..24} + 10   (10 .. 34)
#
# Only qgen links varsub.o, so this cannot affect dbgen's data generation.
patch_qgen_substitutions() {
    local dir="$1"
    local src="$dir/varsub.c"
    local patched=0

    if [ ! -f "$src" ]; then
        return 0
    fi
    if grep -q '^long ccode\[25\] = {10,' "$src"; then
        sed -i 's/^long ccode\[25\] = {10,[^}]*};/long ccode[25] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};/' "$src"
        patched=1
    fi
    if grep -q 'UnifInt((DSS_HUGE)1,(DSS_HUGE)58,qnum)' "$src"; then
        sed -i 's/UnifInt((DSS_HUGE)1,(DSS_HUGE)58,qnum)/UnifInt((DSS_HUGE)0,(DSS_HUGE)57,qnum)/' "$src"
        patched=1
    fi
    if [ "$patched" -eq 1 ]; then
        echo "Patched varsub.c to the 3.0.1 substitution ranges (q4 month, q22 country codes)"
        rm -f "$dir/qgen" "$dir/varsub.o"
    fi
}

ensure_tpch_tools() {
    local dbgen_dir="$1" project_dir="$2"
    shift 2

    if [ ! -d "$dbgen_dir" ] && [ -f "$project_dir/test_datasets/tpch-dbgen.zip" ]; then
        echo "TPC-H tools not found; unzipping test_datasets/tpch-dbgen.zip"
        (cd "$project_dir/test_datasets" && unzip -nq tpch-dbgen.zip)
    fi

    local tool
    for tool in "$@"; do
        if [ "$tool" = "qgen" ]; then
            patch_qgen_substitutions "$dbgen_dir"
        fi
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
