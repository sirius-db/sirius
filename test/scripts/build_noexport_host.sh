#!/usr/bin/env bash
# Relinks the wrapper's DuckDB CLI without `-rdynamic`, producing a host that exports no
# DuckDB symbols: the shape of the official CLI releases (#2174). Reuses the objects the
# wrapper build already produced, so this takes about a minute instead of a DuckDB build.
#
# Usage:
#   test/scripts/build_noexport_host.sh [<wrapper-build-dir>] [<output>]
# Defaults: sirius-duckdb/build/release and <wrapper-build-dir>/duckdb-noexport. A second
# host, <output>-local, is derived from it with both reference symbols turned into LOCAL
# definitions (objcopy --localize-symbol): the shape of the official DuckDB v2.0 preview CLI,
# which keeps them in .symtab without global binding.
#
# The wrapper configures DuckDB with EXPORT_DYNAMIC_SYMBOLS=ON, which puts `-rdynamic` in
# CMAKE_EXE_LINKER_FLAGS; the only difference between the two binaries is that flag. The
# script fails rather than report a fixture it could not verify: the original CLI must be
# unchanged, the output must be freshly written, and the two scan-reference symbols must be
# absent from the dynamic symbol table and present as global definitions in `.symtab`.
set -euo pipefail

build=${1:-sirius-duckdb/build/release}
out=${2:-$build/duckdb-noexport}
original=$build/duckdb
[ -f "$build/build.ninja" ] || { echo "not a ninja build directory: $build" >&2; exit 2; }
[ -x "$original" ] || { echo "wrapper CLI not built: $original" >&2; exit 2; }
for tool in ninja nm objcopy sha256sum python3; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done
out=$(realpath -m "$out")
local_out="$out-local"
original_path=$(realpath "$original")
for target in "$out" "$(realpath -m "$local_out")"; do
  if [ "$target" = "$original_path" ]; then
    echo "an output must not be the wrapper CLI itself: $target" >&2
    exit 2
  fi
done
before=$(sha256sum "$original" | cut -d' ' -f1)
unchanged() {
  [ "$before" = "$(sha256sum "$original" | cut -d' ' -f1)" ] ||
    { echo "the wrapper CLI changed while building $1: $original" >&2; exit 1; }
}

# The last command ninja would run for the CLI target is its link step, in CMake's
# `: && <link> && :` wrapper. Rewrite it with a real tokenizer: drop every -rdynamic, redirect
# the single -o argument, keep everything else, and run it without a shell.
link=$(ninja -C "$build" -t commands duckdb | tail -1)
rm -f "$out"
python3 - "$link" "$out" "$build" <<'PY'
import shlex, subprocess, sys
command, out, build = sys.argv[1:4]
inner = command.strip()
if inner.startswith(": &&"):
    inner = inner[len(": &&"):]
if inner.endswith("&& :"):
    inner = inner[: -len("&& :")]
tokens = shlex.split(inner)
if any(t in ("&&", "||", ";", "|") for t in tokens) or any(t.startswith("@") for t in tokens):
    sys.exit("unsupported CLI link command shape (compound command or response file):\n" + command)
if "-rdynamic" not in tokens:
    sys.exit("the CLI link command carries no -rdynamic; nothing to strip:\n" + command)
outputs = [i for i, t in enumerate(tokens) if t == "-o"]
if len(outputs) != 1 or outputs[0] + 1 >= len(tokens):
    sys.exit("expected exactly one -o <file> in the CLI link command:\n" + command)
if tokens[outputs[0] + 1].split("/")[-1] != "duckdb":
    sys.exit("the link command does not produce the duckdb CLI:\n" + command)
tokens[outputs[0] + 1] = out
tokens = [t for t in tokens if t != "-rdynamic"]
subprocess.run(tokens, cwd=build, check=True)
PY
[ -x "$out" ] || { echo "relink produced no executable at $out" >&2; exit 1; }
unchanged "$out"

# Both scan-reference lookups must miss dynamically and hit as global definitions in .symtab.
dynamic=$(nm -D --defined-only "$out") || { echo "nm -D failed on $out" >&2; exit 1; }
static=$(nm --defined-only "$out") || { echo "nm failed on $out" >&2; exit 1; }
for symbol in _ZN6duckdb17TableScanFunction11GetFunctionEv duckdb_extension_parquet_describe; do
  if printf '%s\n' "$dynamic" | awk -v s="$symbol" '$3 == s { found = 1 } END { exit !found }'; then
    echo "relinked CLI still exports $symbol dynamically: $out" >&2
    exit 1
  fi
  binding=$(printf '%s\n' "$static" | awk -v s="$symbol" '$3 == s && !found { print $2; found = 1 }')
  if [ "$binding" != "T" ]; then
    echo "$symbol is not a global text definition in the static symbol table (got '${binding:-absent}'): $out" >&2
    exit 1
  fi
done
exports=$(printf '%s\n' "$dynamic" | grep -c ' _ZN6duckdb' || true)
echo "no-export host: $out"
echo "  version: $("$out" -csv -noheader -c 'SELECT library_version, source_id FROM pragma_version();')"
echo "  dynamic duckdb:: exports: $exports; TableScanFunction::GetFunction and duckdb_extension_parquet_describe: static GLOBAL"
echo "  wrapper CLI unchanged: $before"

# The localized variant: same image, both reference symbols demoted to LOCAL binding.
rm -f "$local_out"
objcopy --localize-symbol=_ZN6duckdb17TableScanFunction11GetFunctionEv \
        --localize-symbol=duckdb_extension_parquet_describe "$out" "$local_out"
[ -x "$local_out" ] || { echo "objcopy produced no executable at $local_out" >&2; exit 1; }
unchanged "$local_out"
dynamic=$(nm -D --defined-only "$local_out") || { echo "nm -D failed on $local_out" >&2; exit 1; }
static=$(nm --defined-only "$local_out") || { echo "nm failed on $local_out" >&2; exit 1; }
for symbol in _ZN6duckdb17TableScanFunction11GetFunctionEv duckdb_extension_parquet_describe; do
  if printf '%s\n' "$dynamic" | awk -v s="$symbol" '$3 == s { found = 1 } END { exit !found }'; then
    echo "localized CLI still exports $symbol dynamically: $local_out" >&2
    exit 1
  fi
  binding=$(printf '%s\n' "$static" | awk -v s="$symbol" '$3 == s && !found { print $2; found = 1 }')
  if [ "$binding" != "t" ]; then
    echo "$symbol is not a local text definition in $local_out (got '${binding:-absent}')" >&2
    exit 1
  fi
done
echo "localized host: $local_out (both reference symbols static LOCAL, 0 dynamic exports of them)"
