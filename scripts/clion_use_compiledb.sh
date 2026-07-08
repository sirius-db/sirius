#!/usr/bin/env bash
# Switch the CLion project metadata from a (broken) CMake project to a
# compilation-database project, so CLion indexes Sirius via
# build/release/compile_commands.json instead of trying to configure the
# repo-root CMakeLists.txt. See docs/clion.md for the full tutorial.
#
# IMPORTANT: CLion must have this project CLOSED when you run this
# (File -> Close Project, back to the Welcome screen). A running CLion
# re-writes .idea/workspace.xml from its in-memory CMake model and will
# silently undo these edits.
#
# After running: on the CLion Welcome screen choose "Open" and select
#   <repo>/compile_commands.json   (the file, not the folder)
# then "Open as Project". CLion loads it as a Compilation Database project.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
idea="$repo/.idea"
db="$repo/build/release/compile_commands.json"

# 0. Refuse to run while CLion holds the project (avoids a clobber-loop).
if pgrep -f "clion/bin/clion" >/dev/null 2>&1; then
  echo "!! CLion appears to be running. Close the project (File -> Close Project) first," >&2
  echo "!! then re-run this script. Editing .idea under a live CLion will not stick." >&2
  exit 1
fi

# 1. The compilation database comes from the pixi release build.
if [ ! -f "$db" ]; then
  echo "!! $db not found - run \`pixi run make\` first to configure/build." >&2
  exit 1
fi

# 2. Root compile_commands.json symlink (gitignored; auto-fresh on rebuild).
ln -sfn build/release/compile_commands.json "$repo/compile_commands.json"

# 3. Strip every CMake* project-model component from .idea so CLion stops
#    loading the project as CMake. Harmless if .idea does not exist yet.
python3 - "$idea/workspace.xml" "$idea/misc.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
for path in sys.argv[1:]:
    try:
        tree = ET.parse(path)
    except (FileNotFoundError, ET.ParseError):
        continue
    root = tree.getroot()
    removed = []
    for comp in list(root.findall("component")):
        name = comp.get("name", "")
        if name.startswith("CMake") and name != "CMakePythonSetting":
            root.remove(comp); removed.append(name)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    print(f"{path}: removed {removed or 'nothing'}")
PY

echo "OK. Now in CLion: Welcome screen -> Open -> select compile_commands.json -> Open as Project"
