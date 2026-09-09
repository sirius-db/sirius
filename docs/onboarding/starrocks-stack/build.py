#!/usr/bin/env python3
"""Build the offline guide from its checked-in, reviewed snapshot (no network)."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "guide-data.json").read_text())
    assert len(data["prs"]) == 19, "The original draft inventory must remain complete"
    assert (
        len(data["commits"]) == 59
    ), "The reviewed source range has 59 non-merge commits"
    assert (
        len(data["merges"]) == 21
    ), "The reviewed source range has 21 integration merges"
    assert all(c["pr"] or c["package"] for c in data["commits"]), "Unassigned commit"
    assert len({c["sha"] for c in data["commits"]}) == 59, "Duplicate commit"
    assert len({p["id"] for p in data["packages"]}) == len(data["packages"])
    for package in data["packages"]:
        assert (root / "pr-packages" / f"{package['id'].lower()}.md").is_file()
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    # Keep JSON inside its inert script element even if future source text has HTML.
    encoded = (
        encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    template = (root / "index.template.html").read_text()
    assert template.count("__GUIDE_DATA__") == 1
    page = template.replace("__GUIDE_DATA__", encoded)
    (root / "index.html").write_text(page)
    print(f"Built {root / 'index.html'} ({len(page.encode()):,} bytes)")


if __name__ == "__main__":
    main()
