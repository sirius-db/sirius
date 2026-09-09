#!/usr/bin/env python3
"""Build the offline guide from its checked-in, reviewed snapshot (no network)."""

import json
import re
from html import escape, unescape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from render_static import render_slots


def reference_id(path):
    return "reference-" + re.sub(r"[^a-zA-Z0-9_-]", "-", path)


def build_references(root, data):
    """Bundle linked research so a copied HTML file never needs sibling files."""
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".json", ".py"}
        and "__pycache__" not in path.parts
    )
    paths.append(root / "index.template.html")
    documents = {path.relative_to(root).as_posix(): path.read_text() for path in paths}

    def href(value, origin="index.html"):
        parsed = urlsplit(unescape(value))
        if parsed.scheme or parsed.netloc:
            return (
                value if parsed.scheme in {"https", "http", "mailto"} else "#references"
            )
        if not parsed.path:
            # Markdown heading links open their source document; browser chapters
            # and exact deep links retain their own native fragment anchors.
            return value if origin == "index.html" else "#" + reference_id(origin)
        target = ((root / origin).parent / unquote(parsed.path)).resolve()
        if target == root / "index.html":
            return "#" + (parsed.fragment or "overview")
        if target.is_dir() and target.is_relative_to(root):
            return "#references"
        if target.is_relative_to(root):
            name = target.relative_to(root).as_posix()
            if name in documents:
                return "#" + reference_id(name)
            if target.is_file():
                return (
                    "https://github.com/aocsa/sirius/blob/codex/starrocks-stack-onboarding/docs/onboarding/starrocks-stack/"
                    + quote(name)
                )
        # Links to project source outside the bundle use the immutable review tip.
        repo = root.parents[2]
        if target.is_relative_to(repo) and target.is_file():
            path = target.relative_to(repo).as_posix()
            return (
                f"https://github.com/aocsa/sirius/blob/{data['head']}/{quote(path)}"
                + ("#" + parsed.fragment if parsed.fragment else "")
            )
        raise ValueError(f"Broken local link in {origin}: {value}")

    def inline(text, origin):
        # Escape raw HTML. Preserve only the small Markdown vocabulary used by
        # these source notes; code and link targets never become executable HTML.
        token = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^\s)]+)\)|(https?://[^\s<>]+)")
        result, end = [], 0
        for match in token.finditer(text):
            result.append(escape(text[end : match.start()]))
            code, label, target, url = match.groups()
            if code is not None:
                result.append("<code>" + escape(code) + "</code>")
            else:
                target = target or url.rstrip(".,;")
                dest = href(target, origin)
                result.append(
                    f'<a href="{escape(dest, quote=True)}">{escape(label or target)}</a>'
                )
            end = match.end()
        result.append(escape(text[end:]))
        return "".join(result)

    def markdown(text, origin):
        rendered, paragraph, code, fence = [], [], [], None

        def flush():
            if paragraph:
                rendered.append("<p>" + inline(" ".join(paragraph), origin) + "</p>")
                paragraph.clear()

        for line in text.splitlines():
            if line.startswith(("```", "~~~")):
                flush()
                if fence:
                    rendered.append(
                        "<pre><code>" + escape("\n".join(code)) + "</code></pre>"
                    )
                    code, fence = [], None
                else:
                    fence = line[:3]
            elif fence:
                code.append(line)
            elif re.match(r"^#{1,6} ", line):
                flush()
                level = min(len(line) - len(line.lstrip("#")) + 1, 6)
                rendered.append(
                    f"<h{level}>" + inline(line.lstrip("# "), origin) + f"</h{level}>"
                )
            elif not line.strip():
                flush()
            elif re.match(r"^\s*(?:[-*]|\d+\.) ", line) or line.startswith("|"):
                flush()
                rendered.append("<p>" + inline(line, origin) + "</p>")
            else:
                paragraph.append(line)
        flush()
        if code:
            rendered.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
        return "\n".join(rendered)

    sections = []
    for name, contents in documents.items():
        if name in {
            "research/github-pr-snapshot.json",
            "guide-data.json",
            "index.template.html",
            "build.py",
            "render_static.py",
        }:
            # Keep the all-in-one reading guide below the repository's 500 KiB
            # file limit. These full developer artifacts also live in the PR.
            url = (
                "https://github.com/aocsa/sirius/blob/codex/starrocks-stack-onboarding/docs/onboarding/starrocks-stack/"
                + quote(name)
            )
            body = "<p>The complete source artifact is published with the guide.</p>" + (
                f'<p><a href="{url}">View {escape(name)} on GitHub</a> (internet connection required).</p>'
            )
        else:
            if name.endswith(".json"):
                # Keep evidence complete without duplicating its source-file
                # indentation in the portable HTML reader.
                contents = json.dumps(json.loads(contents), separators=(",", ":"))
            body = (
                markdown(contents, name)
                if name.endswith(".md")
                else "<pre><code>" + escape(contents, quote=False) + "</code></pre>"
            )
        sections.append(
            f'<details class="reference-document" id="{reference_id(name)}">'
            f'<summary>{escape(name)}</summary><div class="reference-body">{body}</div></details>'
        )
    return "\n".join(sections), href


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
    for identifier, contents in render_slots(data).items():
        pattern = re.compile(
            rf'(<(?P<tag>\w+)\b[^>]*\bid="{re.escape(identifier)}"[^>]*>)\s*(</(?P=tag)>)'
        )
        page, count = pattern.subn(lambda m: m[1] + contents + m[3], page)
        assert count == 1, f"Missing or duplicate static slot: {identifier}"
    references, resolve_href = build_references(root, data)
    # Rewrite the guide's relative links before injecting source text. References
    # render their own links relative to their originating Markdown file.
    page = re.sub(
        r'href="([^"]*)"',
        lambda m: 'href="' + escape(resolve_href(m[1]), quote=True) + '"',
        page,
    )
    assert page.count("__REFERENCE_DOCUMENTS__") == 1
    page = page.replace("__REFERENCE_DOCUMENTS__", references)
    (root / "index.html").write_text(page)
    print(f"Built {root / 'index.html'} ({len(page.encode()):,} bytes)")


if __name__ == "__main__":
    main()
