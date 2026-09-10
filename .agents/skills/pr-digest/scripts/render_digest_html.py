#!/usr/bin/env python3
"""Render a PR digest Markdown file as a self-contained HTML preview.

The renderer intentionally supports the Markdown subset emitted by the pr-digest skill. It has
no third-party dependencies, colors diff additions/removals, and turns simple Mermaid flowcharts
into inline SVG so the preview works offline in Codex.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

FENCE_RE = re.compile(r"^```([A-Za-z0-9_-]*)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^(\s*)([-+*]|\d+\.)\s+(.+)$")
TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_RE = re.compile(r"`([^`]+)`")
NODE_RE = re.compile(
    r'^([A-Za-z_][\w-]*)(?:\["([^"]*)"\]|\[([^\]]*)\]|\{"([^"]*)"\}|\{([^}]*)\}|\("([^"]*)"\)|\(([^)]*)\))?$'
)
EDGE_RE = re.compile(r"^(.*?)\s*(-->|==>)\s*(?:\|\"?([^|\"]+)\"?\|\s*)?(.*?)$")
SUBGRAPH_RE = re.compile(
    r'^subgraph\s+([A-Za-z_][\w-]*)(?:\["([^"]*)"\]|\[([^\]]*)\])?\s*$'
)
SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


def safe_href(value: str) -> str:
    """Escape a link target and reject schemes that can execute in a local preview."""
    target = value.strip()
    scheme = SCHEME_RE.match(target)
    if scheme and scheme.group(1).lower() not in {"http", "https", "mailto"}:
        target = "#"
    return html.escape(target, quote=True)


def render_inline(value: str) -> str:
    placeholders: list[str] = []

    def hold(rendered: str) -> str:
        token = f"\x00{len(placeholders)}\x00"
        placeholders.append(rendered)
        return token

    def link(match: re.Match[str]) -> str:
        label = render_inline(match.group(1))
        href = safe_href(match.group(2))
        return hold(f'<a href="{href}">{label}</a>')

    value = LINK_RE.sub(link, value)
    value = CODE_RE.sub(
        lambda match: hold(f"<code>{html.escape(match.group(1))}</code>"), value
    )
    value = html.escape(value, quote=False)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    value = re.sub(r"~~(.+?)~~", r"<del>\1</del>", value)
    for index, rendered in enumerate(placeholders):
        value = value.replace(f"\x00{index}\x00", rendered)
    return value


def slugify(value: str, counts: dict[str, int]) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", "", value)).lower()
    base = re.sub(r"[^\w\s-]", "", plain, flags=re.UNICODE).strip()
    # GitHub's heading IDs preserve the two spaces around an em dash as two hyphens after the
    # punctuation is removed. Match that behavior so authored table-of-contents links still work.
    base = re.sub(r"\s", "-", base) or "section"
    seen = counts.get(base, 0)
    counts[base] = seen + 1
    return base if seen == 0 else f"{base}-{seen}"


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


@dataclass
class FlowNode:
    identifier: str
    label: str
    shape: str = "rect"
    group: str | None = None


@dataclass
class FlowEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class FlowGraph:
    direction: str = "TD"
    nodes: dict[str, FlowNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)

    def add_node(self, token: str, group: str | None) -> str | None:
        token = token.strip().rstrip(";")
        match = NODE_RE.match(token)
        if not match:
            return None
        identifier = match.group(1)
        candidates = match.groups()[1:]
        label = next(
            (candidate for candidate in candidates if candidate is not None), identifier
        )
        shape = (
            "decision" if token[len(identifier) :].lstrip().startswith("{") else "rect"
        )
        if identifier not in self.nodes:
            self.nodes[identifier] = FlowNode(identifier, label, shape, group)
            self.order.append(identifier)
        else:
            node = self.nodes[identifier]
            if label != identifier:
                node.label = label
                node.shape = shape
            if node.group is None:
                node.group = group
        return identifier


def parse_flowchart(source: str) -> FlowGraph | None:
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("flowchart "):
        return None
    graph = FlowGraph(direction=lines[0].split(maxsplit=1)[1].upper())
    current_group: str | None = None
    for raw in lines[1:]:
        if raw.startswith("%%") or raw.startswith(
            ("classDef ", "class ", "style ", "linkStyle ")
        ):
            continue
        group_match = SUBGRAPH_RE.match(raw)
        if group_match:
            current_group = group_match.group(1)
            graph.groups[current_group] = (
                group_match.group(2) or group_match.group(3) or current_group
            )
            continue
        if raw == "end":
            current_group = None
            continue

        normalized = re.sub(
            r'-\.\s*"([^"]+)"\s*\.->',
            lambda match: f'-->|"{match.group(1)}"|',
            raw,
        )
        edge_match = EDGE_RE.match(normalized)
        if edge_match:
            source_id = graph.add_node(edge_match.group(1), current_group)
            target_id = graph.add_node(edge_match.group(4), current_group)
            if source_id and target_id:
                graph.edges.append(
                    FlowEdge(source_id, target_id, (edge_match.group(3) or "").strip())
                )
            continue
        graph.add_node(raw, current_group)
    return graph if graph.nodes else None


def wrap_svg_text(value: str, width: int = 28) -> list[str]:
    words = value.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:3]


def render_flowchart(source: str) -> str:
    graph = parse_flowchart(source)
    escaped_source = html.escape(source)
    if graph is None:
        return (
            '<figure class="flow-figure"><p class="diagram-warning">'
            "Diagram preview unavailable for this Mermaid syntax; source retained below.</p>"
            f"<pre><code>{escaped_source}</code></pre></figure>"
        )

    indegree = {identifier: 0 for identifier in graph.nodes}
    outgoing: dict[str, list[str]] = {identifier: [] for identifier in graph.nodes}
    for edge in graph.edges:
        outgoing[edge.source].append(edge.target)
        indegree[edge.target] += 1

    levels = {identifier: 0 for identifier in graph.nodes}
    queue = [identifier for identifier in graph.order if indegree[identifier] == 0]
    visited: list[str] = []
    while queue:
        identifier = queue.pop(0)
        visited.append(identifier)
        for target in outgoing[identifier]:
            levels[target] = max(levels[target], levels[identifier] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    for identifier in graph.order:
        if identifier not in visited:
            levels[identifier] = max(levels.values(), default=0) + 1

    layers: dict[int, list[str]] = {}
    for identifier in graph.order:
        layers.setdefault(levels[identifier], []).append(identifier)
    max_level = max(layers, default=0)
    max_layer_size = max((len(items) for items in layers.values()), default=1)
    horizontal = graph.direction in {"LR", "RL"}

    if horizontal:
        width = max(720, 300 * (max_level + 1))
        height = max(360, 125 * max_layer_size + 100)
    else:
        width = max(760, 280 * max_layer_size + 100)
        height = max(420, 135 * (max_level + 1) + 100)

    positions: dict[str, tuple[float, float]] = {}
    for level, identifiers in layers.items():
        if horizontal:
            x = 150 + level * 280
            for index, identifier in enumerate(identifiers):
                y = (index + 1) * height / (len(identifiers) + 1)
                positions[identifier] = (x, y)
        else:
            y = 80 + level * 130
            for index, identifier in enumerate(identifiers):
                x = (index + 1) * width / (len(identifiers) + 1)
                positions[identifier] = (x, y)

    token = hashlib.sha1(source.encode("utf8"), usedforsecurity=False).hexdigest()[:10]
    pieces = [
        f'<figure class="flow-figure"><svg class="flow-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-{token} desc-{token}">',
        f'<title id="title-{token}">PR change flowchart</title>',
        f'<desc id="desc-{token}">Flowchart generated from the digest Mermaid source.</desc>',
        f'<defs><marker id="arrow-{token}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head" /></marker></defs>',
    ]

    for group_id, title in graph.groups.items():
        members = [
            positions[node.identifier]
            for node in graph.nodes.values()
            if node.group == group_id
        ]
        if not members:
            continue
        min_x = min(point[0] for point in members) - 140
        max_x = max(point[0] for point in members) + 140
        min_y = min(point[1] for point in members) - 65
        max_y = max(point[1] for point in members) + 65
        pieces.append(
            f'<rect class="flow-group" x="{min_x:.1f}" y="{min_y:.1f}" width="{max_x - min_x:.1f}" height="{max_y - min_y:.1f}" rx="16" />'
        )
        pieces.append(
            f'<text class="flow-group-label" x="{min_x + 14:.1f}" y="{min_y + 24:.1f}">{html.escape(title)}</text>'
        )

    for edge in graph.edges:
        source_x, source_y = positions[edge.source]
        target_x, target_y = positions[edge.target]
        if horizontal:
            start_x, start_y = source_x + 120, source_y
            end_x, end_y = target_x - 120, target_y
        else:
            start_x, start_y = source_x, source_y + 38
            end_x, end_y = target_x, target_y - 38
        pieces.append(
            f'<path class="flow-edge" marker-end="url(#arrow-{token})" d="M {start_x:.1f} {start_y:.1f} L {end_x:.1f} {end_y:.1f}" />'
        )
        if edge.label:
            label_x = (start_x + end_x) / 2
            label_y = (start_y + end_y) / 2 - 7
            pieces.append(
                f'<text class="flow-edge-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(edge.label)}</text>'
            )

    for identifier in graph.order:
        node = graph.nodes[identifier]
        x, y = positions[identifier]
        if node.shape == "decision":
            points = f"{x:.1f},{y - 42:.1f} {x + 130:.1f},{y:.1f} {x:.1f},{y + 42:.1f} {x - 130:.1f},{y:.1f}"
            pieces.append(
                f'<polygon class="flow-node flow-decision" points="{points}" />'
            )
        else:
            pieces.append(
                f'<rect class="flow-node" x="{x - 120:.1f}" y="{y - 34:.1f}" width="240" height="68" rx="12" />'
            )
        text_lines = wrap_svg_text(node.label)
        start_y = y - (len(text_lines) - 1) * 10
        pieces.append(f'<text class="flow-text" x="{x:.1f}" y="{start_y:.1f}">')
        for index, line in enumerate(text_lines):
            dy = "0" if index == 0 else "20"
            pieces.append(f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>')
        pieces.append("</text>")

    pieces.extend(
        [
            "</svg>",
            '<details class="diagram-source"><summary>Mermaid source</summary>',
            f"<pre><code>{escaped_source}</code></pre></details></figure>",
        ]
    )
    return "".join(pieces)


class DigestRenderer:
    def __init__(self) -> None:
        self.heading_counts: dict[str, int] = {}
        self.diff_blocks = 0
        self.additions = 0
        self.removals = 0
        self.flowcharts = 0

    def render_code(self, language: str, code: str) -> str:
        if language == "diff":
            self.diff_blocks += 1
            rendered: list[str] = []
            for line in code.splitlines():
                kind = "context"
                if line.startswith("+") and not line.startswith("+++"):
                    kind = "add"
                    self.additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    kind = "remove"
                    self.removals += 1
                rendered.append(
                    f'<span class="diff-line diff-{kind}">{html.escape(line) or " "}</span>'
                )
            return (
                '<div class="diff-block" role="region" aria-label="Code diff">'
                f'<pre><code>{"".join(rendered)}</code></pre></div>'
            )
        if language == "mermaid":
            self.flowcharts += 1
            return render_flowchart(code)
        language_class = (
            f' class="language-{html.escape(language, quote=True)}"' if language else ""
        )
        return f"<pre><code{language_class}>{html.escape(code)}</code></pre>"

    def render(self, markdown: str) -> str:
        lines = markdown.splitlines()
        output: list[str] = []
        paragraph: list[str] = []
        index = 0

        def flush_paragraph() -> None:
            if paragraph:
                output.append(
                    f'<p>{render_inline(" ".join(part.strip() for part in paragraph))}</p>'
                )
                paragraph.clear()

        while index < len(lines):
            line = lines[index]
            fence_match = FENCE_RE.match(line)
            if fence_match:
                flush_paragraph()
                language = fence_match.group(1).lower()
                index += 1
                code_lines: list[str] = []
                while index < len(lines) and lines[index] != "```":
                    code_lines.append(lines[index])
                    index += 1
                output.append(self.render_code(language, "\n".join(code_lines)))
                index += 1
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush_paragraph()
                level = len(heading_match.group(1))
                rendered = render_inline(heading_match.group(2))
                slug = slugify(rendered, self.heading_counts)
                output.append(f'<h{level} id="{slug}">{rendered}</h{level}>')
                index += 1
                continue

            if re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", line):
                flush_paragraph()
                output.append("<hr />")
                index += 1
                continue

            if line.startswith(">"):
                flush_paragraph()
                quote_lines: list[str] = []
                while index < len(lines) and lines[index].startswith(">"):
                    quote_lines.append(lines[index][1:].lstrip())
                    index += 1
                quote_paragraphs: list[str] = []
                current: list[str] = []
                for quote_line in quote_lines + [""]:
                    if quote_line:
                        current.append(quote_line)
                    elif current:
                        quote_paragraphs.append(
                            f'<p>{render_inline(" ".join(current))}</p>'
                        )
                        current = []
                output.append(f'<blockquote>{"".join(quote_paragraphs)}</blockquote>')
                continue

            if (
                index + 1 < len(lines)
                and "|" in line
                and TABLE_RULE_RE.match(lines[index + 1])
            ):
                flush_paragraph()
                headers = split_table_row(line)
                index += 2
                rows: list[list[str]] = []
                while (
                    index < len(lines) and "|" in lines[index] and lines[index].strip()
                ):
                    rows.append(split_table_row(lines[index]))
                    index += 1
                head = "".join(f"<th>{render_inline(cell)}</th>" for cell in headers)
                body = "".join(
                    "<tr>"
                    + "".join(f"<td>{render_inline(cell)}</td>" for cell in row)
                    + "</tr>"
                    for row in rows
                )
                output.append(
                    f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
                )
                continue

            list_match = LIST_RE.match(line)
            if list_match:
                flush_paragraph()
                items: list[tuple[int, str, str]] = []
                while index < len(lines):
                    current_match = LIST_RE.match(lines[index])
                    if not current_match:
                        if (
                            items
                            and lines[index].startswith(" ")
                            and lines[index].strip()
                        ):
                            depth, marker, value = items[-1]
                            items[-1] = (
                                depth,
                                marker,
                                value + " " + lines[index].strip(),
                            )
                            index += 1
                            continue
                        break
                    indent, marker, value = current_match.groups()
                    depth = max(0, len(indent.expandtabs(2)) // 2)
                    items.append((depth, marker, value))
                    index += 1
                rendered_items = []
                for depth, marker, value in items:
                    rendered_items.append(
                        f'<div class="list-item" style="--depth:{depth}"><span class="list-marker">{html.escape(marker)}</span><span>{render_inline(value)}</span></div>'
                    )
                output.append(
                    f'<div class="list-block">{"".join(rendered_items)}</div>'
                )
                continue

            if not line.strip():
                flush_paragraph()
                index += 1
                continue

            paragraph.append(line)
            index += 1

        flush_paragraph()
        return "\n".join(output)


STYLE = r"""
:root {
  --page: #f6f8fa; --surface: #fff; --surface-muted: #f6f8fa; --text: #1f2328;
  --muted: #59636e; --border: #d0d7de; --link: #0969da; --accent: #8250df;
  --add-bg: #dafbe1; --add-text: #116329; --add-border: #aceebb;
  --remove-bg: #ffebe9; --remove-text: #82071e; --remove-border: #ffcecb;
  --shadow: 0 8px 28px rgba(31,35,40,.08);
}
@media (prefers-color-scheme: dark) {
  :root {
    --page:#0d1117; --surface:#161b22; --surface-muted:#0d1117; --text:#e6edf3;
    --muted:#9da7b1; --border:#30363d; --link:#58a6ff; --accent:#a371f7;
    --add-bg:#12261e; --add-text:#7ee787; --add-border:#2ea043;
    --remove-bg:#321c22; --remove-text:#ff7b72; --remove-border:#f85149;
    --shadow:0 10px 36px rgba(0,0,0,.35);
  }
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--page); color:var(--text); font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.toolbar { position:sticky; top:0; z-index:5; display:flex; justify-content:space-between; gap:1rem; align-items:center; padding:.7rem max(1rem,calc((100vw - 1120px)/2)); border-bottom:1px solid var(--border); background:var(--surface); }
.toolbar strong { font-size:.92rem; }
.toolbar nav { display:flex; gap:.9rem; font-size:.9rem; }
main { width:min(1120px,calc(100% - 2rem)); margin:1.5rem auto 4rem; padding:clamp(1.25rem,3.2vw,3rem); background:var(--surface); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); }
h1,h2,h3 { line-height:1.25; scroll-margin-top:5rem; }
h1 { margin-top:0; font-size:clamp(1.85rem,4vw,2.7rem); }
h2 { margin-top:2.4rem; padding-bottom:.35rem; border-bottom:1px solid var(--border); }
h3 { margin-top:2rem; }
a { color:var(--link); text-decoration-thickness:1px; text-underline-offset:.16em; }
blockquote { margin:1.25rem 0; padding:.3rem 1rem; color:var(--muted); border-left:4px solid var(--accent); background:var(--surface-muted); }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
th,td { padding:.65rem .8rem; border:1px solid var(--border); vertical-align:top; }
th { background:var(--surface-muted); text-align:left; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.9em; }
:not(pre)>code { padding:.12em .35em; border-radius:5px; background:var(--surface-muted); }
pre { margin:0; overflow-x:auto; }
hr { margin:3rem 0; border:0; border-top:1px solid var(--border); }
.list-block { margin:.85rem 0; }
.list-item { display:grid; grid-template-columns:2.2rem minmax(0,1fr); margin:.28rem 0; margin-left:calc(var(--depth)*1.55rem); }
.list-marker { color:var(--muted); text-align:right; padding-right:.65rem; }
.diff-block { margin:1rem 0 1.6rem; overflow:hidden; border:1px solid var(--border); border-radius:10px; background:var(--surface-muted); }
.diff-block code { display:block; min-width:max-content; padding:.55rem 0; }
.diff-line { display:block; min-height:1.45em; padding:0 1rem; white-space:pre; }
.diff-add { color:var(--add-text); background:var(--add-bg); border-left:3px solid var(--add-border); }
.diff-remove { color:var(--remove-text); background:var(--remove-bg); border-left:3px solid var(--remove-border); }
.diff-context { color:var(--muted); border-left:3px solid transparent; }
.flow-figure { margin:1rem 0 1.75rem; padding:1rem; overflow-x:auto; border:1px solid var(--border); border-radius:12px; background:var(--surface-muted); }
.flow-svg { display:block; width:100%; height:auto; min-width:680px; }
.flow-group { fill:color-mix(in srgb,var(--surface) 76%,transparent); stroke:var(--border); stroke-width:1.5; stroke-dasharray:5 4; }
.flow-group-label { fill:var(--muted); font:600 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; text-anchor:start; }
.flow-node { fill:var(--surface); stroke:var(--border); stroke-width:2; }
.flow-decision { fill:var(--surface); stroke:var(--accent); }
.flow-edge { fill:none; stroke:var(--muted); stroke-width:2.2; }
.arrow-head { fill:var(--muted); }
.flow-text,.flow-edge-label { fill:var(--text); font:600 15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; text-anchor:middle; }
.flow-edge-label { fill:var(--muted); font-size:13px; }
.diagram-source { margin-top:.9rem; color:var(--muted); }
.diagram-source pre,.diagram-warning+pre { margin-top:.6rem; padding:.8rem; border-radius:8px; background:var(--surface); }
.diagram-warning { color:var(--muted); }
.preview-footer { width:min(1120px,calc(100% - 2rem)); margin:-2.5rem auto 2rem; color:var(--muted); font-size:.85rem; text-align:center; }
@media (max-width:680px) {
  .toolbar { align-items:flex-start; flex-direction:column; }
  main { width:min(100% - 1rem,1120px); margin-top:.5rem; padding:1rem; border-radius:10px; }
  .flow-figure { padding:.4rem; }
}
@media print {
  .toolbar { display:none; }
  body { background:#fff; }
  main { width:100%; margin:0; padding:0; border:0; box-shadow:none; }
  .preview-footer { margin-top:1rem; }
}
"""


def build_document(markdown_path: Path, content: str, title: str) -> str:
    source_name = html.escape(markdown_path.name, quote=True)
    key_changes = re.search(r'id="(key-changes-at-a-glance[^"]*)"', content)
    coverage_notes = re.search(r'id="(not-covered-in-this-pass[^"]*)"', content)
    key_changes_href = f"#{key_changes.group(1)}" if key_changes else "#"
    coverage_notes_href = f"#{coverage_notes.group(1)}" if coverage_notes else "#"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="light dark" />
  <title>{html.escape(title)} — HTML Preview</title>
  <style>{STYLE}</style>
</head>
<body>
  <header class="toolbar">
    <strong>Rendered PR digest</strong>
    <nav>
      <a href="{source_name}">Markdown source</a>
      <a href="{key_changes_href}">Key changes</a>
      <a href="{coverage_notes_href}">Coverage notes</a>
    </nav>
  </header>
  <main>{content}</main>
  <footer class="preview-footer">Generated from <code>{source_name}</code>. Regenerate after editing the Markdown source.</footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, help="source PR digest Markdown path")
    parser.add_argument("html", type=Path, help="destination HTML preview path")
    args = parser.parse_args()

    markdown = args.markdown.read_text(encoding="utf8")
    renderer = DigestRenderer()
    content = renderer.render(markdown)
    title_match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    title = title_match.group(1) if title_match else args.markdown.stem
    document = build_document(args.markdown, content, title)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(document, encoding="utf8")
    print(
        f"Rendered {args.html}: {renderer.diff_blocks} diff blocks, "
        f"{renderer.additions} additions, {renderer.removals} removals, "
        f"{renderer.flowcharts} flowcharts"
    )


if __name__ == "__main__":
    main()
