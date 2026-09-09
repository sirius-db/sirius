"""Render the guide's data slots as safe, useful HTML when JavaScript is absent.

The interactive page replaces these slots at startup.  Keeping the fallback here
also gives the generated guide meaningful content under a restrictive CSP, in a
text browser, and in the print-to-PDF path.
"""

from __future__ import annotations

import re
from html import escape, unescape
from typing import Any


def _text(value: Any) -> str:
    text = str(value if value is not None else "").translate(
        {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}
    )
    return escape(text, quote=True)


def _description(value: str) -> str:
    """Preserve readable text from the curated inline code markup."""
    return _text(unescape(re.sub(r"</?(?:code|strong|em)>", "", value)))


def _source(data: dict[str, Any], item: list[Any]) -> str:
    path, line, label = item
    href = f"https://github.com/aocsa/sirius/blob/{data['head']}/{path}#L{line}"
    return (
        f'<a href="{_text(href)}" target="_blank" rel="noopener">{_text(label)} ↗</a>'
    )


def _sources(data: dict[str, Any], items: list[list[Any]]) -> str:
    return (
        '<div class="source-links">'
        + "".join(_source(data, item) for item in items)
        + "</div>"
    )


def _pr(number: int) -> str:
    href = f"https://github.com/sirius-db/sirius/pull/{number}"
    return f'<a href="{href}" target="_blank" rel="noopener">#{number}</a>'


def _commit(sha: str) -> str:
    href = f"https://github.com/aocsa/sirius/commit/{sha}"
    return f'<a href="{href}" target="_blank" rel="noopener"><code>{_text(sha[:8])}</code></a>'


def _details(identifier: str, summary: str, body: str, classes: str = "package") -> str:
    return f'<details class="{classes}" id="{_text(identifier)}"><summary>{summary}</summary><div class="pr-body">{body}</div></details>'


def render_slots(data: dict[str, Any]) -> dict[str, str]:
    """Return HTML for every initially empty dynamic container in the template."""
    architecture = data["architecture"]
    stages = data["stages"]
    commits = data["commits"]
    prs = data["prs"]
    packages = data["packages"]

    flow = "".join(
        f'<a class="flow-node" href="#architecture-node-{i}"><span>0{i + 1} / {_text(node["language"])}</span>'
        f'<strong>{_text(node["title"])}</strong><span>{_text(node["short"])}</span></a>'
        for i, node in enumerate(architecture)
    )
    architecture_detail = "".join(
        f'<article id="architecture-node-{i}"><div class="detail-head"><span class="step-number">0{i + 1}</span>'
        f'<h3>{_text(node["title"])}</h3></div><p>{_description(node["description"])}</p>{_sources(data, node["sources"])}</article>'
        for i, node in enumerate(architecture)
    )
    foundations = (
        '<div class="table-wrap"><table><thead><tr><th>Landed capability</th><th>PR</th><th>Why the new stack needs it</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{_text(item["title"])}</td><td>{_pr(item["pr"])} <span class="pill landed">merged</span></td>'
            f'<td>{_text(item["why"])}</td></tr>'
            for item in data["foundations"]
        )
        + "</tbody></table></div>"
    )

    stage_buttons = "".join(
        f'<a class="btn" href="#stage-{i}" aria-label="Step {i + 1}: {_text(stage["title"])}">{i + 1}</a>'
        for i, stage in enumerate(stages)
    )
    zone_names = ("Sender pool", "Sender arena", "Receiver arena", "Receiver pool")
    zone_notes = (
        "Parked cuDF output",
        "Registered device region",
        "Registered device region",
        "Copied input table",
    )
    first = stages[0]
    memory_zones = "".join(
        f'<div class="memory-zone {"active" if first["live"][i] else ""}"><div class="eyebrow">{"CN A" if i < 2 else "CN B"}</div>'
        f'<h4>{name}</h4><span>{note}</span><span class="token">{_text(first["live"][i] or "No frame here")}</span></div>'
        for i, (name, note) in enumerate(zip(zone_names, zone_notes))
    )
    stage_detail = "".join(
        f'<article class="step-detail" id="stage-{i}"><div class="eyebrow">Step {i + 1} of {len(stages)}</div>'
        f'<h3>{_text(stage["title"])}</h3><p>{_description(stage["description"])}</p>{_sources(data, stage["sources"])}</article>'
        for i, stage in enumerate(stages)
    )
    total = 48512 / 1024
    remaining = total - 40 - 4
    memory_bar = (
        f'<span class="pool" style="width:{100 * 40 / total}%">40 GiB</span>'
        f'<span class="arena" style="width:{100 * 4 / total}%">4 GiB</span>'
        f'<span class="headroom" style="width:{100 * remaining / total}%">{remaining:.1f}</span>'
    )
    memory_detail = (
        f"<strong>Per CN: 44 GiB explicitly budgeted; {remaining:.2f} GiB remains before overhead.</strong><br>"
        "2 CNs × 40 GiB host tier = 80 GiB host budget; at the notes' 44 GiB ceiling, 88 GiB. "
        "Against the documented 124 GiB host, that leaves 36 GiB before FE, OS and other processes. "
        "Each CN must fit its own MIG instance."
    )

    group_order = ("Engine & staging", "Scan", "FFI", "Translator", "Compute node")
    atlas = ""
    for group in group_order:
        group_prs = [item for item in prs if item["group"] == group]
        cards = "".join(
            _details(
                f'pr-{item["number"]}',
                f'<span class="pill draft">Draft</span> <span class="pr-title">#{item["number"]} · {_text(item["feature"])}</span>'
                f'<span class="pr-subtitle">{_text(item["summary"])}</span>',
                f'<p>{_text(item["description"])}</p><p class="small"><strong>Git base:</strong> <code>{_text(item["base"])}</code><br>'
                f'<strong>Head:</strong> <code>{_text(item["headRef"])}</code> · {_commit(item["headSha"])}<br>'
                f'<strong>Checks:</strong> {_text(item["checks"])}<br><strong>Review:</strong> {_text(item["review"])}<br>'
                f'<strong>Merge state at snapshot:</strong> {_text(item.get("mergeState") or "not reported")}</p>'
                f'<p class="small"><strong>Gate:</strong> {_text(item["gate"])}</p>'
                f'<div class="label-list">{"".join(f"<span class=\"label\">suggested: {_text(label)}</span>" for label in item["labels"])}</div>'
                f'<div class="source-links">{_pr(item["number"])}{"".join(_source(data, source) for source in item["sources"])}</div>',
                "pr-card",
            )
            for item in group_prs
        )
        atlas += f'<div class="stack-group"><div class="eyebrow">{_text(group)}</div><div class="pr-list">{cards}</div></div>'

    dependency_lanes = (
        "".join(
            f'<div class="row"><strong class="small">{_text(lane["name"])}</strong><div class="lineage">'
            + '<span class="arrow">→</span>'.join(_pr(number) for number in lane["prs"])
            + "</div></div>"
            for lane in data["lanes"]
        )
        + '<p class="small muted">Arrows above are verified PR git-base relationships. Additional prerequisites: #1704\'s expression fixes for Q1; #1693 + #1694 for packed FFI; #1705–#1707 for distributed CN bring-up. These are semantic dependencies across stacks.</p>'
    )
    review_waves = "".join(
        f'<div class="wave"><div class="wave-num">0{i + 1}</div><div><h3>{_text(wave["title"])}</h3>'
        f'<p>{" · ".join(_pr(number) for number in wave["prs"])}</p><p class="small muted">{_text(wave["why"])}</p></div></div>'
        for i, wave in enumerate(data["waves"])
    )
    created_prs = "".join(
        f'<div class="package"><span class="pill draft">New draft</span><h3>{_pr(item["number"])} · {_text(item["title"])}</h3>'
        f'<p>{_text(item["description"])}</p><div class="row small"><span class="muted">Source commits from benchmark branch:</span> '
        f'{" ".join(_commit(sha) for sha in item["commits"])}<span class="muted">Base: {_text(item.get("base") or "dev")}</span></div>'
        f'<p class="small muted">Validation: {_text(item["validation"])}</p></div>'
        for item in data["created"]
    )
    new_packages = "".join(
        _details(
            f'package-{item["id"]}',
            f'<span class="pill {"draft" if item.get("pr") else "proposal"}">{_text(item["id"])}'
            f'{" · Draft #" + _text(item["pr"]) if item.get("pr") else ""}</span> <span class="pr-title">{_text(item["title"])}</span>',
            f'<p>{_text(item["description"])}</p><p class="small"><strong>Prerequisites:</strong> {_text(item["dependencies"])}<br>'
            f'<strong>Validation gate:</strong> {_text(item["gate"])}</p><div class="row">{" ".join(_commit(sha) for sha in item["commits"])}</div>'
            f'<div class="label-list">{"".join(f"<span class=\"label\">{_text(label)}</span>" for label in item["labels"])}</div>'
            f'<p class="small"><a href="pr-packages/{_text(item["id"].lower())}.md">Prepared PR description / extraction notes ↗</a></p>',
        )
        for item in packages
    )
    commit_rows = "".join(
        f'<tr><td>{_commit(item["sha"])}</td><td>{_text(item["subject"])}</td><td>'
        f'{_pr(item["pr"]) if item.get("pr") else _text(item["package"])}</td></tr>'
        for item in commits
    )
    coverage = (
        f'<strong>{len(commits) + len(data["merges"])} branch-only commits = {len(commits)} feature/test/docs commits + '
        f'{len(data["merges"])} integration merges.</strong><br>{sum(bool(item.get("pr")) for item in commits)} feature commits are covered by the original drafts; '
        f'{sum(not item.get("pr") for item in commits)} are assigned to new PRs or extraction packages. Baseline: current upstream <code>{_text(data["base"][:8])}</code>; '
        "diff scope is the branch's merge base."
    )
    merge_details = (
        f'<p>{_text(data["mappingMethod"])}</p><ul>'
        + "".join(
            f'<li>{_commit(item["sha"])} {_text(item["subject"])}</li>'
            for item in data["merges"]
        )
        + "</ul>"
    )
    findings = "".join(
        f'<article class="finding"><div class="severity">{_text(item["priority"])} · {_text(item["kind"])}</div>'
        f'<h3>{_text(item["title"])}</h3><p>{_text(item["description"])}</p><div class="evidence"><p class="small">'
        f'<strong>Trigger / evidence:</strong> {_text(item["evidence"])}</p><p class="small"><strong>Fix / landing gate:</strong> {_text(item["fix"])}</p></div>'
        f'{_sources(data, item["sources"])}</article>'
        for item in data["findings"]
    )
    validation = (
        '<div class="table-wrap"><table><thead><tr><th>Check</th><th>Result</th><th>What it establishes</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{_text(item["name"])}</td><td>{_text(item["result"])}</td><td>{_text(item["meaning"])}</td></tr>'
            for item in data["validation"]
        )
        + "</tbody></table></div>"
    )
    reading = (
        '<div class="table-wrap"><table><thead><tr><th>Area</th><th>Start here</th><th>Then inspect</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{_text(item["area"])}</td><td>{"<br>".join(_source(data, source) for source in item["start"])}</td>'
            f'<td>{"<br>".join(_source(data, source) for source in item["next"])}</td></tr>'
            for item in data["reading"]
        )
        + "</tbody></table></div>"
    )

    return {
        "architecture-flow": flow,
        "architecture-detail": architecture_detail,
        "foundations": foundations,
        "stage-buttons": stage_buttons,
        "memory-zones": memory_zones,
        "stage-detail": stage_detail,
        "memory-bar": memory_bar,
        "memory-detail": memory_detail,
        "staging-sources": "".join(
            _source(data, item) for item in data["stagingSources"]
        ),
        "pr-count": f"{len(prs)} of {len(prs)} original draft PRs",
        "pr-atlas": atlas,
        "available-labels": " ".join(
            f'<span class="label">{_text(label)}</span>' for label in data["labels"]
        ),
        "dependency-lanes": dependency_lanes,
        "review-waves": review_waves,
        "created-prs": created_prs,
        "new-packages": new_packages,
        "coverage-summary": coverage,
        "commit-result": f"{len(commits)} of {len(commits)} non-merge commits",
        "commit-rows": commit_rows,
        "merge-details": merge_details,
        "findings-list": findings,
        "validation-record": validation,
        "reading-map": reading,
    }
