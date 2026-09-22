"""Generate the public API reference without building Sirius."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build/docs"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    header = OUTPUT / "header.html"
    subprocess.run(
        [
            "doxygen",
            "-w",
            "html",
            str(header),
            str(OUTPUT / "footer.html"),
            str(OUTPUT / "doxygen.css"),
        ],
        cwd=ROOT,
        check=True,
    )
    template = header.read_text()
    template = template.replace(
        "</head>", (ROOT / "docs/api/head.html").read_text() + "</head>"
    )
    template = template.replace(
        '<img alt="Logo" src="$relpath^$projectlogo"$logosize/>',
        '<img class="sirius-logo-light" alt="Sirius" '
        'src="$relpath^$projectlogo"$logosize/>'
        '<img class="sirius-logo-dark" alt="Sirius" '
        'src="$relpath^sirius_dark_full.svg"$logosize/>',
    )
    header.write_text(template)
    # Removed headers must not leave stale API pages in a subsequent build.
    shutil.rmtree(OUTPUT / "html", ignore_errors=True)
    subprocess.run(["doxygen", "docs/api/Doxyfile"], cwd=ROOT, check=True)
    print(f"API reference: {OUTPUT / 'html/index.html'}")


if __name__ == "__main__":
    main()
