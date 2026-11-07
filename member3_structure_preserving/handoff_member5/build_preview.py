"""Compile the four report drafts and render their PDF for visual QA.

The preview uses standard article fonts; the group report keeps its IEEE style.
No automatic TeX package installation or shell escape is enabled.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    build = root / "tmp/pdfs/member3_handoff"
    final = root / "output/pdf/member3_handoff_preview.pdf"
    build.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "references.bib", build / "references.bib")

    def command(argv, cwd, log):
        result = subprocess.run(argv, cwd=cwd, capture_output=True)
        (build / log).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed: {argv[0]}; see {build / log}")

    latex = ["pdflatex", "--disable-installer", "-interaction=nonstopmode", "-halt-on-error",
             "-no-shell-escape", f"-output-directory={build}", "handoff_preview.tex"]
    command(latex, source, "compile-1.txt")
    if not (build / "handoff_preview.aux").is_file():
        raise RuntimeError("Expected auxiliary file not produced")
    command(["bibtex", "--disable-installer", "handoff_preview"], build, "bibliography.txt")
    if not (build / "handoff_preview.bbl").is_file():
        raise RuntimeError("Bibliography did not compile")
    command(latex, source, "compile-2.txt")
    command(latex, source, "compile-3.txt")
    log = (build / "handoff_preview.log").read_text(encoding="utf-8", errors="replace")
    if "There were undefined references" in log or "undefined on input line" in log:
        raise RuntimeError("Unresolved citations or cross-references")
    if "Overfull" in log:
        raise RuntimeError("Overfull boxes require layout review")
    pdf = build / "handoff_preview.pdf"
    shutil.copyfile(pdf, final)
    shutil.copyfile(pdf, source / "member3_handoff_preview.pdf")
    command(["pdftoppm", "-r", "105", "-png", str(pdf), str(build / "page")], root, "render.txt")
    print(f"PDF: {final}")
    print(f"Rendered pages: {build}")


if __name__ == "__main__":
    main()
