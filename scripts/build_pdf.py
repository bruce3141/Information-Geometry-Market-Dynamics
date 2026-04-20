#!/usr/bin/env python3
"""Build an Entropy-style PDF from the current draft using pandoc + XeLaTeX.

Reads paper/p3_1_v57.md, preprocesses the markdown (converts front matter to
YAML-pandoc metadata, wraps the abstract in the entropyabstract environment,
substitutes Unicode checkmarks for LaTeX \\checkmark in tables), and emits
a PDF via pandoc driving XeLaTeX with paper/entropy_header.tex as the
header-includes file.

Requirements:
  - pandoc >= 3.0
  - XeLaTeX (via TeX Live, MacTeX, or MiKTeX)
  - TeX Gyre Pagella font (Palatino clone) via texlive-fonts-recommended
  - Core LaTeX packages: mathpazo, fontspec, geometry, titling, titlesec,
    fancyhdr, booktabs, hyperref, xcolor, mdframed, microtype, caption

Run from repository root:
    python3 scripts/build_pdf.py

Output: paper/p3_1_v57.pdf
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to repo root (one level up from scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC       = REPO_ROOT / "paper" / "p3_1_v57.md"
HEADER    = REPO_ROOT / "paper" / "entropy_header.tex"
BUILD_DIR = REPO_ROOT / "paper"
BUILD     = BUILD_DIR / "p3_1_v57_build.md"
OUT_PDF   = REPO_ROOT / "paper" / "p3_1_v57.pdf"
FIG_DIR   = REPO_ROOT / "paper"


def preprocess(text: str) -> tuple[str, dict]:
    """Return (transformed_markdown, metadata_dict)."""
    m_title = re.search(r'^# (.+)$', text, flags=re.MULTILINE)
    title = m_title.group(1).strip() if m_title else "Paper"

    author_block = ""
    date_line = ""
    if m_title:
        after_title = text[m_title.end():]
        parts = after_title.split('\n---\n', 1)
        if len(parts) == 2:
            author_block = parts[0].strip()
            rest = parts[1]
            m_date = re.search(r'^(April \d{4}.*?)$', author_block, flags=re.MULTILINE)
            if m_date:
                date_line = m_date.group(1).strip()
                author_block = author_block.replace(m_date.group(0), '').strip()
            text = rest
        else:
            text = after_title

    m_fn = re.search(r'\[\^affil\]:\s*(.+)$', author_block, flags=re.MULTILINE)
    thanks = m_fn.group(1).strip() if m_fn else ""
    author_block = re.sub(r'\[\^affil\]:.*$', '', author_block, flags=re.MULTILINE)
    author_block = re.sub(r'\[\^affil\]', '', author_block)
    author_lines = [ln.strip().rstrip('\\').strip()
                    for ln in author_block.split('\n') if ln.strip()]
    if author_lines:
        author_lines[0] = re.sub(r'^\*\*(.+?)\*\*', r'\1', author_lines[0])
    if thanks:
        author_lines[0] = author_lines[0] + r'\thanks{' + thanks + '}'
    author_latex = r' \\ '.join(author_lines)

    # Abstract: wrap in entropyabstract environment via raw-LaTeX tags,
    # leaving the prose as markdown so pandoc tokenizes it properly.
    def _wrap_abstract(m):
        body = m.group(1).strip()
        return (
            '```{=latex}\n'
            r'\begin{entropyabstract}' + '\n'
            r'\noindent\textbf{\textcolor{EntropyAccent}{Abstract.}}\ ' + '\n'
            '```\n\n'
            + body + '\n\n'
            '```{=latex}\n'
            r'\end{entropyabstract}' + '\n'
            '```\n\n'
        )

    text = re.sub(
        r'## Abstract\s*\n\n(.*?)\n\n---\n',
        _wrap_abstract,
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Horizontal rules: replace with small vertical spacing
    text = re.sub(r'^---$', r'\\vspace{0.4em}', text, flags=re.MULTILINE)

    # Table B3 checkmarks: substitute with LaTeX $\checkmark$
    text = text.replace('✓', r'$\checkmark$')

    meta = {
        "title": title,
        "author": author_latex,
        "date": date_line,
    }
    return text, meta


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: paper source not found at {SRC}", file=sys.stderr)
        return 2
    if not HEADER.exists():
        print(f"ERROR: LaTeX header not found at {HEADER}", file=sys.stderr)
        return 2

    text = SRC.read_text()
    processed, meta = preprocess(text)
    BUILD.write_text(processed)

    cmd = [
        "pandoc",
        str(BUILD),
        "-o", str(OUT_PDF),
        "--pdf-engine=xelatex",
        "--from=markdown+tex_math_dollars+raw_tex+fenced_divs+raw_attribute",
        "--include-in-header=" + str(HEADER),
        "--resource-path=" + str(FIG_DIR),
        "--variable=title:" + meta["title"],
        "--variable=author:" + meta["author"],
        "--variable=date:" + meta["date"],
        "--variable=documentclass:article",
        "--variable=fontsize:10pt",
        "--variable=mainfont:TeX Gyre Pagella",
        "--variable=fontfamily:mathpazo",
        "--variable=colorlinks:true",
        "--standalone",
    ]
    print("Running pandoc...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr, file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    # Clean up the intermediate build markdown; the PDF is the deliverable.
    if BUILD.exists():
        BUILD.unlink()

    print(f"Built {OUT_PDF.relative_to(REPO_ROOT)} "
          f"({OUT_PDF.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
