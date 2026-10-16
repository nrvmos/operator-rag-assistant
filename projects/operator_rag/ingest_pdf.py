"""PDF ingestion for the Operator Knowledge RAG retriever.

Reads a PDF page by page with pypdf and emits one ``Chunk`` per page, using the
exact ``industrial_ai_portfolio.rag.Chunk`` schema the TF-IDF retriever already
loads (``source``, ``title``, ``text``). The page number is kept in ``source`` as
``"<name>:p<N>"`` (one-based) so a citation that the retriever returns can name the
page it came from. The retriever itself is not touched: ``retrieve_chunks`` copies
each chunk's ``source`` and ``title`` into its result rows, so the page marker
round-trips to whatever builds the final answer.

``make_fixture_pdf`` builds a small, fully synthetic, public-safe two-page PDF (no
plant documents, no live services) for the tests.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader

from industrial_ai_portfolio.rag import Chunk


def _norm(text: str) -> str:
    """Collapse whitespace so a phrase that pypdf wraps across a line break still matches."""
    return " ".join(text.split())


def page_text(path: str | Path) -> list[str]:
    """Extract each page's text (whitespace-normalised), in page order."""
    reader = PdfReader(str(path))
    return [_norm((page.extract_text() or "")) for page in reader.pages]


def _default_title(path: str | Path, name: str | None) -> str:
    stem = name or Path(str(path)).name
    stem = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", Path(stem).name)  # drop the extension
    return re.sub(r"[_\-]+", " ", stem).strip().title() or "Document"


def ingest_pdf(
    path: str | Path,
    *,
    name: str | None = None,
    title: str | None = None,
    chunk_size: int = 0,
) -> list[Chunk]:
    """Chunk a PDF by page into the retriever's ``Chunk`` schema.

    ``source`` is ``"<name>:p<page>"`` (page one-based); ``name`` defaults to the file
    name. With ``chunk_size <= 0`` (the default) each page is one chunk. A larger
    ``chunk_size`` splits a page's words into windows like ``chunk_documents``, and the
    page marker is kept on every chunk so citations still name the page.
    """
    pages = page_text(path)
    name = name or Path(str(path)).name
    title = title or _default_title(path, name)
    chunks: list[Chunk] = []
    for index, text in enumerate(pages, start=1):
        words = text.split()
        if chunk_size and chunk_size > 0 and len(words) > chunk_size:
            step = chunk_size  # one page, no overlap: a citation must name the page
            for start in range(0, len(words), step):
                window = " ".join(words[start:start + chunk_size])
                chunks.append(Chunk(source=f"{name}:p{index}", title=title, text=window))
        else:
            chunks.append(Chunk(source=f"{name}:p{index}", title=title, text=text))
    return chunks


def make_fixture_pdf(out_path: str | Path, pages: list[list[str]]) -> Path:
    """Write a synthetic, public-safe PDF with one logical page per entry in ``pages``.

    ``pages`` is a list of pages; each page is a list of the lines to print on it. No
    plant documents and no live services are involved; the text is plain, self-describing
    filler.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(False)
    for lines in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in lines:
            pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(out_path))
    return Path(str(out_path))
