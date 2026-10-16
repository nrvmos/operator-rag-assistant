"""PDF ingestion: page count, a phrase landing on the right page, and a citation
round-trip through the existing (unmodified) retriever. The fixture is a synthetic,
public-safe two-page PDF generated in the test — no plant documents, no live services.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from industrial_ai_portfolio.rag import retrieve_chunks
from projects.operator_rag.ingest_pdf import ingest_pdf, make_fixture_pdf

PHRASE_ONE = "reactor coolant flow rate"
PHRASE_TWO = "turbine lube oil pressure"


def _fixture(tmp_path: Path) -> Path:
    path = tmp_path / "public_study_note.pdf"
    make_fixture_pdf(
        path,
        [
            ["This is a public-safe study note.",
             f"The {PHRASE_ONE} is measured with a venturi.",
             "It is described in general terms only."],
            ["This is a second synthetic page.",
             f"The {PHRASE_TWO} is logged every shift.",
             "No live site data is used."],
        ],
    )
    return path


def test_pdf_page_count(tmp_path):
    path = _fixture(tmp_path)
    assert len(PdfReader(str(path)).pages) == 2
    chunks = ingest_pdf(path, name="public_study_note.pdf")
    assert len(chunks) == 2
    assert [c.source for c in chunks] == ["public_study_note.pdf:p1", "public_study_note.pdf:p2"]
    assert all(c.title == "Public Study Note" for c in chunks)
    # same schema the markdown loader and retriever already consume
    for c in chunks:
        assert set(c.__dict__) == {"source", "title", "text"}


def test_pdf_known_phrase_in_right_page_chunk(tmp_path):
    path = _fixture(tmp_path)
    chunks = {c.source: c.text for c in ingest_pdf(path, name="public_study_note.pdf")}
    assert PHRASE_ONE in chunks["public_study_note.pdf:p1"]
    assert PHRASE_TWO in chunks["public_study_note.pdf:p2"]
    # the phrase does not bleed onto the other page
    assert PHRASE_ONE not in chunks["public_study_note.pdf:p2"]
    assert PHRASE_TWO not in chunks["public_study_note.pdf:p1"]


def test_pdf_citation_round_trip_through_retriever(tmp_path):
    path = _fixture(tmp_path)
    chunks = ingest_pdf(path, name="public_study_note.pdf")
    # the unmodified retriever ranks the page that matches the question, and its
    # result rows carry the chunk's source, so the citation names the page.
    top = retrieve_chunks(f"what is the {PHRASE_ONE}", chunks, top_k=1)
    assert top
    assert top[0]["source"] == "public_study_note.pdf:p1"
    assert top[0]["title"] == "Public Study Note"
    assert PHRASE_ONE in top[0]["text"]
    other = retrieve_chunks(f"what is the {PHRASE_TWO}", chunks, top_k=1)
    assert other
    assert other[0]["source"] == "public_study_note.pdf:p2"
    assert PHRASE_TWO in other[0]["text"]
    # the page marker is what a citation would print
    assert top[0]["source"].endswith(":p1") and other[0]["source"].endswith(":p2")
