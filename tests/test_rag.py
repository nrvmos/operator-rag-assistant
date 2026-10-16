from pathlib import Path

from industrial_ai_portfolio.rag import (
    Chunk,
    build_study_answer,
    chunk_documents,
    is_unsafe_live_operations_request,
    load_markdown_documents,
    retrieve_chunks,
)

BUNDLED_DOCUMENTS = (
    Path(__file__).resolve().parents[1] / "projects" / "operator_rag" / "documents"
)


def test_refuses_live_operations_request():
    question = "What should I do right now during a live plant trip?"

    assert is_unsafe_live_operations_request(question)
    assert "cannot advise on live operations" in build_study_answer(question, [])


def test_retrieval_returns_matching_chunk():
    chunks = [
        Chunk(source="a.md", title="Pump Note", text="Pump vibration trends can indicate reliability problems."),
        Chunk(source="b.md", title="Cooking Note", text="This text is unrelated to industrial equipment."),
    ]

    results = retrieve_chunks("pump vibration reliability", chunks, top_k=1)

    assert results
    assert results[0]["source"] == "a.md"


def test_bundled_pump_reliability_indicators_matches_reported_results():
    question = "pump reliability indicators"
    documents = load_markdown_documents(BUNDLED_DOCUMENTS)
    chunks = chunk_documents(documents)
    retrieved = retrieve_chunks(question, chunks)
    answer = build_study_answer(question, retrieved)
    top_excerpt = next(line[2:] for line in answer.splitlines() if line.startswith("> "))

    assert len(documents) == 3
    assert len(chunks) == 6
    assert [f"{float(row['score']):.4f}" for row in retrieved] == ["0.3065", "0.0441", "0.0430"]
    assert [row["source"] for row in retrieved] == [
        "pump_reliability_study_note.md",
        "shift_handover_simulated_sop.md",
        "industrial_ai_guardrails.md",
    ]
    assert top_excerpt == "This public-safe study note summarizes general pump reliability concepts."
