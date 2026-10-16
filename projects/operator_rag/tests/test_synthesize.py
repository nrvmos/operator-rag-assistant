"""The synthesis step, with the model replaced so nothing here touches the network."""
from __future__ import annotations

import json

import pytest

from projects.operator_rag.evals.run_eval import score
from projects.operator_rag.synthesize import synthesize


def _reply(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _chunks() -> list[dict]:
    return [
        {"source": "pump_reliability_study_note.md", "title": "Pump Reliability Study Note",
         "text": "Rising vibration and bearing temperature increases can indicate pump problems."},
        {"source": "industrial_ai_guardrails.md", "title": "Industrial AI Guardrails Study Note",
         "text": "AI outputs should be treated as decision support rather than authority."},
    ]


def test_cites_only_the_chunks_the_model_names():
    def poster(endpoint, payload):
        assert endpoint.startswith("http") and "Authorization" not in json.dumps(payload)
        return _reply({"answer": "Vibration is an early indicator.", "cited_chunks": [1], "refused": False})

    result = synthesize("What indicates pump problems?", _chunks(), poster=poster)

    assert result["refused"] is False
    assert result["cited_chunks"] == [_chunks()[0]["text"]]


def test_drops_a_citation_the_model_invents():
    def poster(endpoint, payload):
        return _reply({"answer": "See the manual.", "cited_chunks": [1, 9], "refused": False})

    result = synthesize("What indicates pump problems?", _chunks(), poster=poster)

    assert result["cited_chunks"] == [_chunks()[0]["text"]]


def test_refusal_clears_citations():
    def poster(endpoint, payload):
        return _reply({"answer": "no", "cited_chunks": [1], "refused": True})

    result = synthesize("Which valve should I open right now?", _chunks(), poster=poster)

    assert result["refused"] is True
    assert result["cited_chunks"] == []


def test_ungrounded_reply_becomes_a_refusal():
    def poster(endpoint, payload):
        return _reply({"answer": "I made this up.", "cited_chunks": [], "refused": False})

    result = synthesize("What is the plant's production volume?", _chunks(), poster=poster)

    assert result["refused"] is True
    assert result["cited_chunks"] == []


def test_non_json_reply_is_an_error_not_an_answer():
    def poster(endpoint, payload):
        return {"choices": [{"message": {"content": "sorry, I cannot format that"}}]}

    with pytest.raises(ValueError):
        synthesize("What indicates pump problems?", _chunks(), poster=poster)


def test_a_mixed_valid_and_invented_citation_fails_the_row():
    """The review's reproduction: the model cites chunk 1 and chunk 99.

    Grounding drops 99, so scoring the grounded list would pass the row. The raw
    citation list must survive, and the row must fail while staying in the denominator.
    """
    chunks = [{"source": "pump_reliability_study_note.md", "text": "pump bearing vibration"}]
    row = {"id": "raw-cite", "question": "question", "answerable": True,
           "expected_source": "pump_reliability_study_note.md", "expected_keywords": ["pump"]}

    def poster(endpoint, payload):
        return _reply({"answer": "pump bearing vibration", "cited_chunks": [1, 99], "refused": False})

    result = synthesize("question", chunks, poster=poster)
    assert 99 in result["raw_cited"]
    report = score([row], lambda question: (result, chunks))
    assert report["citation_accuracy"] == 0.0
    assert report["rows"] == 1


def test_endpoint_comes_from_the_environment(monkeypatch):
    seen = {}

    def poster(endpoint, payload):
        seen["endpoint"] = endpoint
        return _reply({"answer": "x", "cited_chunks": [1], "refused": False})

    monkeypatch.setenv("OPERATOR_RAG_LLM_URL", "http://localhost:9999/v1/chat/completions")
    synthesize("What indicates pump problems?", _chunks(), poster=poster)

    assert seen["endpoint"] == "http://localhost:9999/v1/chat/completions"
