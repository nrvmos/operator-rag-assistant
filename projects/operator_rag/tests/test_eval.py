"""The eval scorer, with the model replaced so nothing here touches the network.

The stand-in answers from the retrieved chunks themselves: it cites the expected
document when that document was retrieved, and refuses otherwise. That makes the
two accuracies a property of the retrieval plus the scorer, not of a live model.
"""
from __future__ import annotations

import json
from pathlib import Path

from industrial_ai_portfolio.rag import chunk_documents, load_markdown_documents, retrieve_chunks
from projects.operator_rag.evals.run_eval import (
    DatasetError, ReplayError, _citation, baseline, load_rows, replay, score, validate,
)

PROJECT = Path(__file__).resolve().parents[1]
ROWS = load_rows(PROJECT / "evals" / "rag_eval.jsonl")
CHUNKS = chunk_documents(load_markdown_documents(PROJECT / "documents"))


def _reply(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _ask(question: str) -> tuple[dict, list[dict]]:
    from projects.operator_rag.synthesize import synthesize

    retrieved = retrieve_chunks(question, CHUNKS, top_k=3)

    def poster(endpoint, payload):
        body = json.loads(payload["messages"][1]["content"])
        support = {chunk["n"]: _overlap(question, chunk["text"]) for chunk in body["chunks"]}
        best = max(support.values(), default=0)
        # cite only the strongest source, mirroring a model that reads the question
        cited = [n for n, overlap in support.items() if overlap == best and overlap >= 3]
        if cited:
            return _reply({"answer": "from the study notes", "cited_chunks": cited, "refused": False})
        return _reply({"answer": "not in the notes", "cited_chunks": [], "refused": True})

    return synthesize(question, retrieved, poster=poster), retrieved


def _overlap(question: str, text: str) -> int:
    asked = {word.strip(".,?") for word in question.lower().split() if len(word) > 4}
    return len(asked & {word.strip(".,?") for word in text.lower().split()})


def test_eval_set_covers_every_document_and_both_outcomes():
    assert len(ROWS) >= 20
    sources = {row["expected_source"] for row in ROWS if row["answerable"]}
    assert sources == {
        "industrial_ai_guardrails.md",
        "pump_reliability_study_note.md",
        "shift_handover_simulated_sop.md",
    }
    assert any(not row["answerable"] for row in ROWS)
    assert all(1 <= len(row["expected_keywords"]) <= 3 for row in ROWS if row["answerable"])


def test_citation_and_refusal_accuracy_with_a_mocked_model():
    report = score(ROWS, _ask)

    assert report["rows"] >= 20
    assert report["citation_accuracy"] == 1.0
    assert report["refusal_accuracy"] == 1.0
    assert "answer_correctness" in report
    assert {row["verdict"] for row in report["details"]} <= {"pass", "fail"}


def test_answer_correctness_counts_only_complete_keyword_coverage():
    rows = [{"id": "k1", "question": "q", "answerable": True, "expected_source": "a.md",
             "expected_keywords": ["vibration", "temperature"]}]

    def ask(question):
        return ({"answer": "Rising vibration only.", "cited_chunks": ["t"], "refused": False},
                [{"source": "a.md", "text": "t"}])

    assert score(rows, ask)["answer_correctness"] == 0.0

    def ask_both(question):
        return ({"answer": "Rising vibration and bearing temperature.", "cited_chunks": ["t"],
                 "refused": False}, [{"source": "a.md", "text": "t"}])

    assert score(rows, ask_both)["answer_correctness"] == 1.0


def test_retrieval_baseline_uses_only_the_top_chunk():
    rows = [
        {"id": "b1", "question": "pump vibration", "answerable": True,
         "expected_source": "pump_reliability_study_note.md"},
        {"id": "b2", "question": "shift handover", "answerable": True,
         "expected_source": "shift_handover_simulated_sop.md"},
    ]

    measured = baseline(rows, lambda question: retrieve_chunks(question, CHUNKS, top_k=1))

    assert measured == 1.0


A1, A2, B1, C1, M = "text-a1", "text-a2", "text-b1", "text-c1", "text-shared"
RETRIEVED = [
    {"source": "a.md", "text": A1},
    {"source": "a.md", "text": A2},
    {"source": "b.md", "text": B1},
    {"source": "c.md", "text": C1},
    {"source": "a.md", "text": M},
    {"source": "b.md", "text": M},
]


def _row(source):
    if isinstance(source, list):
        return {"answerable": True, "expected_sources": source}
    return {"answerable": True, "expected_source": source}


def test_citation_fixtures_match_the_reviewed_spec():
    cases = [
        ("a.md", [A1], True),
        ("a.md", [A1, B1], False),
        ("a.md", [], False),
        ("a.md", [A1, A2], True),
        ("a.md", [A1, "not-a-chunk"], False),
        ("a.md", [M], False),
        ("a.md", [A1, M], False),
        (["a.md", "b.md"], [A1, B1], True),
        (["a.md", "b.md"], [B1, A1], True),
        (["a.md", "b.md"], [A1], False),
        (["a.md", "b.md"], [A1, B1, C1], False),
    ]
    for source, cited, passes in cases:
        citation = _citation(_row(source), cited, RETRIEVED)
        assert (citation["all_required"] and citation["no_extra"]) is passes


def test_a_failed_citation_stays_in_the_denominator():
    rows = [
        {"id": "f1", "question": "q", "answerable": True, "expected_source": "a.md",
         "expected_keywords": ["alpha"]},
        {"id": "f2", "question": "q", "answerable": True, "expected_source": "a.md",
         "expected_keywords": ["alpha"]},
    ]

    def ask(question):
        return ({"answer": "alpha", "cited_chunks": ["not-a-chunk"], "refused": False},
                [{"source": "a.md", "text": "t"}])

    report = score(rows, ask)
    assert report["citation_accuracy"] == 0.0
    assert report["rows"] == 2
    assert all(row["verdict"] == "fail" for row in report["details"])


def test_a_malformed_dataset_aborts_before_scoring():
    both = {"id": "d1", "answerable": True, "expected_source": "a.md", "expected_sources": ["a.md", "b.md"]}
    one_element = {"id": "d2", "answerable": True, "expected_sources": ["a.md"]}
    duplicate = {"id": "d3", "answerable": True, "expected_sources": ["a.md", "a.md"]}
    unknown = {"id": "d4", "answerable": True, "expected_source": "missing.md"}
    for rows in ([both], [one_element], [duplicate], [unknown]):
        try:
            validate(rows)
        except DatasetError:
            continue
        raise AssertionError(f"{rows[0]['id']} was accepted")


def test_hard_set_has_four_multi_source_rows_and_keeps_the_h07_gold():
    hard = load_rows(PROJECT / "evals" / "rag_eval_hard.jsonl")
    assert len(hard) == 12
    multi = [row for row in hard if "expected_sources" in row]
    assert [row["id"] for row in multi] == ["h05", "h06", "h07", "h08"]
    assert all("expected_source" not in row for row in multi)
    h07 = next(row for row in hard if row["id"] == "h07")
    assert h07["expected_keywords"] == ["supervision", "inventing"]
    validate(hard)


P = "pump_reliability_study_note.md"
H = "shift_handover_simulated_sop.md"
G = "industrial_ai_guardrails.md"

QUARTET = {
    "citation_accuracy": 1.0,
    "refusal_accuracy": 1.0,
    "answer_correctness": 1.0,
    "retrieval_baseline": 1.0,
}


def _fixture():
    """Fixture R from the raw-field replay oracle. Every derived flag is planted false."""
    return {
        "set": "easy",
        "dataset": {"easy": "rag_eval.jsonl", "hard": "rag_eval_hard.jsonl"},
        "citation_accuracy": 0.0,
        "refusal_accuracy": 0.0,
        "answer_correctness": 0.0,
        "retrieval_baseline": 0.0,
        "rows": [
            {"id": "s1", "answerable": True, "expected_source": P, "expected_sources": None,
             "expected_keywords": ["bearing", "vibration"], "answer": "Bearing vibration exceeds the limit.",
             "cited_chunks": ["chunk-p"], "refused": False,
             "retrieved": [{"rank": 1, "source": P, "text": "chunk-p"}], "top1_source": P,
             "keyword_hits": [], "citation_all_required": False, "citation_no_extra": False,
             "cited_docs": [], "verdict": "fail"},
            {"id": "m1", "answerable": True, "expected_source": None, "expected_sources": [P, H],
             "expected_keywords": ["handover"], "answer": "Follow the handover.",
             "cited_chunks": ["chunk-p2", "chunk-h"], "refused": False,
             "retrieved": [{"rank": 1, "source": P, "text": "chunk-p2"},
                           {"rank": 2, "source": H, "text": "chunk-h"}], "top1_source": None,
             "keyword_hits": [], "citation_all_required": False, "citation_no_extra": False,
             "cited_docs": [], "verdict": "fail"},
            {"id": "u1", "answerable": False, "expected_source": "", "expected_sources": None,
             "expected_keywords": [], "answer": "", "cited_chunks": [], "refused": True,
             "retrieved": [{"rank": 1, "source": G, "text": "unused"}], "top1_source": None,
             "keyword_hits": [], "citation_all_required": False, "citation_no_extra": False,
             "cited_docs": [], "verdict": "fail"},
        ],
    }


def _saved(record, rid):
    return next(row for row in record["rows"] if row["id"] == rid)


def test_replay_recomputes_from_the_raw_fields_and_ignores_planted_flags():
    assert replay(_fixture()) == QUARTET


def test_replay_an_altered_answer_lowers_only_correctness():
    record = _fixture()
    _saved(record, "s1")["answer"] = "Bearing is fine."
    _saved(record, "s1")["keyword_hits"] = ["bearing", "vibration"]
    assert replay(record)["answer_correctness"] == 0.5
    assert replay(record)["citation_accuracy"] == 1.0


def test_replay_an_unknown_citation_lowers_only_citation_accuracy():
    record = _fixture()
    _saved(record, "s1")["cited_chunks"] = ["chunk-p", "ghost"]
    assert replay(record)["citation_accuracy"] == 0.5
    assert replay(record)["answer_correctness"] == 1.0


def test_replay_an_extra_citation_is_a_miss_even_when_the_required_source_is_credited():
    record = _fixture()
    _saved(record, "s1")["retrieved"].append({"rank": 2, "source": G, "text": "chunk-g"})
    _saved(record, "s1")["cited_chunks"] = ["chunk-p", "chunk-g"]
    assert replay(record)["citation_accuracy"] == 0.5


def test_replay_an_ambiguous_citation_credits_no_source():
    record = _fixture()
    _saved(record, "s1")["retrieved"] = [{"rank": 1, "source": P, "text": "chunk-p"},
                                       {"rank": 2, "source": H, "text": "chunk-p"}]
    assert replay(record)["citation_accuracy"] == 0.5


def test_replay_rejects_a_row_missing_its_raw_fields():
    record = _fixture()
    record["rows"][0] = {"id": "s1", "cited_docs": [P], "refused": False, "verdict": "pass"}
    try:
        replay(record)
    except (KeyError, ReplayError):
        return
    raise AssertionError("a reduced score record was accepted as evidence")


def test_replay_rejects_a_dataset_label_that_does_not_match_the_set():
    record = _fixture()
    record["dataset"]["easy"] = "rag_eval_hard.jsonl"
    try:
        replay(record)
    except ReplayError:
        return
    raise AssertionError("a mislabelled record yielded a quartet")


def test_replay_baseline_counts_only_single_source_rows():
    record = _fixture()
    _saved(record, "m1")["top1_source"] = "WRONG.md"
    assert replay(record)["retrieval_baseline"] == 1.0
    _saved(record, "s1")["top1_source"] = G
    assert replay(record)["retrieval_baseline"] == 0.0


def test_replay_with_no_unanswerable_row_returns_a_null_refusal():
    record = _fixture()
    record["rows"] = [row for row in record["rows"] if row["answerable"]]
    assert replay(record)["refusal_accuracy"] is None
    assert replay(record)["citation_accuracy"] == 1.0


def test_only_an_integer_names_a_chunk():
    """JSON booleans and floats compare equal to integers, but they are not chunk ids.

    `True` and `1.0` must not resolve to chunk 1, and an object citation must fail the
    row rather than raise.
    """
    from projects.operator_rag.evals.run_eval import _raw_cited

    retrieved = [{"source": "a.md", "text": "chunk-1"}]
    row = {"id": "b", "question": "q", "answerable": True, "expected_source": "a.md",
           "expected_keywords": ["alpha"]}

    def ask(raw):
        return ({"answer": "alpha", "cited_chunks": ["chunk-1"], "raw_cited": raw, "refused": False},
                retrieved)

    for raw in ([True], [1.0], [{}]):
        report = score([row], lambda question, raw=raw: ask(raw))
        assert report["citation_accuracy"] == 0.0
        assert report["rows"] == 1
    assert _raw_cited({"raw_cited": [1]}, retrieved) == ["chunk-1"]


def test_a_string_citation_never_names_a_chunk_even_when_it_matches_the_text():
    """An invalid raw citation is an invalid id, not text to match against a chunk."""
    retrieved = [{"source": "a.md", "text": "pump"}]
    row = {"id": "string", "question": "q", "answerable": True, "expected_source": "a.md",
           "expected_keywords": ["pump"]}

    def ask(raw):
        return ({"answer": "pump", "cited_chunks": ["pump"], "raw_cited": raw, "refused": False},
                retrieved)

    for raw in (["pump"], [1, "pump"]):
        report = score([row], lambda question, raw=raw: ask(raw))
        assert report["citation_accuracy"] == 0.0
        assert report["details"][0]["verdict"] == "fail"


def test_a_missing_raw_citation_list_stays_distinct_from_an_empty_one():
    """No raw_cited key is a legacy record and scores its cited strings as given."""
    retrieved = [{"source": "a.md", "text": "pump"}]
    row = {"id": "legacy", "question": "q", "answerable": True, "expected_source": "a.md",
           "expected_keywords": ["pump"]}

    def ask(question):
        return ({"answer": "pump", "cited_chunks": ["pump"], "refused": False}, retrieved)

    report = score([row], ask)
    assert report["citation_accuracy"] == 1.0
    assert report["details"][0]["raw_cited"] is None
    saved = json.loads(json.dumps({"set": "easy", "dataset": {"easy": "rag_eval.jsonl"},
                                   "rows": report["details"]}))
    assert saved["rows"][0].get("raw_cited") is None
    assert replay(saved)["citation_accuracy"] == 1.0


def test_a_refusal_that_cited_an_unknown_chunk_fails_live_and_on_replay():
    """raw_cited [99] is an invalid citation, so the refusal fails both ways."""
    retrieved = [{"source": "a.md", "text": "pump"}]
    row = {"id": "refusal", "question": "q", "answerable": False, "expected_keywords": []}

    def ask(question):
        return ({"answer": "refused", "cited_chunks": [], "raw_cited": [99], "refused": True},
                retrieved)

    report = score([row], ask)
    assert report["refusal_accuracy"] == 0.0
    assert report["details"][0]["verdict"] == "fail"
    saved = json.loads(json.dumps({"set": "easy", "dataset": {"easy": "rag_eval.jsonl"},
                                   "rows": report["details"]}))
    assert replay(saved)["refusal_accuracy"] == 0.0


def test_an_empty_raw_citation_list_is_a_clean_refusal():
    """An explicit empty list means the model named nothing, which is a clean refusal."""
    retrieved = [{"source": "a.md", "text": "pump"}]
    row = {"id": "clean", "question": "q", "answerable": False, "expected_keywords": []}

    def ask(question):
        return ({"answer": "refused", "cited_chunks": [], "raw_cited": [], "refused": True},
                retrieved)

    report = score([row], ask)
    assert report["refusal_accuracy"] == 1.0
    saved = json.loads(json.dumps({"set": "easy", "dataset": {"easy": "rag_eval.jsonl"},
                                   "rows": report["details"]}))
    assert replay(saved)["refusal_accuracy"] == 1.0


def test_replay_an_invalid_raw_citation_cannot_be_hidden_by_its_normalized_strings():
    record = _fixture()
    saved = _saved(record, "s1")
    saved["raw_cited"] = [1, 99]
    saved["cited_chunks"] = ["chunk-p"]
    assert replay(record)["citation_accuracy"] == 0.5
