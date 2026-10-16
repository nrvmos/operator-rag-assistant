"""Score the operator-RAG eval set, and record one run.

Usage, from the repo root, with an endpoint that speaks the OpenAI chat format:

    export OPERATOR_RAG_LLM_URL=http://127.0.0.1:8080/v1/chat/completions   # PowerShell: $env:OPERATOR_RAG_LLM_URL = "..."
    python projects/operator_rag/evals/run_eval.py

Each row of evals/rag_eval.jsonl names the document that should be cited, or marks
the question unanswerable. Answerable rows also carry one to three expected
keywords. The script reports three numbers:

- citation accuracy: the share of answerable questions whose cited chunks all
  come from the expected document;
- refusal accuracy: the share of unanswerable questions the run refused;
- answer correctness: the share of answerable questions whose answer contains
  every expected keyword.

It also reports a retrieval-only baseline: the share of answerable questions
whose single top-ranked TF-IDF chunk comes from the expected document. That row
uses no model. It looks at one chunk while the model sees three, so the gap between
it and citation accuracy mixes the model's effect with the wider context.

The run is written to evals/results/<stamp>-<model>-<set>.json, one object per row
with the cited documents, whether the run refused, and the verdict.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from industrial_ai_portfolio.rag import chunk_documents, load_markdown_documents, retrieve_chunks  # noqa: E402
from projects.operator_rag.synthesize import synthesize  # noqa: E402

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

CORPUS = {
    "industrial_ai_guardrails.md",
    "pump_reliability_study_note.md",
    "shift_handover_simulated_sop.md",
}


class DatasetError(Exception):
    """The eval file itself is malformed. The run stops before any metric."""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(rows: list[dict]) -> None:
    """Reject a malformed dataset before any fraction is emitted.

    A row carries one source field. `expected_source` is a single corpus name, or
    empty for an unanswerable row. `expected_sources` is two or more distinct corpus
    names, and only on a row that has no `expected_source`. Anything else is a data
    error, not a low score.
    """
    for row in rows:
        rid = row.get("id", "?")
        has_single = "expected_source" in row
        has_multi = "expected_sources" in row
        if has_single and has_multi:
            raise DatasetError(f"{rid} carries both expected_source and expected_sources")
        if not has_single and not has_multi:
            raise DatasetError(f"{rid} carries no source field")
        if has_multi:
            sources = row["expected_sources"]
            if (not isinstance(sources, list) or len(sources) < 2
                    or len(set(sources)) != len(sources)
                    or any(source not in CORPUS for source in sources)):
                raise DatasetError(f"{rid} has an invalid expected_sources")
        elif row["answerable"] and row["expected_source"] not in CORPUS:
            raise DatasetError(f"{rid} names an unknown expected_source")


def _required(row: dict) -> set[str]:
    sources = row.get("expected_sources")
    if sources:
        return set(sources)
    return {row["expected_source"]}


def _citation(row: dict, cited: list[str], retrieved: list[dict]) -> dict:
    """Score one row's citations against the chunks that were actually retrieved.

    A cited string counts only when it matches exactly one retrieved chunk. A string
    that matches nothing, or that matches two sources, adds nothing and forces
    `no_extra` false, so the row fails and still counts in the denominator.
    """
    sources_by_text: dict[str, set[str]] = {}
    for chunk in retrieved:
        sources_by_text.setdefault(chunk["text"], set()).add(chunk["source"])
    required = _required(row)
    credited: set[str] = set()
    no_extra = True
    for text in cited:
        sources = sources_by_text.get(text, set())
        if len(sources) != 1:
            no_extra = False
            continue
        source = next(iter(sources))
        if source in required:
            credited.add(source)
        else:
            no_extra = False
    return {
        "all_required": required <= credited,
        "no_extra": no_extra,
        "credited": sorted(credited),
    }


def _split_raw(result: dict, retrieved: list[dict]) -> tuple[list[str], bool]:
    """Separate valid chunk ids from everything else the model cited.

    A missing `raw_cited` or one that is None is a legacy record, scored on its
    cited strings as given, with no invalid flag. A present list, including an
    empty one, is the model's own citation list: only an actual integer that names
    a retrieved chunk becomes that chunk's text. A boolean, a float, a string, an
    object or an unknown number sets the invalid flag and is left out of the text
    list, so it can never match a chunk's text by accident.
    """
    raw = result.get("raw_cited")
    if raw is None:
        return list(result["cited_chunks"]), False
    by_number = {index: chunk["text"] for index, chunk in enumerate(retrieved, start=1)}
    texts: list[str] = []
    invalid = False
    for item in raw:
        if type(item) is int and item in by_number:
            texts.append(by_number[item])
        else:
            invalid = True
    return texts, invalid


def _raw_cited(result: dict, retrieved: list[dict]) -> list[str]:
    """The chunk texts the model actually cited, before grounding."""
    return _split_raw(result, retrieved)[0]


def _scored_citation(row: dict, result: dict, retrieved: list[dict]) -> dict:
    """One row's citation verdict, with an invalid raw citation failing the row."""
    texts, invalid = _split_raw(result, retrieved)
    citation = _citation(row, texts, retrieved)
    if invalid:
        citation["no_extra"] = False
    return citation


def _refusal_ok(result: dict, retrieved: list[dict]) -> bool:
    """A clean refusal names no chunk and carries no invalid citation."""
    texts, invalid = _split_raw(result, retrieved)
    return bool(result["refused"]) and not texts and not invalid


def _verdict(row: dict, result: dict, retrieved: list[dict]) -> str:
    if row["answerable"]:
        citation = _scored_citation(row, result, retrieved)
        return "pass" if citation["all_required"] and citation["no_extra"] else "fail"
    return "pass" if _refusal_ok(result, retrieved) else "fail"


def _detail(row: dict, result: dict, retrieved: list[dict], top1: list[dict]) -> dict:
    """One auditable row: enough to recompute every headline without rerunning.

    `cited_chunks` is the chunk text the model actually cited. `retrieved` keeps the
    text, source and rank of every chunk offered to the model. `keyword_hits` records
    which expected keywords the answer contained, so correctness can be recomputed.
    """
    if row["answerable"]:
        citation = _scored_citation(row, result, retrieved)
    else:
        citation = {"all_required": False, "no_extra": False, "credited": []}
    keywords = row.get("expected_keywords", [])
    answer = result["answer"]
    hits = [keyword for keyword in keywords if keyword.lower() in answer.lower()]
    return {
        "id": row["id"],
        "answerable": row["answerable"],
        "expected_source": row.get("expected_source"),
        "expected_sources": row.get("expected_sources"),
        "expected_keywords": keywords,
        "answer": answer,
        "cited_chunks": _raw_cited(result, retrieved),
        "raw_cited": None if result.get("raw_cited") is None else list(result["raw_cited"]),
        "raw_response": result.get("raw_response", ""),
        "refused": result["refused"],
        "retrieved": [
            {"rank": index, "source": chunk["source"], "text": chunk["text"]}
            for index, chunk in enumerate(retrieved, start=1)
        ],
        "top1_source": top1[0]["source"] if top1 else None,
        "keyword_hits": hits,
        "citation_all_required": citation["all_required"],
        "citation_no_extra": citation["no_extra"],
        "cited_docs": citation["credited"],
        "verdict": _verdict(row, result, retrieved),
    }


def score(rows: list[dict], ask) -> dict:
    """ask(question) -> (result, retrieved).

    Citation accuracy counts an answerable question only when every required source
    is cited and no cited chunk comes from outside it. Answer correctness counts one
    only when the answer contains every expected keyword. Both are keyword coverage,
    not semantic correctness. A row that fails still counts in the denominator.
    """
    citation_hits = refusal_hits = correct_hits = answerable = unanswerable = 0
    details = []
    for row in rows:
        result, retrieved = ask(row["question"])
        if row["answerable"]:
            answerable += 1
            citation = _scored_citation(row, result, retrieved)
            if citation["all_required"] and citation["no_extra"]:
                citation_hits += 1
            answer = result["answer"].lower()
            if all(keyword.lower() in answer for keyword in row["expected_keywords"]):
                correct_hits += 1
        else:
            unanswerable += 1
            if _refusal_ok(result, retrieved):
                refusal_hits += 1
        details.append(_detail(row, result, retrieved, []))
    return {
        "rows": len(rows),
        "citation_accuracy": round(citation_hits / answerable, 3) if answerable else None,
        "refusal_accuracy": round(refusal_hits / unanswerable, 3) if unanswerable else None,
        "answer_correctness": round(correct_hits / answerable, 3) if answerable else None,
        "details": details,
    }


def baseline(rows: list[dict], retrieve) -> float | None:
    """Share of single-source answerable questions whose top chunk is the expected document.

    Multi-source rows are excluded: one chunk cannot cover two documents, and that
    coverage is a separate metric. Returns None when no single-source row is answerable.
    """
    return _baseline_from([row for row in rows if row["answerable"] and "expected_source" in row],
                          lambda row: (retrieve(row["question"]) or [{}])[0].get("source"))


def _baseline_from(rows: list[dict], top1_source) -> float | None:
    hits = total = 0
    for row in rows:
        total += 1
        if top1_source(row) == row["expected_source"]:
            hits += 1
    return round(hits / total, 3) if total else None


DATASETS = {"easy": "rag_eval.jsonl", "hard": "rag_eval_hard.jsonl"}


class ReplayError(Exception):
    """A saved record cannot be recomputed from its raw fields."""


def replay(record: dict) -> dict:
    """Recompute the four headlines from a saved record, without calling the model.

    Every number comes from the raw fields: the answer text, the citation list the
    model returned, and the retrieved chunks' text and source. When a row carries
    `raw_cited`, the citations are recomputed from it, so editing the derived
    `cited_chunks` cannot hide an invalid citation. The stored verdict, citation
    flags, keyword hits and headline numbers are never read, so planting them false
    cannot move a score. A row missing a raw field, or a record whose dataset label
    does not match its set, raises ReplayError and yields no quartet.
    """
    expected = DATASETS.get(record["set"])
    if expected is None or record["dataset"][record["set"]] != expected:
        raise ReplayError(f"{record['set']} record names the wrong dataset file")
    answerable = unanswerable = 0
    citation_hits = refusal_hits = correct_hits = 0
    baseline_rows = []
    for row in record["rows"]:
        retrieved = [{"source": chunk["source"], "text": chunk["text"]} for chunk in row["retrieved"]]
        if row["answerable"]:
            answerable += 1
            citation = _scored_citation(row, row, retrieved)
            if citation["all_required"] and citation["no_extra"]:
                citation_hits += 1
            if all(keyword.lower() in row["answer"].lower() for keyword in row["expected_keywords"]):
                correct_hits += 1
            if row["expected_source"]:
                baseline_rows.append(row)
        else:
            unanswerable += 1
            if _refusal_ok(row, retrieved):
                refusal_hits += 1
    return {
        "citation_accuracy": round(citation_hits / answerable, 3) if answerable else None,
        "refusal_accuracy": round(refusal_hits / unanswerable, 3) if unanswerable else None,
        "answer_correctness": round(correct_hits / answerable, 3) if answerable else None,
        "retrieval_baseline": _baseline_from(baseline_rows, lambda row: row["top1_source"]),
    }


def _model_name(endpoint: str) -> str:
    import urllib.request
    url = endpoint.split("/chat/")[0] + "/models"
    with urllib.request.urlopen(url, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["data"][0]["id"]


def _revision() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def _write(name: str, rows: list[dict], report: dict, retrieve_one, meta: dict) -> Path:
    details = report["details"]
    for row, detail in zip(rows, details):
        top1 = retrieve_one(row["question"]) if row["answerable"] and "expected_source" in row else []
        detail["top1_source"] = top1[0]["source"] if top1 else None
    record = {
        **meta,
        "set": name,
        "n": report["rows"],
        "citation_accuracy": report["citation_accuracy"],
        "refusal_accuracy": report["refusal_accuracy"],
        "answer_correctness": report["answer_correctness"],
        "retrieval_baseline": baseline(rows, retrieve_one),
        "rows": details,
    }
    recomputed = replay(record)
    for key, value in recomputed.items():
        if record[key] != value:
            raise SystemExit(f"{name}: saved {key} {record[key]} does not replay to {value}")
    stamp = meta["run_utc"].replace("-", "").replace(":", "")
    out = Path(__file__).parent / "results" / f"{stamp}-{meta['model']}-{name}.json"
    if out.exists():
        raise SystemExit(f"{out.name} already exists; refusing to overwrite evidence")
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    documents = load_markdown_documents(PROJECT / "documents")
    chunks = chunk_documents(documents)

    easy = load_rows(Path(__file__).parent / "rag_eval.jsonl")
    hard = load_rows(Path(__file__).parent / "rag_eval_hard.jsonl")
    validate(easy)
    validate(hard)

    def ask(question: str) -> tuple[dict, list[dict]]:
        retrieved = retrieve_chunks(question, chunks, top_k=3)
        return synthesize(question, retrieved), retrieved

    endpoint = os.environ.get("OPERATOR_RAG_LLM_URL", DEFAULT_ENDPOINT)
    meta = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_revision": _revision(),
        "dataset": {"easy": "rag_eval.jsonl", "hard": "rag_eval_hard.jsonl"},
        "model": _model_name(endpoint),
        "endpoint": endpoint,
        "generation": {"temperature": 0, "top_k": 3},
    }
    retrieve_one = lambda question: retrieve_chunks(question, chunks, top_k=1)

    summaries = {}
    for name, rows in (("easy", easy), ("hard", hard)):
        out = _write(name, rows, score(rows, ask), retrieve_one, meta)
        summaries[name] = out.name
        print(f"wrote {out.relative_to(REPO).as_posix()}")

    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
