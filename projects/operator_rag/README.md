# Operator Knowledge RAG Assistant

This project demonstrates a citation-first assistant over public-safe study material and simulated industrial documents.

## Setup

Python 3.12 with the packages in `pyproject.toml` (scikit-learn, Streamlit). From the repo root:

```powershell
python -m pip install -e ".[dev,pdf]"
```

The corpus is the three markdown files in `projects/operator_rag/documents`.

## Demo

From the repo root:

```powershell
python -m streamlit run projects/operator_rag/app.py
```

Retrieval is TF-IDF with cosine similarity over 140-word chunks with a 30-word overlap, and the vectorizer is fit on the chunks and the question is transformed, so scores are not comparable across questions. The answer is synthesized by a local LLM from the retrieved chunks, and it names the source file. It refuses live operating requests and returns no match when nothing overlaps the question.

## Answer Synthesis

`synthesize.py` sends the retrieved chunks to an OpenAI-compatible chat endpoint and returns JSON with three fields: `answer`, `cited_chunks`, and `refused`. The endpoint URL comes from `OPERATOR_RAG_LLM_URL` and defaults to `http://127.0.0.1:8080/v1/chat/completions`. No key is read and none is stored in the repo. A citation the model invents, or a reply that cites nothing, comes back as a refusal rather than an ungrounded answer.

## Evaluation

`evals/rag_eval.jsonl` holds 20 questions over the simulated documents: 15 the corpus can answer, each naming the document that should be cited and one to three keywords the answer should contain, and 5 it cannot. `evals/rag_eval_hard.jsonl` holds 12 harder questions. Eight name one document. Four, h05 to h08, name two, and a citation passes only when both are cited and nothing outside them is. `evals/run_eval.py` reports three numbers. Citation accuracy is the share of answerable questions whose cited chunks cover every required document and no other. Refusal accuracy is the share of unanswerable questions the run refused. Answer correctness is the share of answerable questions whose answer contains every expected keyword. That is keyword coverage, not semantic correctness: a wrong answer that happens to contain the keyword still counts. It also reports a retrieval-only baseline: the share of single-source answerable questions whose top-ranked TF-IDF chunk comes from the expected document, with no model involved. Multi-source rows are left out of that baseline, because one chunk cannot cover two documents, and that coverage is a separate metric. A question the model cites badly fails and stays in the denominator. A malformed dataset stops the run before any number is emitted. Each run is written to `evals/results/` as one JSON file per set, with the cited documents, whether the run refused, and the verdict.

The unit tests mock the model, so `pytest` passes with no endpoint running. To score a live endpoint, from the repo root:

```powershell
$env:OPERATOR_RAG_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
python projects/operator_rag/evals/run_eval.py
```

## Results

Retrieval only, run on 2026-10-12 against the three bundled documents (3 documents, 6 chunks). The query "pump reliability indicators" returned three chunks:

| Rank | Score | Source |
|---|---:|---|
| 1 | 0.3065 | pump_reliability_study_note.md |
| 2 | 0.0441 | shift_handover_simulated_sop.md |
| 3 | 0.0430 | industrial_ai_guardrails.md |

The top excerpt returned was "This public-safe study note summarizes general pump reliability concepts." This shows that source text affects the output, and the two lower-ranked chunks are lexical matches that do not address the question. It is not an answer-quality evaluation.

Answer quality comes from one run of `evals/run_eval.py` at 2026-10-13T18:29:18Z against a local llama.cpp server at `http://127.0.0.1:8080/v1/chat/completions`. The model was Qwen3.6 27B; the server reported 27.3B parameters and Q4_K_S quantization. Generation is temperature 0 with the top 3 chunks. The records are `evals/results/20261013T182918Z-local-inference-easy.json` and `evals/results/20261013T182918Z-local-inference-hard.json`. Each row keeps the answer, the cited strings, and the retrieved chunks with source and rank, and `replay()` recomputes the headlines below from those raw fields alone. These records were captured after grounding, so a citation naming a chunk that was not retrieved was dropped rather than saved; later runs also keep the raw model response.

| Set | n | Citation accuracy | Refusal accuracy | Answer correctness | Retrieval baseline |
|---|---:|---:|---:|---:|---:|
| Easy | 20 | 93.3% | 100% | 100% | 100% |
| Hard | 12 | 100% | n/a | 41.7% | 75% |

The easy retrieval baseline is 100%, so the top-ranked chunk was already the expected document for every answerable question and the model added nothing on citation there. The hard set is where the metrics separate. The 75% retrieval baseline counts only the eight single-source rows, and it misses h09 and h10, whose gold file ranks second. Citation accuracy is 100% because the model cites from the top three chunks, which cover every required file, including both files on each of h05 to h08. So the 75% to 100% gap reflects the wider context as much as the model; a top-3 retrieval baseline would separate the two. Answer correctness is 41.7%, and that is keyword coverage, not semantic correctness.

h07 is the row that shows the difference between finding the file and finding the span. Its gold keywords stay `supervision` and `inventing`, and both are scored. Retrieval returns the guardrails file, but the returned text ends at "say so rather than", so it never contains `inventing`. The file was found and the answer span was missed. The keyword stays in the gold, so the miss counts against the score.

Limitation: each figure is a single run of one model. The easy set is 20 hand-written questions and the hard set is 12, all on three simulated documents. Answer correctness is keyword coverage, so it is not a measure of whether the answer is right. The numbers are not a general measure of the assistant.

## What It Shows

- Markdown document ingestion.
- Chunking and TF-IDF retrieval.
- Answer synthesis through a local LLM, with the cited chunks returned alongside the answer.
- Refusal behavior for unsafe live operating requests and for questions the documents do not support.

## Safety Boundary

This is not a live operating assistant. It is a study/reference demo using public-safe or simulated text only.
