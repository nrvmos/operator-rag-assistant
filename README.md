# Operator RAG Assistant

**An assistant that answers questions from operating documents and shows the passage each answer came from. It refuses when the documents don't cover the question, or when someone asks for live operating advice.**

![Two recorded examples: a cited answer drawn from two documents, and a refused request to bypass an interlock](docs/examples.png)

## Why I built it

I'm Nathan Ramos. I ran chemical process units for more than ten years at NOVA Chemicals, CF Industries, INEOS Styrolution and GreenMantra, and I hold a TSSA 3rd Class Operating Engineer certificate. On shift, nobody acts on an answer they can't trace back to a procedure, and the worst tool is one that makes something up with confidence. So this assistant is built around two rules:

1. Every answer cites its source, and if the model cites a passage it wasn't given, the tool refuses rather than answering.
2. It refuses live operating requests ("which valve should I open right now", "how do I bypass the interlock") and questions the documents don't cover.

## How well it works

I tested it against a baseline with no AI in it, which just returns the best-matching passage, so the table shows what the language model actually adds.

| Question set | Questions | Cited the right source | Refused correctly | Answer had the key facts* | No-AI baseline: right document ranked first |
|---|---:|---:|---:|---:|---:|
| Easy | 20 | 93.3% | 100% (5 of 5) | 100% | 100% |
| Hard | 12 | 100% | n/a | 41.7% | 75% |

\*This counts whether the expected keywords appear in the answer. It doesn't judge whether the answer is right.

The first question set turned out to be too easy, because the baseline already found the right document every time. So I wrote a harder set with reworded questions, questions that need two documents, and near-miss distractors. On the hard set, the model cites the right source every time while the no-AI baseline ranks the right document first only 75% of the time. Part of that gap comes from the model seeing the top 3 passages instead of 1. The weak spot is the answers themselves: only 41.7% contain all the expected key facts, so the model often finds the right passage and then leaves part of the answer out. The baseline column measures finding the document, not answering, so it isn't directly comparable with the key-facts column. Every number here comes from a single run of one local model (27B, 4-bit, temperature 0) over three simulated documents. A replay script recomputes the figures from the saved raw outputs. The full method is in [projects/operator_rag/README.md](projects/operator_rag/README.md).

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/operator-rag-assistant-dark.png">
  <img alt="A question passes a safety rule check, TF-IDF search for the top 3 passages, a local LLM limited to those passages, and a citation check. Failing any gate refuses and cites nothing." src="docs/operator-rag-assistant.png">
</picture>

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,pdf]"
python -m pytest -q                                   # 36 tests; the LLM is mocked
python -m streamlit run projects/operator_rag/app.py
```

To score a live model, point `OPERATOR_RAG_LLM_URL` at any OpenAI-compatible chat endpoint and run `python projects/operator_rag/evals/run_eval.py`.

## How this was built

I wrote the code with AI coding agents working under my direction. I chose the problem and decided what counts as a pass. I designed the evals, including the baseline and the hard set, and every change had to pass its tests and an independent review before it landed. The domain judgement is mine: what an operator would trust, what the tool must refuse, and which limits have to sit beside each result.

## Limits

- The three documents are simulated study material, not real plant manuals.
- The question sets are small (20 and 12) and were scored on a single run.
- Retrieval is keyword-based (TF-IDF), so a question worded very differently from the document can miss.

There's no employer data, no confidential procedures and no live operating advice in this repo.

## Also by me

- [Plant fault detection on the Tennessee Eastman benchmark](https://github.com/nrvmos/plant-fault-detection): alarms judged by nuisance alarms per shift and minutes to detection.
